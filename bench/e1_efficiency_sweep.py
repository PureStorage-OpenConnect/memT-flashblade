#!/usr/bin/env python
"""E1: end-to-end latency / query TTFT / state size vs history length (MemT paper Sec 6.5 + 6.6).

Systems (one model load; memory_state=None is exactly the Qwen3.6-35B-A3B backbone):
  nano    : history -> memory in chat-formatted <=2048-token user-message slices (commit_memory,
            document=True) = prep. Query = chat-templated user turn with the 512-token query
            content only, read from memory_state.
  fullctx : one chat-templated user turn = history + "\n\n" + query content. Prep = chunked
            prefill (8192) of [user-prefix + history] into a DynamicCache. Query = continue from
            the prepared cache with ["\n\n" + query content + end-of-user + assistant prefix].
Per query: restore prepared state from a (pinned) CPU copy -> GPU, prefill query, forced 128-token
greedy decode (EOS ignored). ttft_s includes restore (paper: "TTFT include[s] state restoration").

Example:
  CUDA_VISIBLE_DEVICES=0,1 HF_HOME=/mnt/data/robert-scripts/hf-cache HF_HUB_OFFLINE=1 \
    .venv/bin/python bench/e1_efficiency_sweep.py --histories 2 --lengths 512,2048 --reps 2
--histories: an integer = first K conversations; a comma list (e.g. "0,5," or "3,7") = indices.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import importlib.util
import json
import os
import platform
import socket
import sys
import time
import traceback
import uuid

# 256K chunked prefill reached ~80 GB reserved through fragmentation (allocator OOM-retry warnings);
# expandable segments avoid that. Must be set before torch initialises CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import beam_histories as bh  # noqa: E402

MODEL_ID = "MemTensor/MemT-1.0-Nano"
CHUNK_TOKENS = 2048
PREFILL_CHUNK = 8192
GEN_TOKENS = 128
IMPL_PREFILL, IMPL_DECODE = "grouped_mm", "batched_mm"
DEFAULT_LENGTHS = [512, 2048, 8192, 32768, 65536, 131072, 262144]


# ---------------------------------------------------------------- utils
def sync():
    for d in range(torch.cuda.device_count()):
        torch.cuda.synchronize(d)


def log(msg):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def mean(x):
    return sum(x) / len(x) if x else None


def peak_mem():
    return {f"cuda:{d}": round(torch.cuda.max_memory_allocated(d) / 2**30, 3) for d in range(torch.cuda.device_count())}


def reset_peak():
    for d in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(d)


class Impl:
    """Switch HF MoE experts kernel: grouped_mm for prefills, batched_mm for 1-token decode."""

    def __init__(self, model):
        self.model, self.cur = model, model.config._experts_implementation

    def set(self, impl):
        if impl != self.cur:
            self.model.set_experts_implementation(impl)
            self.cur = impl


# ---------------------------------------------------------------- cache <-> tensors
def cache_tensors(cache):
    from transformers.cache_utils import DynamicLayer, LinearAttentionCacheLayerMixin
    flat = {}
    for i, layer in enumerate(cache.layers):
        if isinstance(layer, LinearAttentionCacheLayerMixin):
            if layer.is_conv_states_initialized:
                flat[f"{i}.conv_states"] = layer.conv_states
            if layer.is_recurrent_states_initialized:
                flat[f"{i}.recurrent_states"] = layer.recurrent_states
        elif isinstance(layer, DynamicLayer):
            if getattr(layer, "is_initialized", False):
                flat[f"{i}.keys"] = layer.keys
                flat[f"{i}.values"] = layer.values
        else:
            raise TypeError(f"unexpected cache layer {type(layer)}")
    return flat


def cache_from_tensors(config, flat, devices):
    from transformers.cache_utils import DynamicCache, DynamicLayer, LinearAttentionCacheLayerMixin
    cache = DynamicCache(config=config)
    for i, layer in enumerate(cache.layers):
        dev = devices[i]
        if isinstance(layer, LinearAttentionCacheLayerMixin):
            layer.update_conv_state(flat[f"{i}.conv_states"].to(dev, non_blocking=True))
            layer.update_recurrent_state(flat[f"{i}.recurrent_states"].to(dev, non_blocking=True))
        elif isinstance(layer, DynamicLayer):
            k = flat[f"{i}.keys"].to(dev, non_blocking=True)
            v = flat[f"{i}.values"].to(dev, non_blocking=True)
            layer.lazy_initialization(k, v)
            layer.keys, layer.values = k, v
    return cache


def to_pinned_cpu(d):
    out = {}
    for k, v in d.items():
        c = torch.empty(v.shape, dtype=v.dtype, device="cpu", pin_memory=True)
        c.copy_(v, non_blocking=True)
        out[k] = c
    sync()
    return out


def nbytes(d):
    return sum(t.numel() * t.element_size() for t in d.values())


def serialized_bytes(d, metadata=None):
    from safetensors.torch import save
    return len(save({k: v.contiguous() for k, v in d.items()}, metadata=metadata))


# ---------------------------------------------------------------- core ops
class Runner:
    def __init__(self, model, tok):
        self.model, self.tok = model, tok
        self.impl = Impl(model)
        self.dev0 = model.get_input_embeddings().weight.device
        layers = model.model.language_model.layers
        self.layer_dev = {i: next(l.parameters()).device for i, l in enumerate(layers)}
        self.frame = bh.chat_frame(tok)

    def t(self, ids):
        return torch.tensor([ids], dtype=torch.long, device=self.dev0)

    @torch.no_grad()
    def nano_prep(self, slices):
        self.impl.set(IMPL_PREFILL)
        state, times = None, []
        for s in slices:
            x = self.t(s)
            sync()
            t0 = time.perf_counter()
            state = self.model.commit_memory(x, state, document=True)
            sync()
            times.append(time.perf_counter() - t0)
        return state, times

    @torch.no_grad()
    def fullctx_prep(self, ids, chunk=PREFILL_CHUNK):
        self.impl.set(IMPL_PREFILL)
        cache, times = None, []
        for s in range(0, len(ids), chunk):
            x = self.t(ids[s:s + chunk])
            sync()
            t0 = time.perf_counter()
            out = self.model(input_ids=x, past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = out.past_key_values
            del out
            sync()
            times.append(time.perf_counter() - t0)
        return cache, times

    @torch.no_grad()
    def query(self, q_ids, restore, n_new=GEN_TOKENS, memory=False):
        """restore() -> (memory_state or cache). Timed end to end, returns dict."""
        sync()
        t0 = time.perf_counter()
        st = restore()
        sync()
        t_r = time.perf_counter()
        kw = {"memory_state": st} if memory else {"past_key_values": st}
        self.impl.set(IMPL_PREFILL)
        out = self.model(input_ids=self.t(q_ids), use_cache=True, logits_to_keep=1, **kw)
        nxt = out.logits[:, -1].argmax(-1)
        first = nxt.item()
        t1 = time.perf_counter()
        cache = out.past_key_values
        del out
        self.impl.set(IMPL_DECODE)
        toks = [first]
        mem_kw = {"memory_state": st} if memory else {}
        for _ in range(n_new - 1):
            out = self.model(input_ids=nxt[:, None], past_key_values=cache, use_cache=True, logits_to_keep=1,
                             **mem_kw)
            nxt = out.logits[:, -1].argmax(-1)
            toks.append(nxt.item())
        sync()
        t2 = time.perf_counter()
        self.impl.set(IMPL_PREFILL)
        del cache, st
        return {"restore_s": t_r - t0, "ttft_s": t1 - t0, "prefill_s": t1 - t_r, "total_s": t2 - t0,
                "decode_tok_s": (n_new - 1) / (t2 - t1), "tokens": toks}

    def warmup(self, ids, q_content):
        """Throwaway runs so fla/Triton autotune+JIT and cuBLAS heuristics do not land in timed paths.
        Triton specialises kernels on whether int args are divisible by 16, so every path is run with
        both an aligned and an unaligned token count. Queries use the real query token ids."""
        log("warm-up: fresh/continued prefill, commit, memory+cache queries, decode (aligned+unaligned)")
        t0 = time.perf_counter()
        f = self.frame
        budget = CHUNK_TOKENS - len(f["prefix"]) - len(f["end_user"])
        sl_full = f["prefix"] + ids[:budget] + f["end_user"]
        sl_odd = f["prefix"] + ids[:509] + f["end_user"]
        state, _ = self.nano_prep([sl_full, sl_odd, sl_full])
        for n in (517, 2048, 2053):
            c, _ = self.fullctx_prep(ids[:n], chunk=PREFILL_CHUNK)
            del c
        cache, _ = self.fullctx_prep(ids[:2 * CHUNK_TOKENS + 522], chunk=CHUNK_TOKENS)  # fresh, cont aligned, cont odd
        cache2, _ = self.fullctx_prep(ids[:2 * CHUNK_TOKENS + 512], chunk=CHUNK_TOKENS)
        flats = [to_pinned_cpu(cache_tensors(c)) for c in (cache, cache2)]
        del cache, cache2
        sep = self.tok.encode(bh.SEP, add_special_tokens=False)
        q_nano = f["prefix"] + q_content + f["gen_suffix"]
        q_full = sep + q_content + f["gen_suffix"]
        for _ in range(2):
            self.query(q_nano, lambda: state, n_new=24, memory=True)
            for fl in flats:
                self.query(q_full, lambda: cache_from_tensors(self.model.config, fl, self.layer_dev), n_new=24)
        del state, flats
        gc.collect()
        torch.cuda.empty_cache()
        log(f"warm-up done in {time.perf_counter() - t0:.1f}s")


# ---------------------------------------------------------------- per-combo runs
def make_slices(R_or_frame, hist, sep, mode):
    frame = R_or_frame
    if mode == "dense":
        return bh.pack_slices_dense(hist, frame, CHUNK_TOKENS)
    return bh.pack_slices(hist, frame, sep, cap=CHUNK_TOKENS)


def run_nano(R, hist, q_content, reps, writer, slice_mode):
    f = R.frame
    slices = make_slices(f, hist, hist_sep(R), slice_mode)
    q_ids = f["prefix"] + q_content + f["gen_suffix"]
    reset_peak()
    state, slice_times = R.nano_prep(slices)
    prep_s = sum(slice_times)
    peak_prep = peak_mem()
    sync()
    t0 = time.perf_counter()
    cpu = to_pinned_cpu({k: v for k, v in state.items()})
    offload_s = time.perf_counter() - t0
    logical = nbytes(cpu)
    ser = serialized_bytes({f"memory.{k}": v for k, v in cpu.items()}, {"format": "memt-memory-v1"})
    numel = sum(v.numel() for v in cpu.values())
    del state
    torch.cuda.empty_cache()
    restore = lambda: {k: v.to(R.layer_dev[k], non_blocking=True) for k, v in cpu.items()}  # noqa: E731
    warm = R.query(q_ids, restore, memory=True)
    qs = [R.query(q_ids, restore, memory=True) for _ in range(reps)]
    rec = {
        "slices": len(slices), "slice_tokens_total": sum(len(s) for s in slices),
        "slice_tokens_max": max(len(s) for s in slices), "slice_times_s": [round(x, 4) for x in slice_times],
        "slice_mean_s": mean(slice_times), "prep_s": prep_s, "offload_s": offload_s,
        "state": {"logical_bytes": logical, "serialized_bytes": ser, "numel": numel,
                  "dtype": str(next(iter(cpu.values())).dtype), "layers": sorted(cpu)},
        "query_tokens": len(q_ids), "peak_gpu_GiB_prep": peak_prep,
    }
    return rec, warm, qs


def hist_sep(R):
    return R.tok.encode(bh.SEP, add_special_tokens=False)


def run_fullctx(R, hist, q_content, reps):
    f = R.frame
    sep = hist_sep(R)
    prep_ids = f["prefix"] + hist.ids
    q_ids = sep + q_content + f["gen_suffix"]
    reset_peak()
    cache, chunk_times = R.fullctx_prep(prep_ids)
    prep_s = sum(chunk_times)
    peak_prep = peak_mem()
    seq = cache.get_seq_length()
    sync()
    t0 = time.perf_counter()
    flat = cache_tensors(cache)
    cpu = to_pinned_cpu(flat)
    offload_s = time.perf_counter() - t0
    del flat, cache
    gc.collect()
    torch.cuda.empty_cache()
    kv = {k: v for k, v in cpu.items() if k.endswith((".keys", ".values"))}
    gdn = {k: v for k, v in cpu.items() if not k.endswith((".keys", ".values"))}
    rec_b = {k: v for k, v in gdn.items() if k.endswith(".recurrent_states")}
    conv_b = {k: v for k, v in gdn.items() if k.endswith(".conv_states")}
    ser = serialized_bytes(cpu, {"format": "hf-dynamiccache-hybrid-v0"})
    state = {
        "logical_bytes": nbytes(cpu), "serialized_bytes": ser,
        "kv_bytes": nbytes(kv), "kv_bytes_per_token": nbytes(kv) / seq,
        "gdn_recurrent_bytes": nbytes(rec_b), "gdn_conv_bytes": nbytes(conv_b),
        "gdn_dtype": str(next(iter(rec_b.values())).dtype),
        "gdn_recurrent_bytes_if_fp32": sum(v.numel() * 4 for v in rec_b.values()),
        "logical_bytes_fp32_gdn": nbytes(kv) + nbytes(conv_b) + sum(v.numel() * 4 for v in rec_b.values()),
        "note": "HF LinearAttentionLayer stores the GDN recurrent state in the conv-state dtype (BF16)",
    }
    restore = lambda: cache_from_tensors(R.model.config, cpu, R.layer_dev)  # noqa: E731
    warm = R.query(q_ids, restore)
    qs = [R.query(q_ids, restore) for _ in range(reps)]
    max_pos = R.model.config.get_text_config().max_position_embeddings
    rec = {
        "prefill_chunks": len(chunk_times), "chunk_times_s": [round(x, 4) for x in chunk_times],
        "prep_s": prep_s, "offload_s": offload_s, "cache_seq_len": seq, "prep_tokens": len(prep_ids),
        "state": state, "query_tokens": len(q_ids), "peak_gpu_GiB_prep": peak_prep,
        "exceeds_max_position": seq + len(q_ids) + GEN_TOKENS > max_pos, "max_position_embeddings": max_pos,
    }
    return rec, warm, qs


def summarize(qs, warm):
    keys = ["restore_s", "ttft_s", "prefill_s", "total_s", "decode_tok_s"]
    out = {k: [round(q[k], 5) for q in qs] for k in keys}
    out.update({f"{k}_mean": mean([q[k] for q in qs]) for k in keys})
    out["warmup_query"] = {k: round(warm[k], 5) for k in keys}
    out["first_tokens"] = qs[0]["tokens"][:16] if qs else None
    out["tokens_identical_across_reps"] = all(q["tokens"] == qs[0]["tokens"] for q in qs)
    return out


# ---------------------------------------------------------------- main
def parse_histories(s, n_avail):
    s = s.strip()
    if "," in s:
        return [int(x) for x in s.split(",") if x.strip()]
    return list(range(min(int(s), n_avail)))


def done_keys(path):
    keys = set()
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("status") == "ok":
                    keys.add(combo_key(r["history_idx"], r["N"], r["system"], r.get("writer"), r.get("slice_mode")))
    return keys


def combo_key(h, n, system, writer, slice_mode="messages"):
    return (h, n, system) + ((writer, slice_mode or "messages") if system == "nano" else ())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--histories", default="35")
    ap.add_argument("--lengths", default=",".join(map(str, DEFAULT_LENGTHS)))
    ap.add_argument("--systems", default="nano,fullctx")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--writer", choices=["reference", "fast"], default="fast")
    ap.add_argument("--out", default="/mnt/data/robert-scripts/m1t-test/results/e1_sweep.jsonl")
    ap.add_argument("--query-tokens", type=int, default=512)
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--slice-mode", choices=["messages", "dense"], default="messages",
                    help="nano slices: pack whole messages (default) or fill every slice to 2048 tokens")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-warmup", action="store_true")
    args = ap.parse_args()

    lengths = sorted(int(x) for x in args.lengths.split(","))
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    assert set(systems) <= {"nano", "fullctx"}, systems
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    run_id = dt.datetime.now().strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]

    fast = None
    if args.writer == "fast" and "nano" in systems:
        path = os.path.join(HERE, "memt_fast_writer.py")
        if not os.path.exists(path):
            sys.exit(f"--writer fast requested but {path} does not exist")
        spec = importlib.util.spec_from_file_location("memt_fast_writer", path)
        fast = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fast)
        if not hasattr(fast, "patch_model"):
            sys.exit(f"{path} has no patch_model(model)")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    convs = bh.load_beam()
    hidx = parse_histories(args.histories, len(convs))
    q_content = bh.query_content_ids(tok, args.query_tokens)
    log(f"histories={hidx} lengths={lengths} systems={systems} reps={args.reps} writer={args.writer}")

    t0 = time.perf_counter()
    tcs = {h: bh.tokenize_conversation(tok, convs[h][0], convs[h][1], int(max(lengths) * (1 + args.tol)) + 8)
           for h in hidx}
    log(f"tokenized {len(tcs)} conversations in {time.perf_counter() - t0:.1f}s")
    frame = bh.chat_frame(tok)
    sep = tok.encode(bh.SEP, add_special_tokens=False)

    todo, have = [], done_keys(args.out)
    for n in lengths:
        for h in hidx:
            hist = bh.build_history(tcs[h], n, args.tol)
            for s in systems:
                if combo_key(h, n, s, args.writer, args.slice_mode) in have:
                    continue
                todo.append((n, h, s, hist))
    log(f"{len(todo)} combos to run ({len(have)} already done in {args.out})")
    if args.dry_run:
        for n, h, s, hist in todo:
            extra = {}
            if s == "nano":
                sl = make_slices(frame, hist, sep, args.slice_mode)
                extra = {"slices": len(sl), "slice_tokens_max": max(map(len, sl))}
            print(json.dumps({"N": n, "history_idx": h, "conv_id": hist.conv_id, "system": s,
                              "history_tokens": hist.n_tokens, "n_records": hist.n_records,
                              "truncated_last": hist.truncated_last, **extra}))
        return

    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        log(f"WARNING: expected 2 visible GPUs, have {torch.cuda.device_count()}")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa", experts_implementation=IMPL_PREFILL).eval()
    log(f"model loaded in {time.perf_counter() - t0:.1f}s")
    if fast is not None:
        fast.patch_model(model)
        log("fast writer patched")
    R = Runner(model, tok)

    meta = {
        "run_id": run_id, "host": socket.gethostname(), "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "versions": {"torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__,
                     "fla": _ver("fla"), "python": platform.python_version()},
        "experts_impl": {"prefill": IMPL_PREFILL, "decode": IMPL_DECODE}, "attn": "sdpa",
        "gen_tokens": GEN_TOKENS, "chunk_tokens": CHUNK_TOKENS, "fullctx_prefill_chunk": PREFILL_CHUNK,
        "query_content_tokens": len(q_content), "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "slice_mode": args.slice_mode,
    }
    if not args.no_warmup:
        flat = [t for r in tcs[hidx[0]].records for t in r]
        while len(flat) < 8192:
            flat += flat
        R.warmup(flat[:8192], q_content)

    for n, h, s, hist in todo:
        log(f"N={n} hist={h} ({hist.conv_id}) system={s} tokens={hist.n_tokens}")
        base = {**meta, "timestamp": dt.datetime.now().isoformat(timespec="seconds"), "history_idx": h,
                "conv_id": hist.conv_id, "N": n, "history_tokens": hist.n_tokens, "n_records": hist.n_records,
                "truncated_last": hist.truncated_last, "system": s,
                "writer": args.writer if s == "nano" else None, "reps": args.reps,
                "slice_mode": args.slice_mode if s == "nano" else None,
                "writer_kind": (args.writer if s == "nano" else "n/a"), "fla_version": meta["versions"]["fla"]}
        try:
            t0 = time.perf_counter()
            if s == "nano":
                rec, warm, qs = run_nano(R, hist, q_content, args.reps, args.writer, args.slice_mode)
            else:
                rec, warm, qs = run_fullctx(R, hist, q_content, args.reps)
            summ = summarize(qs, warm)
            out = {**base, "status": "ok", **rec, **summ,
                   "e2e_s": rec["prep_s"] + summ["total_s_mean"],
                   "peak_gpu_GiB": peak_mem(), "wall_s": time.perf_counter() - t0}
            log(f"  prep={rec['prep_s']:.2f}s ttft={summ['ttft_s_mean'] * 1e3:.1f}ms (restore "
                f"{summ['restore_s_mean'] * 1e3:.1f}ms) total={summ['total_s_mean']:.2f}s "
                f"dec={summ['decode_tok_s_mean']:.2f}tok/s e2e={out['e2e_s']:.2f}s wall={out['wall_s']:.1f}s")
        except Exception as e:  # noqa: BLE001
            out = {**base, "status": "error", "error": repr(e)[:2000], "traceback": traceback.format_exc()[-4000:],
                   "peak_gpu_GiB": peak_mem()}
            log(f"  ERROR {e!r}")
        with open(args.out, "a") as fh:
            fh.write(json.dumps(out, default=str) + "\n")
        gc.collect()
        torch.cuda.empty_cache()
    if fast is not None and hasattr(fast, "unpatch_model"):
        fast.unpatch_model(model)
    log("done")


def _ver(mod):
    try:
        return __import__(mod).__version__
    except Exception:  # noqa: BLE001
        return None


if __name__ == "__main__":
    main()
