#!/usr/bin/env python
"""E4 part 2: resume a session from FlashBlade on the HF side (MemT-1.0-Nano, 2 GPUs).

For each (history, N) from bench/e4_build_ids.py (same token ids as the vLLM+pKVA part):
  nano    : commit the history into memory (dense <=2048-token chat slices, fast writer), save the memory
            state as a memt-memory-v1 safetensors snapshot (memory.<layer> FP32 [1,16,256,256]) on FlashBlade,
            fsync, free GPU/CPU copies. Resume = O_DIRECT read (host page cache bypassed) into a pooled,
            cudaHostRegister'd page-aligned buffer -> per-tensor H2D -> memory_state -> 512-token query ->
            128 forced greedy tokens. Also a labeled page-cache-warm variant (buffered read after a priming read).
  fullctx : chunked prefill (8192) of [user prefix + history] into a DynamicCache (attention KV BF16 + GDN conv/
            recurrent state, BF16 in HF), serialized as safetensors on FlashBlade, freed, then restored the same
            way (O_DIRECT -> pinned -> H2D -> DynamicCache) and queried with ["\n\n" + query + gen suffix].
Stage timings (CUDA-synced): read_s (file -> host), h2d_s, prefill_s (query prefill to first token),
ttft_s = read + h2d + prefill, total_s (to the 128th token). Pin/alloc of the host buffer is pooled (done once
per file size, outside the timed region) and reported as alloc_pin_s.

Example:
  CUDA_VISIBLE_DEVICES=4,5 HF_HOME=/mnt/data/robert-scripts/hf-cache HF_HUB_OFFLINE=1 \
    .venv/bin/python bench/e4_memt_resume.py --systems nano,fullctx
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import os
import platform
import socket
import struct
import sys
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e1_efficiency_sweep as e1  # noqa: E402  (sets PYTORCH_CUDA_ALLOC_CONF before torch import)
import e3_persist_io as io  # noqa: E402
import beam_histories as bh  # noqa: E402
import torch  # noqa: E402

log = e1.log
sync = e1.sync
DT = {"F32": torch.float32, "BF16": torch.bfloat16, "F16": torch.float16, "I64": torch.int64}


class HostBuf:
    """Pooled page-aligned host buffer, pinned in place with cudaHostRegister (re-used across samples)."""

    def __init__(self):
        self.m, self.t, self.size = None, None, 0

    def get(self, nbytes):
        info = {"alloc_pin_s": 0.0, "realloc": False}
        if nbytes > self.size:
            self.release()
            t0 = time.perf_counter()
            self.m = io.aligned_buffer(nbytes)
            self.t = torch.frombuffer(self.m, dtype=torch.uint8)
            rc = torch.cuda.cudart().cudaHostRegister(self.t.data_ptr(), self.t.numel(), 0)
            assert int(rc) == 0, f"cudaHostRegister failed rc={rc}"
            self.size = len(self.m)
            info = {"alloc_pin_s": time.perf_counter() - t0, "realloc": True}
        return info

    def release(self):
        if self.t is not None:
            torch.cuda.cudart().cudaHostUnregister(self.t.data_ptr())
            del self.t
            self.t = None
            self.m.close()
            self.m, self.size = None, 0


def write_snapshot(path, tensors: dict, metadata: dict):
    """CPU tensors -> safetensors bytes -> write + fsync. Returns timings and bytes."""
    from safetensors.torch import save
    t0 = time.perf_counter()
    blob = save({k: v.contiguous() for k, v in tensors.items()}, metadata=metadata)
    t1 = time.perf_counter()
    w = io.write_file(path, memoryview(blob), 64 * io.MiB, do_fsync=True)
    return {"serialize_s": t1 - t0, **{f"file_{k}": v for k, v in w.items()}, "file_bytes": len(blob),
            "path": path}


def read_into(hb: HostBuf, path, direct, chunk_mib, io_threads):
    nbytes = os.path.getsize(path)
    pin = hb.get(nbytes)
    t0 = time.perf_counter()
    if direct:
        got, tail = io.read_direct(path, hb.m, nbytes, chunk_mib * io.MiB, io_threads)
    else:
        got, tail = io.read_buffered(path, hb.m, nbytes, chunk_mib * io.MiB), "buffered"
    t1 = time.perf_counter()
    assert got == nbytes, (got, nbytes)
    return nbytes, t1 - t0, tail, pin


def views(hb: HostBuf, nbytes):
    """Typed CPU views (on the pinned buffer) for every tensor in the safetensors image."""
    (hlen,) = struct.unpack("<Q", bytes(memoryview(hb.m)[:8]))
    hdr = json.loads(bytes(memoryview(hb.m)[8:8 + hlen]))
    meta = hdr.pop("__metadata__", {})
    base = 8 + hlen
    out = {}
    for k, v in hdr.items():
        a, b = v["data_offsets"]
        out[k] = hb.t[base + a:base + b].view(DT[v["dtype"]]).view(v["shape"])
    return out, meta


class Resumer:
    def __init__(self, R, args):
        self.R, self.a = R, args
        self.hb = HostBuf()

    def restore_fn(self, path, kind, direct, stash):
        """Returns a restore() closure for Runner.query; stage timings go into stash."""
        R = self.R

        def restore():
            sync()
            nbytes, read_s, tail, pin = read_into(self.hb, path, direct, self.a.chunk_mib, self.a.io_threads)
            t1 = time.perf_counter()
            vs, meta = views(self.hb, nbytes)
            if kind == "nano":
                st = {int(k.split(".", 1)[1]): v.to(R.layer_dev[int(k.split(".", 1)[1])], non_blocking=True)
                      for k, v in vs.items()}
            else:
                st = e1.cache_from_tensors(R.model.config, vs, R.layer_dev)
            sync()
            t2 = time.perf_counter()
            stash.update({"bytes": nbytes, "read_s": read_s, "h2d_s": t2 - t1, "tail_mode": tail,
                          "host_pinned": bool(self.hb.t.is_pinned()), **pin,
                          "read_gbps": nbytes / read_s / 1e9, "h2d_gbps": nbytes / (t2 - t1) / 1e9})
            return st
        return restore

    def sample(self, path, kind, q_ids, direct, n_new=e1.GEN_TOKENS):
        pool = self.hb.get(os.path.getsize(path))  # pooled pinned buffer: (re)allocated outside the timed region
        if not direct:  # page-cache-warm variant: prime with an untimed buffered read
            read_into(self.hb, path, False, self.a.chunk_mib, 1)
        stash = {"pool_alloc_pin_s": pool["alloc_pin_s"], "pool_realloc": pool["realloc"]}
        q = self.R.query(q_ids, self.restore_fn(path, kind, direct, stash), n_new=n_new, memory=(kind == "nano"))
        return {**stash, "restore_s": q["restore_s"], "prefill_s": q["prefill_s"], "ttft_s": q["ttft_s"],
                "total_s": q["total_s"], "decode_tok_s": q["decode_tok_s"], "tokens": q["tokens"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", default="/mnt/data/robert-scripts/m1t-test/scratch_state/e4/e4_ids.json")
    ap.add_argument("--out", default="/mnt/data/robert-scripts/m1t-test/results/e4_memt_resume.jsonl")
    ap.add_argument("--state-dir", default="/mnt/data/robert-scripts/m1t-test/scratch_state/e4")
    ap.add_argument("--systems", default="nano,fullctx")
    ap.add_argument("--only", default="", help="comma list of <h>_<N> keys")
    ap.add_argument("--fullctx-max-n", type=int, default=262144)
    ap.add_argument("--nano-cold", type=int, default=3)
    ap.add_argument("--nano-warm", type=int, default=2)
    ap.add_argument("--fullctx-cold", type=int, default=2)
    ap.add_argument("--chunk-mib", type=int, default=64)
    ap.add_argument("--io-threads", type=int, default=1)
    ap.add_argument("--keep-fullctx-files", action="store_true",
                    help="default deletes fullctx KV files after their samples (bytes are recorded)")
    ap.add_argument("--no-warmup", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.state_dir, exist_ok=True)
    systems = [s for s in a.systems.split(",") if s]
    D = json.load(open(a.ids))
    keys = [k for k in D["histories"] if not a.only or k in a.only.split(",")]
    run_id = dt.datetime.now().strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers
    import importlib.util
    tok = AutoTokenizer.from_pretrained(e1.MODEL_ID, trust_remote_code=True)
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        e1.MODEL_ID, trust_remote_code=True, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa", experts_implementation=e1.IMPL_PREFILL).eval()
    log(f"model loaded in {time.perf_counter() - t0:.1f}s")
    spec = importlib.util.spec_from_file_location("memt_fast_writer", os.path.join(HERE, "memt_fast_writer.py"))
    fast = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fast)
    fast.patch_model(model)
    R = e1.Runner(model, tok)
    frame = D["frame"]
    assert frame["prefix"] == R.frame["prefix"] and frame["gen_suffix"] == R.frame["gen_suffix"]
    sep, q = D["sep"], D["q_content"]
    q_nano = frame["prefix"] + q + frame["gen_suffix"]
    q_full = sep + q + frame["gen_suffix"]
    if not a.no_warmup:
        h0 = D["histories"][keys[0]]["hist_ids"]
        flat = list(h0)
        while len(flat) < 8192:
            flat += flat
        R.warmup(flat[:8192], q)
    meta = {"run_id": run_id, "experiment": "e4_memt_resume", "host": socket.gethostname(),
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "versions": {"torch": torch.__version__, "cuda": torch.version.cuda,
                         "transformers": transformers.__version__, "fla": e1._ver("fla"),
                         "python": platform.python_version()},
            "model": e1.MODEL_ID, "writer": "fast", "slice_mode": "dense",
            "experts_impl": {"prefill": e1.IMPL_PREFILL, "query_prefill": e1.IMPL_QUERY, "decode": e1.IMPL_DECODE},
            "gen_tokens": e1.GEN_TOKENS, "fullctx_prefill_chunk": e1.PREFILL_CHUNK,
            "io": {"chunk_mib": a.chunk_mib, "io_threads": a.io_threads, "host_buffer": "mmap+cudaHostRegister pooled"},
            "storage": "FlashBlade NFSv3 over RDMA (/mnt/data, nconnect=16)", "ids_file": a.ids}
    fout = open(a.out, "a")

    def emit(rec):
        fout.write(json.dumps({**meta, "ts": dt.datetime.now().isoformat(timespec="seconds"), **rec},
                              default=str) + "\n")
        fout.flush()

    RS = Resumer(R, a)
    for key in keys:
        H = D["histories"][key]
        hist = bh.History(H["conv_id"], H["N"], H["hist_ids"])
        base = {"history_idx": H["history_idx"], "conv_id": H["conv_id"], "N": H["N"],
                "history_tokens": H["history_tokens"]}
        for s in systems:
            if s == "fullctx" and H["N"] > a.fullctx_max_n:
                log(f"skip fullctx {key} (> --fullctx-max-n)")
                continue
            log(f"== {key} {s}: history {H['history_tokens']}")
            try:
                e1.reset_peak()
                if s == "nano":
                    slices = bh.pack_slices_dense(hist, R.frame, e1.CHUNK_TOKENS)
                    state, times = R.nano_prep(slices)
                    prep = {"slices": len(slices), "prep_s": sum(times)}
                    ref = R.query(q_nano, lambda: state, n_new=8, memory=True)["tokens"]
                    sync()
                    t0 = time.perf_counter()
                    cpu = {f"memory.{k}": v.detach().to("cpu") for k, v in state.items()}
                    d2h_s = time.perf_counter() - t0
                    path = os.path.join(a.state_dir, f"{H['history_idx']}_{H['N']}.safetensors")
                    smeta = {"format": "memt-memory-v1", "model": e1.MODEL_ID, "conv_id": str(H["conv_id"]),
                             "history_tokens": str(H["history_tokens"])}
                    q_ids = q_nano
                    del state
                else:
                    cache, times = R.fullctx_prep(frame["prefix"] + H["hist_ids"])
                    prep = {"prefill_chunks": len(times), "prep_s": sum(times), "cache_seq_len": cache.get_seq_length()}
                    sync()
                    t0 = time.perf_counter()
                    cpu = {k: v.detach().to("cpu") for k, v in e1.cache_tensors(cache).items()}
                    d2h_s = time.perf_counter() - t0
                    del cache
                    gc.collect()
                    torch.cuda.empty_cache()
                    ref = R.query(q_full, lambda: e1.cache_from_tensors(R.model.config, cpu, R.layer_dev),
                                  n_new=8)["tokens"]
                    path = os.path.join(a.state_dir, f"fullctx_{H['history_idx']}_{H['N']}.safetensors")
                    smeta = {"format": "hf-dynamiccache-hybrid-v0", "model": e1.MODEL_ID, "conv_id": str(H["conv_id"]),
                             "history_tokens": str(H["history_tokens"])}
                    q_ids = q_full
                kinds = {}
                for k, v in cpu.items():
                    kk = k.rsplit(".", 1)[-1]
                    kinds.setdefault(kk, [0, str(v.dtype)])
                    kinds[kk][0] += v.numel() * v.element_size()
                gc.collect()
                torch.cuda.empty_cache()
                w = write_snapshot(path, cpu, smeta)
                del cpu
                gc.collect()
                emit({**base, "system": s, "label": "save", **prep, "d2h_s": d2h_s, **w,
                      "bytes_by_kind": kinds, "reference_tokens": ref, "peak_gpu_GiB": e1.peak_mem(),
                      "note": "HF GDN recurrent state is BF16 (pKVA checkpoints it at ~65 MB)" if s == "fullctx" else None})
                log(f"  prep {prep['prep_s']:.1f}s, saved {w['file_bytes']} B in {w['file_total_s']:.2f}s "
                    f"(fsync {w['file_fsync_s']:.2f}s)")
                plan = [("odirect", True, a.nano_cold if s == "nano" else a.fullctx_cold)]
                if s == "nano":
                    plan.append(("pagecache_warm", False, a.nano_warm))
                for label, direct, n in plan:
                    for i in range(n):
                        r = RS.sample(path, s, q_ids, direct)
                        toks = r.pop("tokens")
                        emit({**base, "system": s, "label": label, "sample": i, **r,
                              "first_tokens": toks[:16], "matches_reference_8": toks[:8] == ref,
                              "query_tokens": len(q_ids)})
                        log(f"  {label}{i}: ttft={r['ttft_s'] * 1e3:.1f}ms (read {r['read_s'] * 1e3:.1f} "
                            f"h2d {r['h2d_s'] * 1e3:.1f} prefill {r['prefill_s'] * 1e3:.1f}) total={r['total_s']:.2f}s "
                            f"match={toks[:8] == ref}")
                if s == "fullctx" and not a.keep_fullctx_files:
                    os.unlink(path)
            except Exception as e:  # noqa: BLE001
                import traceback
                emit({**base, "system": s, "label": "error", "error": repr(e)[:2000],
                      "traceback": traceback.format_exc()[-4000:]})
                log(f"  ERROR {e!r}")
            gc.collect()
            torch.cuda.empty_cache()
    RS.hb.release()
    log("done")


if __name__ == "__main__":
    main()
