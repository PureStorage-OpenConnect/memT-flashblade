#!/usr/bin/env python
"""Smoke test for MemT-1.0-Nano under HF transformers (memory mode + full-context mode).

Run with:  CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python bench/smoke_memt_hf.py
Every result line is a JSON object on stdout ("evt" field names the record).
"""
import argparse
import json
import os
import random
import subprocess
import time

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

MODEL_ID = "MemTensor/MemT-1.0-Nano"
CHUNK = 2048
# HF MoE experts kernel per phase. On A100 (sm80) torch's grouped_mm loops over all 256 experts on the
# host (~20k aten::mm + .item() per decode step), so decode uses batched_mm; prefill keeps grouped_mm
# (batched_mm gathers one weight copy per token*topk and would explode memory on long prefills).
# Forwards with <= SMALL_FWD tokens (decode steps, short query prompts) use IMPL["decode"].
IMPL = {"prefill": "grouped_mm", "decode": "batched_mm"}
SMALL_FWD = 256
_cur_impl = [None]


def use_impl(model, n_tokens):
    want = IMPL["decode"] if n_tokens <= SMALL_FWD else IMPL["prefill"]
    if want != _cur_impl[0]:
        model.set_experts_implementation(want)
        _cur_impl[0] = want


def emit(evt, **kw):
    print(json.dumps({"evt": evt, **kw}, default=str), flush=True)


def sync():
    for d in range(torch.cuda.device_count()):
        torch.cuda.synchronize(d)


def gpu_mem():
    out = {}
    for d in range(torch.cuda.device_count()):
        out[f"cuda:{d}"] = {
            "alloc_GiB": round(torch.cuda.memory_allocated(d) / 2**30, 3),
            "reserved_GiB": round(torch.cuda.memory_reserved(d) / 2**30, 3),
            "max_alloc_GiB": round(torch.cuda.max_memory_allocated(d) / 2**30, 3),
        }
    return out


def nvsmi():
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
    except Exception as e:  # noqa: BLE001
        return [repr(e)]


# ---------------------------------------------------------------- synthetic history
FIRST = ["Alder", "Brisa", "Corvin", "Dalia", "Emrys", "Fennick", "Galen", "Halvard", "Ione", "Jorund",
         "Kestrel", "Liora", "Maelis", "Nerys", "Oswin", "Perrin", "Quilla", "Rowan", "Sabine", "Tobiah",
         "Ulla", "Varek", "Wren", "Xanthe", "Yorick", "Zelda"]
LAST = ["Ashgrove", "Blackthorn", "Copperfield", "Dunmore", "Everhart", "Foxworth", "Greywater",
        "Hollowell", "Ironwood", "Juniper", "Kingsley", "Larkspur", "Merriweather", "Northcott",
        "Oakhaven", "Pembrook", "Quenby", "Ravenscroft", "Stormvale", "Thistlewood"]
CITIES = ["Velmora", "Ostrakand", "Pellucid Bay", "Haverlock", "Quintara", "Sorrowmere", "Drennick",
          "Calloway Reach", "Tamsin Falls", "Brightwater", "Ulmengard", "Kestrelholt"]
ITEMS = ["telescopes", "copper kettles", "violins", "maps", "lanterns", "beehives", "clocks",
         "sailcloths", "porcelain cups", "chess sets", "quills", "barometers"]


def build_facts(n, seed=1234):
    rng = random.Random(seed)
    names = [f"{f} {l}" for f in FIRST for l in LAST]
    rng.shuffle(names)
    facts = []
    for i in range(n):
        name = names[i]
        num = rng.randint(1000, 9999)
        city = rng.choice(CITIES)
        item = rng.choice(ITEMS)
        facts.append({
            "name": name, "num": num, "city": city, "item": item,
            "text": f"Record {i + 1}: {name} lives in {city} and owns exactly {num} {item}.",
        })
    return facts


def pack_documents(tok, facts, cap=CHUNK):
    """Greedily pack fact sentences into user messages whose chat-formatted length <= cap tokens."""
    docs, cur = [], []

    def n_tokens(lines):
        text = tok.apply_chat_template([{"role": "user", "content": "\n".join(lines)}], tokenize=False)
        return len(tok.encode(text, add_special_tokens=False))

    for f in facts:
        if cur and n_tokens(cur + [f["text"]]) > cap:
            docs.append(cur)
            cur = []
        cur.append(f["text"])
    if cur:
        docs.append(cur)
    out = []
    for lines in docs:
        text = tok.apply_chat_template([{"role": "user", "content": "\n".join(lines)}], tokenize=False)
        out.append(tok.encode(text, add_special_tokens=False))
    return out


def query_ids(tok, question):
    text = tok.apply_chat_template([{"role": "user", "content": question}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    return tok.encode(text, add_special_tokens=False)


# ---------------------------------------------------------------- generation helpers
@torch.no_grad()
def greedy(model, ids, n_new, memory_state=None, cache=None, eos=()):
    """Manual greedy loop. Returns (tokens, ttft_s, decode_tok_per_s). Ignores EOS (forced length)."""
    dev = model.get_input_embeddings().weight.device
    inp = torch.tensor([ids], dtype=torch.long, device=dev)
    use_impl(model, len(ids))
    sync()
    t0 = time.perf_counter()
    out = model(input_ids=inp, past_key_values=cache, use_cache=True,
                memory_state=memory_state or None, logits_to_keep=1)
    nxt = out.logits[:, -1].argmax(-1)
    first = nxt.item()  # forces sync
    t1 = time.perf_counter()
    use_impl(model, 1)
    cache = out.past_key_values
    toks = [first]
    for _ in range(n_new - 1):
        out = model(input_ids=nxt[:, None], past_key_values=cache, use_cache=True,
                    memory_state=memory_state or None, logits_to_keep=1)
        nxt = out.logits[:, -1].argmax(-1)
        toks.append(nxt.item())
    t2 = time.perf_counter()
    return toks, t1 - t0, (n_new - 1) / (t2 - t1), cache


def answer_text(tok, toks, eos):
    cut = toks
    for i, t in enumerate(toks):
        if t in eos:
            cut = toks[:i]
            break
    return tok.decode(cut, skip_special_tokens=True).strip()


# ---------------------------------------------------------------- cache (de)serialisation
def cache_tensors(cache):
    """Flatten a hybrid DynamicCache (DynamicLayer + LinearAttentionLayer) into a name->tensor dict."""
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
    """Rebuild a DynamicCache from flat tensors; devices maps layer idx -> torch.device."""
    from transformers.cache_utils import DynamicCache, DynamicLayer, LinearAttentionCacheLayerMixin
    cache = DynamicCache(config=config)
    for i, layer in enumerate(cache.layers):
        dev = devices[i]
        if isinstance(layer, LinearAttentionCacheLayerMixin):
            conv = flat[f"{i}.conv_states"].to(dev)
            rec = flat[f"{i}.recurrent_states"].to(dev)
            layer.update_conv_state(conv)       # sets has_previous_state=True
            layer.update_recurrent_state(rec)
        elif isinstance(layer, DynamicLayer):
            k = flat[f"{i}.keys"].to(dev)
            v = flat[f"{i}.values"].to(dev)
            layer.lazy_initialization(k, v)
            layer.keys, layer.values = k, v
    return cache


def tensor_bytes(d):
    return sum(t.numel() * t.element_size() for t in d.values())


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--n-facts", type=int, default=240)
    ap.add_argument("--gen-tokens", type=int, default=128)
    ap.add_argument("--state-dir", default="/mnt/data/robert-scripts/m1t-test/scratch_state")
    ap.add_argument("--max-mem", default="75GiB")
    ap.add_argument("--prefill-experts", default="grouped_mm")
    ap.add_argument("--decode-experts", default="batched_mm")
    ap.add_argument("--skip-fullctx", action="store_true")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers
    emit("env", torch=torch.__version__, cuda=torch.version.cuda, transformers=transformers.__version__,
         visible=os.environ.get("CUDA_VISIBLE_DEVICES"), n_gpu=torch.cuda.device_count(),
         gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
         nvidia_smi=nvsmi())
    try:
        import fla
        emit("kernels", fla=fla.__version__)
    except Exception as e:  # noqa: BLE001
        emit("kernels", fla=None, err=repr(e))

    # (a) load -------------------------------------------------------------------
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    IMPL.update(prefill=args.prefill_experts, decode=args.decode_experts)
    kw = {"experts_implementation": args.prefill_experts}
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16, device_map="auto",
        max_memory={i: args.max_mem for i in range(torch.cuda.device_count())},
        attn_implementation="sdpa", **kw)
    model.eval()
    _cur_impl[0] = model.config._experts_implementation
    sync()
    load_s = time.perf_counter() - t0
    layers = model.model.language_model.layers
    layer_dev = {i: next(l.parameters()).device for i, l in enumerate(layers)}
    split = {}
    for i, d in layer_dev.items():
        split.setdefault(str(d), []).append(i)
    emit("load", seconds=round(load_s, 1), attn=model.config._attn_implementation,
         experts=getattr(model.config, "_experts_implementation", None), decode_experts=IMPL["decode"],
         layer_split={k: [v[0], v[-1], len(v)] for k, v in split.items()},
         embed_dev=str(model.get_input_embeddings().weight.device), lm_head_dev=str(model.lm_head.weight.device),
         gpu=gpu_mem(), nvidia_smi=nvsmi())

    text_cfg = model.config.get_text_config()
    mem_layers = [i for i, t in enumerate(text_cfg.layer_types) if t == "full_attention"]
    eos = set(model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, list)
              else [model.generation_config.eos_token_id])

    # (b) history -> memory in <=2048-token chat-formatted slices ----------------
    facts = build_facts(args.n_facts)
    docs = pack_documents(tok, facts)
    total = sum(len(d) for d in docs)
    emit("history", n_facts=len(facts), n_slices=len(docs), slice_tokens=[len(d) for d in docs], total_tokens=total)

    dev0 = model.get_input_embeddings().weight.device
    state = None
    slice_times = []
    torch.cuda.reset_peak_memory_stats()
    for j, d in enumerate(docs):
        ids = torch.tensor([d], dtype=torch.long, device=dev0)
        sync()
        t = time.perf_counter()
        use_impl(model, len(d))
        state = model.commit_memory(ids, state, document=True)
        sync()
        dt = time.perf_counter() - t
        slice_times.append(dt)
        emit("mem_write_slice", idx=j, tokens=len(d), seconds=round(dt, 3), tok_per_s=round(len(d) / dt, 1))
    # isolate the memory-write overhead: same slice, plain backbone prefill (no memory), and read-only memory
    use_impl(model, 10**9)
    for label, kwargs in [("plain_prefill", {}), ("read_only_prefill", {"memory_state": state})]:
        ids = torch.tensor([docs[1]], dtype=torch.long, device=dev0)
        with torch.no_grad():
            model(input_ids=ids, use_cache=True, logits_to_keep=1, **kwargs)
            sync()
            t = time.perf_counter()
            model(input_ids=ids, use_cache=True, logits_to_keep=1, **kwargs)
            sync()
        emit("slice_compare", kind=label, tokens=len(docs[1]), seconds=round(time.perf_counter() - t, 3))
    emit("mem_write_total", seconds=round(sum(slice_times), 3), tokens=total, gpu=gpu_mem())

    # (c) questions answered from memory ----------------------------------------
    picks = [facts[5], facts[len(facts) // 2], facts[-7]]
    questions = [
        (f"How many {p['item']} does {p['name']} own? Answer with the number only.", str(p["num"]))
        for p in picks[:2]
    ] + [(f"In which city does {picks[2]['name']} live? Answer with the city name only.", picks[2]["city"])]

    # warm-up query (kernel autotune / first-call overhead), not reported as TTFT
    greedy(model, query_ids(tok, "Say hello."), 4, memory_state=state)

    def ask_all(st, tag):
        res = []
        for q, gold in questions:
            toks, ttft, tps, _ = greedy(model, query_ids(tok, q), args.gen_tokens, memory_state=st)
            ans = answer_text(tok, toks, eos)
            ok = gold.lower() in ans.lower()
            res.append(ok)
            emit("query", mode=tag, question=q, gold=gold, answer=ans[:200], correct=ok,
                 prompt_tokens=len(query_ids(tok, q)), ttft_s=round(ttft, 4), decode_tok_per_s=round(tps, 2),
                 gen_tokens=args.gen_tokens)
        return res

    ask_all(state, "memory")
    # Same question with memory read OFF (sanity: should not know the answer)
    toks, ttft, tps, _ = greedy(model, query_ids(tok, questions[0][0]), 32, memory_state=None)
    emit("query", mode="no_memory_control", question=questions[0][0], gold=questions[0][1],
         answer=answer_text(tok, toks, eos)[:200], ttft_s=round(ttft, 4))

    # HF generate() path with memory_state (as MemTSession uses it)
    q_ids = torch.tensor([query_ids(tok, questions[0][0])], device=dev0)
    use_impl(model, 1)
    sync()
    t = time.perf_counter()
    g = model.generate(q_ids, attention_mask=torch.ones_like(q_ids), memory_state=state, max_new_tokens=32,
                       do_sample=False)
    sync()
    emit("generate_api", answer=tok.decode(g[0, q_ids.shape[1]:], skip_special_tokens=True).strip()[:200],
         seconds=round(time.perf_counter() - t, 3))

    # (d) extract / save / restore memory state ------------------------------------
    numel = sum(t.numel() for t in state.values())
    nbytes = sum(t.numel() * t.element_size() for t in state.values())
    emit("mem_state", layers=sorted(state), shape=list(next(iter(state.values())).shape),
         dtype=str(next(iter(state.values())).dtype), numel=numel, bytes=nbytes, MiB=nbytes / 2**20,
         devices={k: str(v.device) for k, v in state.items()})

    os.makedirs(args.state_dir, exist_ok=True)
    sess = model.create_session(tok)
    identity = sess._state_identity()
    path = os.path.join(args.state_dir, "smoke_memt_state.safetensors")
    tensors = {f"memory.{k}": v.detach().to("cpu", torch.float32).contiguous() for k, v in state.items()}
    t = time.perf_counter()
    save_file(tensors, path, metadata={"format": "memt-memory-v1", "identity": identity})
    with open(path, "rb") as fh:
        os.fsync(fh.fileno())
    save_s = time.perf_counter() - t
    emit("mem_save", path=path, file_bytes=os.path.getsize(path), seconds=round(save_s, 4))

    # fresh state: drop everything and reload from FlashBlade
    del state, tensors
    torch.cuda.empty_cache()
    sync()
    t = time.perf_counter()
    loaded = load_file(path, device="cpu")
    restored = {int(k.split(".")[1]): v.to(layer_dev[int(k.split(".")[1])]) for k, v in loaded.items()}
    sync()
    restore_s = time.perf_counter() - t
    emit("mem_restore", seconds=round(restore_s, 4), n=len(restored))
    q, gold = questions[1]
    toks, ttft, tps, _ = greedy(model, query_ids(tok, q), args.gen_tokens, memory_state=restored)
    ans = answer_text(tok, toks, eos)
    emit("query", mode="memory_restored", question=q, gold=gold, answer=ans[:200], correct=gold.lower() in ans.lower(),
         ttft_s=round(ttft, 4), ttft_plus_restore_s=round(ttft + restore_s, 4), decode_tok_per_s=round(tps, 2))
    # also via the official session loader (validates metadata/identity/shape)
    use_impl(model, 1)
    s2 = model.create_session(tok, state=path)
    a2 = s2.chat(questions[2][0], enable_thinking=False, do_sample=False, max_new_tokens=32)
    emit("session_restore", answer=a2.strip()[:200], gold=questions[2][1], correct=questions[2][1].lower() in a2.lower())
    s2.close()
    del restored
    torch.cuda.empty_cache()

    if args.skip_fullctx:
        emit("done")
        return

    # (e) full-context baseline -------------------------------------------------------
    hist_text = tok.apply_chat_template([{"role": "user", "content": "\n".join(f["text"] for f in facts)}],
                                        tokenize=False)
    hist_ids = tok.encode(hist_text, add_special_tokens=False)
    # warm-up of the full-context path (prefill, then a multi-token continuation on a cache with
    # GDN initial state -> different fla kernel variant, otherwise ~15 s of Triton autotune lands in TTFT)
    use_impl(model, 10**9)
    with torch.no_grad():
        w = model(input_ids=torch.tensor([hist_ids[:512]], device=dev0), use_cache=True, logits_to_keep=1)
        greedy(model, query_ids(tok, "Say hello."), 4, cache=w.past_key_values)
        del w
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_alloc = {d: torch.cuda.memory_allocated(d) for d in range(torch.cuda.device_count())}
    inp = torch.tensor([hist_ids], device=dev0)
    use_impl(model, len(hist_ids))
    sync()
    t = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=inp, use_cache=True, logits_to_keep=1)  # no memory_state -> plain backbone
    sync()
    prefill_s = time.perf_counter() - t
    cache = out.past_key_values
    del out
    flat = cache_tensors(cache)
    kv = {k: v for k, v in flat.items() if k.endswith((".keys", ".values"))}
    lin = {k: v for k, v in flat.items() if not k.endswith((".keys", ".values"))}
    seq = cache.get_seq_length()
    emit("fullctx_prefill", tokens=len(hist_ids), seconds=round(prefill_s, 3),
         tok_per_s=round(len(hist_ids) / prefill_s, 1), cache_class=type(cache).__name__,
         layer_classes=sorted({type(l).__name__ for l in cache.layers}), seq_len=seq,
         kv_bytes=tensor_bytes(kv), kv_bytes_per_token=tensor_bytes(kv) / seq,
         linear_state_bytes=tensor_bytes(lin), linear_state_MiB=tensor_bytes(lin) / 2**20,
         total_bytes=tensor_bytes(flat),
         shapes={k: [list(v.shape), str(v.dtype)] for k, v in list(flat.items())[:8]},
         gpu_delta_GiB={d: round((torch.cuda.memory_allocated(d) - base_alloc[d]) / 2**30, 3) for d in base_alloc},
         gpu=gpu_mem())

    # save cache to FlashBlade
    cpath = os.path.join(args.state_dir, "smoke_fullctx_cache.safetensors")
    t = time.perf_counter()
    save_file({k: v.detach().cpu().contiguous() for k, v in flat.items()}, cpath,
              metadata={"format": "hf-dynamiccache-hybrid-v0", "seq_len": str(seq)})
    with open(cpath, "rb") as fh:
        os.fsync(fh.fileno())
    emit("fullctx_save", path=cpath, file_bytes=os.path.getsize(cpath), seconds=round(time.perf_counter() - t, 3))

    # reference answer from a deep copy of the live cache
    import copy
    live = copy.deepcopy(cache)
    q, gold = questions[0]
    toks_ref, ttft_ref, tps_ref, _ = greedy(model, query_ids(tok, q), args.gen_tokens, cache=live)
    ans_ref = answer_text(tok, toks_ref, eos)
    emit("query", mode="fullctx_live", question=q, gold=gold, answer=ans_ref[:200], correct=gold.lower() in ans_ref.lower(),
         ttft_s=round(ttft_ref, 4), decode_tok_per_s=round(tps_ref, 2), context_tokens=seq)
    del live, cache, flat, kv, lin
    torch.cuda.empty_cache()

    # restore from disk into a fresh cache and re-ask all questions
    for qi, (q, gold) in enumerate(questions):
        sync()
        t = time.perf_counter()
        flat2 = load_file(cpath, device="cpu")
        cache2 = cache_from_tensors(model.config, flat2, layer_dev)
        sync()
        rs = time.perf_counter() - t
        toks, ttft, tps, _ = greedy(model, query_ids(tok, q), args.gen_tokens, cache=cache2)
        ans = answer_text(tok, toks, eos)
        extra = {"matches_live": toks == toks_ref} if qi == 0 else {}
        emit("query", mode="fullctx_restored", question=q, gold=gold, answer=ans[:200],
             correct=gold.lower() in ans.lower(), restore_s=round(rs, 4), ttft_s=round(ttft, 4),
             ttft_plus_restore_s=round(ttft + rs, 4), decode_tok_per_s=round(tps, 2), context_tokens=seq, **extra)
        del cache2, flat2
        torch.cuda.empty_cache()
    emit("done", gpu=gpu_mem(), nvidia_smi=nvsmi())


if __name__ == "__main__":
    main()
