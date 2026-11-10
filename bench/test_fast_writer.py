#!/usr/bin/env python
"""Verify and benchmark memt_fast_writer against the shipped MemTMemory write loop.

Every result is a JSON line on stdout. Modes (run with CUDA_VISIBLE_DEVICES=2,3):
  random    : random hidden states + real layer weights, L in 1..64 and 425/2048, vs FP64/FP32 loop
  capture   : load the full model, commit ~4K tokens of text, save real MemTMemory inputs
  real      : offline compare on captured inputs (all 10 layers) + 3-slice chaining
  micro     : single-layer write-step timing, L=2048, reference vs fast
  e2e       : full model, ~6K-token history committed with reference and fast writer;
              greedy 32 tokens for 3 questions; commit_memory timing for a 2048 slice
"""
import argparse
import importlib
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memt_fast_writer as fw  # noqa: E402

SNAP = ("/mnt/data/robert-scripts/hf-cache/hub/models--MemTensor--MemT-1.0-Nano/snapshots/"
        "30fe0b22422c4cc4c87475f98db5c8baa5c1a902")
MODULES = "/mnt/data/robert-scripts/hf-cache/modules"
DYN = "transformers_modules.MemTensor.MemT_hyphen_1_dot_0_hyphen_Nano.30fe0b22422c4cc4c87475f98db5c8baa5c1a902"
SCRATCH = ("/tmp/claude-1000/-mnt-data-robert-scripts-m1t-test/d52341ba-cd8c-4b9e-83de-e2ab4ba1debb/"
           "scratchpad/fastwriter")
MEM_LAYERS = list(range(3, 40, 4))


def emit(evt, **kw):
    print(json.dumps({"evt": evt, **kw}, default=str), flush=True)


def sync():
    for d in range(torch.cuda.device_count()):
        torch.cuda.synchronize(d)


def memt_class():
    if MODULES not in sys.path:
        sys.path.insert(0, MODULES)
    return importlib.import_module(DYN + ".modeling_memt").MemTMemory


def load_memory_module(layer, device):
    from safetensors import safe_open
    cfg = json.load(open(os.path.join(SNAP, "config.json")))
    text = cfg["backbone_configs"]["text_config"]
    cls = memt_class()
    m = cls(cfg["memt"], text["hidden_size"], text["num_attention_heads"], text["head_dim"],
            text["rms_norm_eps"])
    prefix = f"model.language_model.layers.{layer}.self_attn.memt."
    with safe_open(os.path.join(SNAP, "memt_block.safetensors"), "pt") as f:
        sd = {k[len(prefix):]: f.get_tensor(k) for k in f.keys() if k.startswith(prefix)}
    m.load_state_dict(sd)
    return m.to(device=device, dtype=torch.bfloat16).eval()


def errs(out, ref, q=None, scale=None):
    """max-abs / max-rel(of max) state error, plus error of the next read q.M."""
    out, ref = out.double(), ref.double()
    d = (out - ref).abs()
    r = {"state_max_abs": d.max().item(), "state_rel": d.max().item() / ref.abs().max().item(),
         "state_rel_fro": ((out - ref).norm() / ref.norm()).item()}
    if q is not None:
        qd = q.double()
        ro = scale * torch.einsum("blhk,bhkv->blhv", qd, out)
        rr = scale * torch.einsum("blhk,bhkv->blhv", qd, ref)
        rd = (ro - rr).abs()
        r.update(read_max_abs=rd.max().item(), read_rel=rd.max().item() / rr.abs().max().item())
    return r


def query_of(m, hidden):
    w = m.value_dim
    shape = (*hidden.shape[:2], m.num_heads, m.head_dim)
    p = torch.nn.functional.linear(hidden, m.in_proj.weight[:w]).reshape(shape)
    return m._normalize(p, True)


def timeit(fn, n=3):
    sync()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t)
    return min(ts), ts


# ----------------------------------------------------------------------------- random
@torch.no_grad()
def run_random(args):
    dev = "cuda:0"
    torch.manual_seed(0)
    m = load_memory_module(args.layer, dev)
    # hidden-state scale: real pre-attention hidden states (post input_layernorm) are O(1) per channel
    lengths = list(range(1, 65)) if args.all_small else [1, 2, 3, 7, 16, 31, 63, 64, 65, 100, 425, 2048]
    worst = {}
    for L in lengths:
        h = torch.randn(1, L, 2048, device=dev).bfloat16()
        for init in ("zero", "rand"):
            s0 = torch.zeros(1, 16, 256, 256, device=dev) if init == "zero" else \
                0.05 * torch.randn(1, 16, 256, 256, device=dev)
            ref64 = fw.reference_commit_state(m, h, s0, torch.float64)
            ref32 = fw.reference_commit_state(m, h, s0, torch.float32)
            q = query_of(m, torch.randn(1, 8, 2048, device=dev).bfloat16())
            out = fw.fast_commit_state(m, h, s0)
            # also check the actual module (unpatched) output equals our FP32 replica
            _, mod = m(h, s0, True)
            e = errs(out, ref64, q, m.read_scale)
            e32 = errs(ref32, ref64, q, m.read_scale)
            emit("random", layer=args.layer, L=L, init=init, fast_vs_fp64=e, ref32_vs_fp64=e32,
                 fast_vs_ref32=errs(out, ref32), module_vs_replica=(mod - ref32).abs().max().item())
            for k in ("state_rel", "read_rel"):
                worst[k] = max(worst.get(k, 0), e[k])
    emit("random_worst", **worst)


# ----------------------------------------------------------------------------- capture
def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("MemTensor/MemT-1.0-Nano", trust_remote_code=True)
    t = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        "MemTensor/MemT-1.0-Nano", trust_remote_code=True, dtype=torch.bfloat16, device_map="auto",
        max_memory={i: "75GiB" for i in range(torch.cuda.device_count())},
        attn_implementation="sdpa", experts_implementation="grouped_mm")
    model.eval()
    emit("load", seconds=round(time.perf_counter() - t, 1))
    return tok, model


def history_docs(tok, n_facts):
    import smoke_memt_hf as smoke
    facts = smoke.build_facts(n_facts)
    return facts, smoke.pack_documents(tok, facts)


@torch.no_grad()
def run_capture(args):
    tok, model = load_model()
    facts, docs = history_docs(tok, 160)
    ids = [t for d in docs for t in d][:args.capture_tokens]
    dev0 = model.get_input_embeddings().weight.device
    saved = {}
    hooks = []
    for i, layer in enumerate(model.model.language_model.layers):
        if hasattr(layer, "self_attn"):
            def hook(mod, a, kw, i=i):
                hidden, state, commit = a[0], a[1], a[2]
                ws = kw.get("write_state")
                n = sum(1 for k in saved if k.startswith(f"{i}."))
                saved[f"{i}.{n}.hidden"] = hidden.detach().cpu()
                if ws is not None:
                    saved[f"{i}.{n}.write_state"] = ws.detach().cpu()
            hooks.append(layer.self_attn.memt.register_forward_pre_hook(hook, with_kwargs=True))
    ids_t = torch.tensor([ids], device=dev0)
    model.commit_memory(ids_t, None, document=True)
    for h in hooks:
        h.remove()
    path = os.path.join(SCRATCH, "captured_inputs.pt")
    torch.save(saved, path)
    emit("capture", path=path, tokens=len(ids), n=len(saved),
         shapes={k: list(v.shape) for k, v in list(saved.items())[:4]})


@torch.no_grad()
def run_real(args):
    dev = "cuda:0"
    saved = torch.load(os.path.join(SCRATCH, "captured_inputs.pt"))
    worst = {}
    for layer in MEM_LAYERS:
        m = load_memory_module(layer, dev)
        slices = sorted({int(k.split(".")[1]) for k in saved if k.startswith(f"{layer}.")})
        hs = [saved[f"{layer}.{s}.hidden"].to(dev) for s in slices]
        q = query_of(m, hs[-1][:, -16:])
        g = fw.write_inputs(m, hs[0])[3]
        # per-slice from zero state (slice 0 = real first event)
        s0 = torch.zeros(1, 16, 256, 256, device=dev)
        ref64 = fw.reference_commit_state(m, hs[0], s0, torch.float64)
        ref32 = fw.reference_commit_state(m, hs[0], s0, torch.float32)
        out = fw.fast_commit_state(m, hs[0], s0)
        e = errs(out, ref64, q, m.read_scale)
        emit("real_single", layer=layer, L=hs[0].shape[1], fast_vs_fp64=e,
             ref32_vs_fp64=errs(ref32, ref64, q, m.read_scale),
             log_decay_min=g.min().item(), log_decay_max=g.max().item(),
             state_absmax=ref64.abs().max().item())
        for k in ("state_rel", "read_rel"):
            worst[k] = max(worst.get(k, 0), e[k])
        # chaining: 3 slices (captured slices, padded out by re-using real hidden states)
        chain = (hs * 3)[:3]
        r64, r32, f = s0.double(), s0, s0
        for h in chain:
            r64 = fw.reference_commit_state(m, h, r64, torch.float64)
            r32 = fw.reference_commit_state(m, h, r32, torch.float32)
            f = fw.fast_commit_state(m, h, f)
        e = errs(f, r64, q, m.read_scale)
        emit("real_chain3", layer=layer, lengths=[h.shape[1] for h in chain], fast_vs_fp64=e,
             ref32_vs_fp64=errs(r32, r64, q, m.read_scale), fast_vs_ref32=errs(f, r32, q, m.read_scale))
        for k in ("state_rel", "read_rel"):
            worst["chain_" + k] = max(worst.get("chain_" + k, 0), e[k])
        del m
        torch.cuda.empty_cache()
    emit("real_worst", **worst)


@torch.no_grad()
def run_micro(args):
    dev = "cuda:0"
    m = load_memory_module(args.layer, dev)
    h = torch.randn(1, 2048, 2048, device=dev).bfloat16()
    s0 = torch.zeros(1, 16, 256, 256, device=dev)
    sync()
    t = time.perf_counter()
    fw.fast_commit_state(m, h, s0)
    sync()
    first = time.perf_counter() - t
    u, v, wg, g = fw.write_inputs(m, h)
    ref_t, _ = timeit(lambda: m(h, s0, True), 2)
    fast_t, _ = timeit(lambda: fw.fast_commit_state(m, h, s0), 10)
    kern_t, _ = timeit(lambda: fw.gdn2_state_update(u, v, wg, g, s0), 10)
    inp_t, _ = timeit(lambda: fw.write_inputs(m, h), 10)
    fw.patch_module(m)
    patched_t, _ = timeit(lambda: m(h, s0, True), 10)
    fw.unpatch_module(m)
    emit("micro", L=2048, layer=args.layer, first_call_s=round(first, 2),
         ref_module_forward_s=round(ref_t, 4), fast_module_forward_s=round(patched_t, 5),
         fast_commit_state_s=round(fast_t, 5), kernel_only_s=round(kern_t, 5),
         write_inputs_only_s=round(inp_t, 5), speedup_forward=round(ref_t / patched_t, 1))


# ----------------------------------------------------------------------------- e2e
@torch.no_grad()
def run_e2e(args):
    import smoke_memt_hf as smoke
    tok, model = load_model()
    smoke._cur_impl[0] = model.config._experts_implementation
    facts, docs = history_docs(tok, args.n_facts)
    dev0 = model.get_input_embeddings().weight.device
    ids = torch.tensor([[t for d in docs for t in d]], device=dev0)
    emit("history", tokens=ids.shape[1], n_docs=len(docs))
    eos = set(model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, list)
              else [model.generation_config.eos_token_id])

    def commit(label):
        smoke.use_impl(model, 10**9)
        sync()
        t = time.perf_counter()
        st = model.commit_memory(ids, None, document=True)
        sync()
        emit("commit_full", mode=label, tokens=ids.shape[1], seconds=round(time.perf_counter() - t, 3))
        return st

    # warm backbone (fla GDN autotune) with a reference commit of one slice
    sl = torch.tensor([docs[1][:2048]], device=dev0)
    smoke.use_impl(model, 10**9)
    model.commit_memory(sl[:, :256], None, document=True)

    state_ref = commit("reference")
    variants = {}
    n = fw.patch_model(model)
    emit("patched", modules=n)
    sync()
    t = time.perf_counter()
    model.commit_memory(sl[:, :128], None, document=True)  # kernel first call (autotune)
    sync()
    emit("fast_first_call", seconds=round(time.perf_counter() - t, 2),
         triton_f32_default=os.environ.get("TRITON_F32_DEFAULT"))
    variants["fast"] = commit("fast")
    fw.unpatch_model(model)
    if args.noise_control:
        # Control: reference loop, then Gaussian noise of relative Frobenius size eps on every
        # committed state (eps ~ the kernel's per-slice error vs FP64). Shows how much a
        # perturbation of that size alone moves the deep-layer states and the generated text.
        gen = torch.Generator(device="cpu").manual_seed(0)
        for mod in fw._memory_modules(model):
            ref_fwd = mod.forward

            def noisy(hidden, state=None, commit=False, *, write_state=None, _f=ref_fwd):
                read, upd = _f(hidden, state, commit, write_state=write_state)
                if commit:
                    z = torch.randn(upd.shape, generator=gen).to(upd.device)
                    upd = upd + args.noise_control * upd.std() * z
                return read, upd
            mod.forward = noisy
        variants[f"ref_noise{args.noise_control:g}"] = commit("ref_noise")
        for mod in fw._memory_modules(model):
            del mod.__dict__["forward"]
    for name, st in variants.items():
        for k in sorted(state_ref):
            a, b = st[k].double(), state_ref[k].double().to(st[k].device)
            emit("e2e_state", variant=name, layer=k, max_abs=(a - b).abs().max().item(),
                 rel=(a - b).abs().max().item() / b.abs().max().item(),
                 rel_fro=((a - b).norm() / b.norm()).item())

    picks = [facts[5], facts[len(facts) // 2], facts[-7]]
    questions = [(f"How many {p['item']} does {p['name']} own? Answer with the number only.", str(p["num"]))
                 for p in picks[:2]]
    questions += [(f"In which city does {picks[2]['name']} live? Answer with the city name only.",
                   picks[2]["city"])]
    smoke.greedy(model, smoke.query_ids(tok, "Say hello."), 2, memory_state=state_ref)
    for q, gold in questions:
        qi = smoke.query_ids(tok, q)
        inp = torch.tensor([qi], device=dev0)
        smoke.use_impl(model, len(qi))
        lo_r = model(input_ids=inp, use_cache=False, memory_state=state_ref, logits_to_keep=1).logits[0, -1].float()
        toks_r, *_ = smoke.greedy(model, qi, args.gen_tokens, memory_state=state_ref)
        ans_r = smoke.answer_text(tok, toks_r, eos)
        top2 = lo_r.topk(2).values
        for name, st in variants.items():
            smoke.use_impl(model, len(qi))
            lo = model(input_ids=inp, use_cache=False, memory_state=st, logits_to_keep=1).logits[0, -1].float()
            lo = lo.to(lo_r.device)
            toks_f, *_ = smoke.greedy(model, qi, args.gen_tokens, memory_state=st)
            ans_f = smoke.answer_text(tok, toks_f, eos)
            div = next((i for i, (a, b) in enumerate(zip(toks_r, toks_f)) if a != b), None)
            emit("e2e_query", variant=name, question=q, gold=gold, ref=ans_r, other=ans_f,
                 tokens_match=toks_r == toks_f, answer_match=ans_r == ans_f, first_divergence=div,
                 ref_correct=gold.lower() in ans_r.lower(), other_correct=gold.lower() in ans_f.lower(),
                 first_step_max_logit_diff=(lo - lo_r).abs().max().item(),
                 first_step_logit_absmax=lo_r.abs().max().item(),
                 ref_top1_top2_margin=(top2[0] - top2[1]).item(),
                 argmax_same=bool(lo.argmax() == lo_r.argmax()))
    if args.skip_timing:
        return

    # timing: commit_memory of one 2048-token slice, plus plain prefill of the same slice
    smoke.use_impl(model, 10**9)
    one = torch.tensor([docs[1]], device=dev0)
    for label in ("reference", "fast"):
        if label == "fast":
            fw.patch_model(model)
        best, ts = timeit(lambda: model.commit_memory(one, state_ref, document=True), 3)
        emit("commit_slice_timing", mode=label, tokens=one.shape[1], best_s=round(best, 3),
             all_s=[round(x, 3) for x in ts])
        fw.unpatch_model(model)
    best, _ = timeit(lambda: model(input_ids=one, use_cache=True, logits_to_keep=1), 3)
    emit("plain_prefill_timing", tokens=one.shape[1], best_s=round(best, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["random", "capture", "real", "micro", "e2e"])
    ap.add_argument("--layer", type=int, default=7)
    ap.add_argument("--all-small", action="store_true")
    ap.add_argument("--capture-tokens", type=int, default=4096)
    ap.add_argument("--n-facts", type=int, default=240)
    ap.add_argument("--gen-tokens", type=int, default=32)
    ap.add_argument("--noise-control", type=float, default=0.0)
    ap.add_argument("--skip-timing", action="store_true")
    args = ap.parse_args()
    os.makedirs(SCRATCH, exist_ok=True)
    emit("env", torch=torch.__version__, visible=os.environ.get("CUDA_VISIBLE_DEVICES"))
    {"random": run_random, "capture": run_capture, "real": run_real, "micro": run_micro,
     "e2e": run_e2e}[args.mode](args)


if __name__ == "__main__":
    main()
