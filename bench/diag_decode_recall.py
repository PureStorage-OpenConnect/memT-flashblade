#!/usr/bin/env python
"""Diagnostics: (1) HF decode speed per experts implementation + profile, (2) easy memory recall.

CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python bench/diag_decode_recall.py
"""
import json
import time

import torch

MODEL_ID = "MemTensor/MemT-1.0-Nano"


def emit(evt, **kw):
    print(json.dumps({"evt": evt, **kw}, default=str), flush=True)


def sync():
    for d in range(torch.cuda.device_count()):
        torch.cuda.synchronize(d)


@torch.no_grad()
def decode_bench(model, ids, n, memory_state=None):
    dev = model.get_input_embeddings().weight.device
    out = model(input_ids=torch.tensor([ids], device=dev), use_cache=True, logits_to_keep=1,
                memory_state=memory_state)
    cache, nxt = out.past_key_values, out.logits[:, -1].argmax(-1)
    sync()
    t = time.perf_counter()
    for _ in range(n):
        out = model(input_ids=nxt[:, None], past_key_values=cache, use_cache=True, logits_to_keep=1,
                    memory_state=memory_state)
        nxt = out.logits[:, -1].argmax(-1)
    sync()
    return n / (time.perf_counter() - t), cache, nxt


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, dtype=torch.bfloat16,
                                                 device_map="auto", attn_implementation="sdpa").eval()
    q = tok.apply_chat_template([{"role": "user", "content": "Write a short poem about the sea."}],
                                tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok.encode(q, add_special_tokens=False)
    for impl in ["grouped_mm", "batched_mm", "eager"]:
        try:
            model.set_experts_implementation(impl)
            decode_bench(model, ids, 8)
            tps, _, _ = decode_bench(model, ids, 64)
            emit("decode", experts=impl, tok_per_s=round(tps, 2))
        except Exception as e:  # noqa: BLE001
            emit("decode", experts=impl, error=repr(e)[:500])
    model.set_experts_implementation("grouped_mm")

    # profile a handful of decode steps
    from torch.profiler import ProfilerActivity, profile
    _, cache, nxt = decode_bench(model, ids, 4)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with torch.no_grad():
            for _ in range(5):
                out = model(input_ids=nxt[:, None], past_key_values=cache, use_cache=True, logits_to_keep=1)
                nxt = out.logits[:, -1].argmax(-1)
            sync()
    ka = prof.key_averages()
    emit("profile_cpu_top", rows=[(e.key, round(e.self_cpu_time_total / 5e3, 2), e.count // 5)
                                  for e in sorted(ka, key=lambda e: -e.self_cpu_time_total)[:20]])
    emit("profile_cuda_top", rows=[(e.key, round(e.self_device_time_total / 5e3, 2), e.count // 5)
                                   for e in sorted(ka, key=lambda e: -e.self_device_time_total)[:15]])
    emit("profile_totals_ms_per_step", cpu=round(sum(e.self_cpu_time_total for e in ka) / 5e3, 1),
         cuda=round(sum(e.self_device_time_total for e in ka) / 5e3, 1))

    # (2) easy recall: README example through the official session API
    s = model.create_session(tok)
    s.chat("I have a three-month-old kitten named memmy who loves eating small shrimp and is excellent "
           "at remembering things.", enable_memory=True, enable_thinking=False, do_sample=False, max_new_tokens=128)
    a = s.chat("What are my kitten's name, age, favorite food, and special trait?", enable_thinking=False,
               do_sample=False, max_new_tokens=128)
    emit("recall_readme", answer=a[:400], messages_after=len(s.messages))
    s.close()

    # (3) small document (12 distinctive facts) ingested via session documents
    facts = [
        ("Marguerite Okonkwo", "violinist", "Lisbon", "a three-legged greyhound named Biscuit"),
        ("Tobias Lindqvist", "glassblower", "Reykjavik", "a parrot named Admiral"),
        ("Priya Raghunathan", "volcanologist", "Quito", "a tortoise named Sir Plod"),
        ("Henrik Vasquez", "lighthouse keeper", "Galway", "a ferret named Pickle"),
        ("Aiko Brennan", "cartographer", "Hobart", "a goat named Juniper"),
        ("Dmitri Achebe", "pastry chef", "Montreal", "an iguana named Lord Byron"),
    ]
    doc = " ".join(f"{n} works as a {j} in {c} and keeps {p}." for n, j, c, p in facts)
    s = model.create_session(tok)
    res = []
    for n, j, c, p in facts:
        s.reset()
        a = s.chat(f"What is {n}'s job, which city does {n.split()[0]} work in, and what pet does {n.split()[0]} keep?",
                   memory=doc, enable_thinking=False, do_sample=False, max_new_tokens=64)
        hit = [j.lower() in a.lower(), c.lower() in a.lower(), p.split()[-1].lower() in a.lower()]
        res.append(sum(hit))
        emit("recall_small", person=n, answer=a[:300], hits=hit)
    emit("recall_small_summary", score=sum(res), max=3 * len(facts))
    s.close()


if __name__ == "__main__":
    main()
