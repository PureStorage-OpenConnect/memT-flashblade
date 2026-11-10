#!/usr/bin/env python3
"""TTFT probe for vLLM (+pKVA) full-context baseline.

Builds a deterministic synthetic "history" of exactly N tokens (model tokenizer), appends a fixed
~512-token query, sends it as a token-id prompt to /v1/completions with streaming, max_tokens=128,
ignore_eos, temperature 0. Records TTFT (first streamed token), total time, and deltas of vLLM's
prefix-cache counters (local GPU hits vs external/connector hits) from /metrics.

Usage: bench_ttft.py --n 8192 --label cold [--port 8101] [--out results.jsonl] [--seed-tag hist]
"""
import argparse, json, random, re, time, sys, os
import requests

MODEL_DIR_HINT = "Qwen/Qwen3.6-35B-A3B"
WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa "
         "quebec romeo sierra tango uniform victor whiskey xray yankee zulu memory session token cache store "
         "river mountain forest ocean desert valley city harbor bridge tower garden market library museum "
         "engine signal vector matrix kernel buffer packet stream thread socket server client request reply "
         "red orange yellow green blue indigo violet silver golden copper iron stone glass paper wooden").split()
NAMES = "Ava Ben Chloe Dan Eve Finn Gia Hugo Iris Jack Kai Lena Max Nora Owen Pia Quinn Ray Sara Tom".split()


def synth_text(seed: str, approx_words: int) -> str:
    rng = random.Random(seed)
    out = []
    turn = 0
    while len(out) < approx_words:
        turn += 1
        who = rng.choice(NAMES)
        n = rng.randint(12, 40)
        sent = " ".join(rng.choice(WORDS) for _ in range(n))
        num = rng.randint(0, 99999)
        out.extend(f"[turn {turn}] {who}: {sent}, ref #{num}.".split())
    return " ".join(out)


def build_prompt(tok, n_hist: int, seed_tag: str):
    hist_ids = []
    words = n_hist  # ~1.3 tok/word; grow until long enough
    while len(hist_ids) < n_hist:
        hist_ids = tok.encode(synth_text(f"{seed_tag}-{n_hist}", words), add_special_tokens=False)
        words = int(words * 1.5) + 100
    hist_ids = hist_ids[:n_hist]
    q_text = synth_text("fixed-query", 400)
    q_ids = tok.encode("\n\nQuestion: summarize the conversation above, listing who mentioned which "
                       "reference numbers. Context notes: " + q_text, add_special_tokens=False)[:512]
    return hist_ids + q_ids, len(hist_ids), len(q_ids)


def scrape(port):
    try:
        txt = requests.get(f"http://localhost:{port}/metrics", timeout=10).text
    except Exception:
        return {}
    keys = ["vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total",
            "vllm:external_prefix_cache_queries_total", "vllm:external_prefix_cache_hits_total",
            "vllm:prompt_tokens_total"]
    out = {}
    for line in txt.splitlines():
        if line.startswith("#"):
            continue
        for k in keys:
            if line.startswith(k + "{") or line.startswith(k + " "):
                out[k] = out.get(k, 0.0) + float(line.rsplit(" ", 1)[1])
    # any connector-specific metrics (names vary): capture raw
    for line in txt.splitlines():
        if not line.startswith("#") and re.search(r"kv_connector|purekv|kv_transfer", line, re.I):
            name = line.rsplit(" ", 1)[0]
            try:
                out[name] = float(line.rsplit(" ", 1)[1])
            except ValueError:
                pass
    return out


def run(port, model, prompt_ids, max_tokens=128):
    body = {"model": model, "prompt": prompt_ids, "max_tokens": max_tokens, "temperature": 0,
            "stream": True, "ignore_eos": True,
            "stream_options": {"include_usage": True}}
    t0 = time.perf_counter()
    ttft = None
    ntok_chunks = 0
    usage = None
    with requests.post(f"http://localhost:{port}/v1/completions", json=body, stream=True, timeout=3600) as r:
        r.raise_for_status()
        for raw in r.iter_lines():
            if not raw or not raw.startswith(b"data: "):
                continue
            data = raw[6:]
            if data == b"[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            ch = ev.get("choices") or []
            if ch and ch[0].get("text"):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                ntok_chunks += 1
    total = time.perf_counter() - t0
    return ttft, total, ntok_chunks, usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--port", type=int, default=8101)
    ap.add_argument("--model", default="qwen3.6-35b-a3b")
    ap.add_argument("--seed-tag", default="hist")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR_HINT)
    ids, nh, nq = build_prompt(tok, a.n, a.seed_tag)
    m0 = scrape(a.port)
    ttft, total, nchunks, usage = run(a.port, a.model, ids)
    time.sleep(1.0)  # let stats flush
    m1 = scrape(a.port)
    delta = {k: m1.get(k, 0) - m0.get(k, 0) for k in set(m0) | set(m1) if m1.get(k, 0) != m0.get(k, 0)}
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "label": a.label, "n_hist": nh, "n_query": nq,
           "prompt_tokens": len(ids), "ttft_s": ttft, "total_s": total, "stream_chunks": nchunks,
           "usage": usage, "metric_delta": delta}
    print(json.dumps(rec))
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
