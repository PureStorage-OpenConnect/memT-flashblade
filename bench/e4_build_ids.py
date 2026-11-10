#!/usr/bin/env python
"""E4: build the exact token ids used by both E4 parts (vLLM+pKVA and HF MemT-Nano) once.

Same BEAM histories / 512-token query / chat frame as E1 (bench/beam_histories.py, tol 0.01).
The 256K point is capped at 261120 history tokens so that the vLLM prompt
(prefix + history + "\n\n" + query + gen suffix) + 128 generated tokens fits max-model-len 262144.
MemT-1.0-Nano and Qwen/Qwen3.6-35B-A3B tokenizers are identical (vocab + chat template checked).

Output JSON: {meta, frame, sep, q_content, histories: {"<h>_<N>": {...ids...}}, fillers: {...}}
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
M1T_ROOT = os.environ.get("M1T_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import beam_histories as bh

CAP_256K = 261120

ap = argparse.ArgumentParser()
ap.add_argument("--histories", default="0,1,2",
                help="per-N history indices: plain list = same for every N; 'distinct' = N_i uses 3i,3i+1,3i+2")
ap.add_argument("--lengths", default="2048,8192,32768,65536,131072,262144")
ap.add_argument("--fillers", default="20:131072,21:131072",
                help="history:N pairs used only to evict pKVA's pinned host pool")
ap.add_argument("--out", default=os.path.join(M1T_ROOT, "scratch_state", "e4", "e4_ids.json"))
a = ap.parse_args()

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("MemTensor/MemT-1.0-Nano", trust_remote_code=True)
convs = bh.load_beam()
frame = bh.chat_frame(tok)
sep = tok.encode(bh.SEP, add_special_tokens=False)
q = bh.query_content_ids(tok, 512)
Ns = [int(x) for x in a.lengths.split(",")]
if a.histories == "distinct":
    # pKVA keys chunks by token content only (cache_salt does not reach it), so the same conversation at
    # two lengths shares a stored prefix and would make "cold" a partial external hit. One set per N.
    plan = {n: [3 * i, 3 * i + 1, 3 * i + 2] for i, n in enumerate(Ns)}
else:
    plan = {n: [int(x) for x in a.histories.split(",")] for n in Ns}
hs = sorted({h for v in plan.values() for h in v})
fill = [tuple(map(int, x.split(":"))) for x in a.fillers.split(",") if x]
need = {}
out = {"meta": {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "tokenizer": "MemTensor/MemT-1.0-Nano",
                "tol": 0.01, "cap_256k": CAP_256K, "plan": {str(k): v for k, v in plan.items()}, "beam": bh.BEAM_PARQUET},
       "frame": {k: frame[k] for k in ("prefix", "end_user", "gen_suffix")}, "sep": sep, "q_content": q,
       "histories": {}, "fillers": {}}


def one(h, n):
    tc = bh.tokenize_conversation(tok, convs[h][0], convs[h][1], int(n * 1.01) + 8)
    hist = bh.build_history(tc, n, 0.01)
    ids = hist.ids[:CAP_256K] if n == 262144 else hist.ids
    prompt = frame["prefix"] + ids + sep + q + frame["gen_suffix"]
    return {"history_idx": h, "N": n, "conv_id": hist.conv_id, "history_tokens": len(ids),
            "hist_ids": ids, "prompt_ids": prompt, "prompt_tokens": len(prompt)}


for n in Ns:
    for h in plan[n]:
        r = one(h, n)
        out["histories"][f"{h}_{n}"] = r
        print(h, n, r["conv_id"], r["history_tokens"], r["prompt_tokens"], file=sys.stderr)
for h, n in fill:
    r = one(h, n)
    out["fillers"][f"{h}_{n}"] = r
    print("filler", h, n, r["history_tokens"], file=sys.stderr)
with open(a.out, "w") as f:
    json.dump(out, f)
print(a.out)
