#!/usr/bin/env python3
"""Markdown tables for E4 results. Usage: e4_summarize.py [--run-pkva ID] [--run-memt ID]"""
import argparse, json, statistics as st
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--pkva", default="/mnt/data/robert-scripts/m1t-test/results/e4_pkva_resume.jsonl")
ap.add_argument("--memt", default="/mnt/data/robert-scripts/m1t-test/results/e4_memt_resume.jsonl")
ap.add_argument("--run-pkva"); ap.add_argument("--run-memt")
a = ap.parse_args()


def load(p, run):
    try:
        rs = [json.loads(l) for l in open(p)]
    except FileNotFoundError:
        return []
    if run:
        rs = [r for r in rs if r.get("run_id") == run]
    return rs


def fmt(xs, scale=1.0, nd=3):
    if not xs:
        return "-"
    xs = [x * scale for x in xs]
    return f"{st.median(xs):.{nd}f} [{min(xs):.{nd}f}-{max(xs):.{nd}f}] n={len(xs)}"


P = load(a.pkva, a.run_pkva)
if P:
    print("### vLLM + pKVA (TTFT s, median [min-max] n=samples over 3 histories)\n")
    print("| N | cold | hot | restored (FlashBlade, verified) | invalid restores | ext hits / expected | bytes on FB per history (median) | files | B/token |")
    print("|---|---|---|---|---|---|---|---|---|")
    g = defaultdict(lambda: defaultdict(list))
    for r in P:
        if "N" not in r or r.get("label", "").startswith(("filler", "first")):
            continue
        g[r["N"]][r["label"] + ("" if r.get("valid", True) else "_invalid")].append(r)
    for n in sorted(g):
        d = g[n]
        ttft = lambda k: [r["ttft_s"] for r in d.get(k, [])]
        ext = [r["metric_delta"].get("vllm:external_prefix_cache_hits_total", 0) for r in d.get("restored", [])]
        exp = [r["expected_hit_tokens"] for r in d.get("restored", [])]
        bw = [r["bytes_written"] for r in d.get("cold", [])]
        fw = [r["files_written"] for r in d.get("cold", [])]
        bpt = [r["bytes_written"] / r["history_tokens"] for r in d.get("cold", [])]
        print(f"| {n} | {fmt(ttft('cold'))} | {fmt(ttft('hot'))} | {fmt(ttft('restored'))} | {len(d.get('restored_invalid', []))} | "
              f"{int(min(ext)) if ext else '-'}-{int(max(ext)) if ext else '-'} / {min(exp) if exp else '-'}-{max(exp) if exp else '-'} | "
              f"{st.median(bw)/1e9 if bw else 0:.3f} GB | {st.median(fw) if fw else '-'} | {st.median(bpt)/1024 if bpt else 0:.1f} KiB |")
    print("\nrestored LoadKV: chunks (sum both TP ranks), resident, cpu, load span s (median)\n")
    for n in sorted(g):
        rr = g[n].get("restored", [])
        if rr:
            print(f"- N={n}: chunks={st.median([r['kva']['load_chunks_sum'] for r in rr])} resident_max={max(r['kva']['load_resident_sum'] for r in rr)} "
                  f"cpu={st.median([r['kva']['load_cpu_sum'] for r in rr])} span={fmt([r['kva']['load_span_s'] or 0 for r in rr])} "
                  f"total_s={fmt([r['total_s'] for r in rr])}")
    for r in P:
        if r.get("label", "").startswith(("first", "filler_restore")):
            print(f"- {r['label']} seq={r.get('seq')} filler {r['history_idx']}: ttft={r['ttft_s']:.3f}s chunks={r['kva']['load_chunks_sum']} resident={r['kva']['load_resident_sum']} cpu={r['kva']['load_cpu_sum']} span={r['kva']['load_span_s']}")

M = load(a.memt, a.run_memt)
if M:
    print("\n### HF MemT-Nano / full-context restore (ms, median [min-max] n)\n")
    print("| N | system | variant | file bytes | read_s | h2d_s | prefill_s | ttft_s | total_s | tokens match ref |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    g = defaultdict(list); sv = defaultdict(list)
    for r in M:
        if r.get("label") == "save":
            sv[(r["N"], r["system"])].append(r["file_bytes"])
        elif r.get("label") in ("odirect", "pagecache_warm"):
            g[(r["N"], r["system"], r["label"])].append(r)
    for k in sorted(g):
        rr = g[k]
        f = lambda x: fmt([r[x] for r in rr], 1e3, 1)
        print(f"| {k[0]} | {k[1]} | {k[2]} | {st.median(sv[k[:2]]) / 2**20:.1f} MiB | {f('read_s')} | {f('h2d_s')} | {f('prefill_s')} | "
              f"{f('ttft_s')} | {fmt([r['total_s'] for r in rr], 1, 2)} | {sum(r['matches_reference_8'] for r in rr)}/{len(rr)} |")
    print("\nsave (prep + write):")
    for r in M:
        if r.get("label") == "save":
            print(f"- {r['system']} h{r['history_idx']} N={r['N']}: prep {r['prep_s']:.1f}s, {r['file_bytes']} B, write {r['file_write_s']:.2f}s fsync {r['file_fsync_s']:.2f}s")
        if r.get("label") == "error":
            print(f"- ERROR {r['system']} h{r['history_idx']} N={r['N']}: {r['error'][:200]}")
