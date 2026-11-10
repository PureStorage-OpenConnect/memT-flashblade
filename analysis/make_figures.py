#!/usr/bin/env python3
"""MemT-1.0 reproduction: regenerate every figure and table from results/*.jsonl.

Usage (from the repo root):
    .venv-analysis/bin/python analysis/make_figures.py --out figures/ --summary results/summary.md

Read-only on every input. Missing or partial inputs are tolerated: a live JSONL may end in a
half-written line (skipped), and a figure whose inputs are absent prints "not yet available".
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import sys
import traceback
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator  # noqa: E402

# ------------------------------------------------------------------ constants
KIB, MIB, GIB = 1024, 1024 ** 2, 1024 ** 3
NS = [512, 2048, 8192, 32768, 65536, 131072, 262144]

C_KV = "#2a78d6"      # full context / KV
C_MEM = "#eb6834"     # MemT-Nano / memory
C_LORA = "#1baf7a"    # Temp-LoRA
# E3 storage targets / E4 conditions (validated all-pairs, light surface; yellow needs labels)
C_T = {"rdma": "#6250d6", "tcp": "#e34948", "local": "#eda100"}
M_T = {"rdma": "o", "tcp": "s", "local": "^"}
TXT, TXT2, GRID, BG = "#171a1f", "#4d5565", "#e3e5e8", "#fcfcfb"
PAPER = "paper (hardware unspecified)"

# Paper reference values (MemT-1.0, Sec 6.5/6.6)
P_E2E = {"fullctx": {131072: 30.07, 262144: 94.47}, "nano": {131072: 16.19, 262144: 23.69},
         "lora": {131072: 700.82, 262144: 1400.88}}
P_SPEEDUP = {32768: 1.0, 65536: 1.20, 131072: 1.86, 262144: 3.99}
P_TTFT = {"fullctx": {131072: 0.257, 262144: 0.445}, "nano_range": (0.104, 0.112)}
P_TTFT_RATIO = {131072: 2.44, 262144: 4.19}
P_NANO_BYTES = 40 * MIB
P_LORA_BYTES = 172.5 * MIB
P_FC_256K = 5.061 * GIB
P_FC_PER_TOK, P_FC_CONST = 20 * KIB, 62.5 * MIB


def paper_fc_bytes(n):
    return P_FC_PER_TOK * n + P_FC_CONST


# ------------------------------------------------------------------ io helpers
def read_jsonl(path):
    """Return list of dicts; skip blank / half-written lines (live files)."""
    if not os.path.exists(path):
        return None
    out, bad = [], 0
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    if bad:
        warn(f"{os.path.basename(path)}: skipped {bad} unparsable line(s) (live append?)")
    return out


WARNINGS: list[str] = []
E4_NOTES: list[str] = []
QUALITY: list[str] = []
CAPTIONS: "OrderedDict[str, str]" = OrderedDict()


def warn(msg):
    WARNINGS.append(msg)
    print(f"[warn] {msg}", file=sys.stderr)


def quality(msg):
    if msg not in QUALITY:
        QUALITY.append(msg)


def fmt_tokens(n):
    n = float(n)
    if n >= 1024 and abs(n / 1024 - round(n / 1024)) < 1e-9:
        return f"{int(round(n / 1024))}K"
    return f"{int(n)}"


def fmt_bytes(b, digits=None):
    b = float(b)
    for unit, s in ((GIB, "GiB"), (MIB, "MiB"), (KIB, "KiB")):
        if b >= unit:
            v = b / unit
            if digits is None:
                return f"{v:.0f} {s}" if abs(v - round(v)) < 1e-6 else f"{v:.3g} {s}"
            return f"{v:.{digits}f} {s}"
    return f"{b:.0f} B"


def fmt_seconds(s):
    if s < 1:
        return f"{s * 1e3:.0f} ms"
    return f"{s:g} s"


def md_table(df, floatfmt=None):
    if df is None or len(df) == 0:
        return "_no data_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join("---:" if pd.api.types.is_numeric_dtype(df[c]) else "---" for c in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if v is None or (isinstance(v, float) and math.isnan(v)):
                cells.append("")
            elif isinstance(v, (float, np.floating)):
                f = (floatfmt or {}).get(c, "{:.3g}")
                cells.append(f.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ plotting style
def setup_style():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "axes.edgecolor": TXT2, "axes.labelcolor": TXT, "axes.titlecolor": TXT,
        "xtick.color": TXT2, "ytick.color": TXT2, "text.color": TXT,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2.0, "lines.markersize": 7,
        "font.size": 10.5, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "legend.frameon": False, "legend.fontsize": 9.5,
        "svg.fonttype": "none", "figure.dpi": 100,
    })


def n_axis(ax, ns=NS, lim=True):
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(ns))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_tokens(v)))
    ax.xaxis.set_minor_locator(NullLocator())
    if lim:
        ax.set_xlim(ns[0] / 1.4, ns[-1] * 1.4)
    ax.set_xlabel("history length N (tokens)")


def sec_axis(ax, lo, hi):
    ax.set_yscale("log")
    cands = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500,
             1000, 2000, 5000]
    ticks = [t for t in cands if lo / 1.05 <= t <= hi * 1.05]
    if len(ticks) > 8:
        ticks = [t for t in ticks if str(t).lstrip("0.").startswith(("1", "5")) or t in (2, 20, 200)]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_seconds(v)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(lo, hi)


def bytes_axis(ax, lo, hi, axis="y"):
    cands = [1 * MIB, 10 * MIB, 40 * MIB, 100 * MIB, 1 * GIB, 10 * GIB, 100 * GIB]
    ticks = [t for t in cands if lo <= t <= hi]
    a = ax.yaxis if axis == "y" else ax.xaxis
    (ax.set_yscale if axis == "y" else ax.set_xscale)("log")
    a.set_major_locator(FixedLocator(ticks))
    a.set_major_formatter(FuncFormatter(lambda v, _: fmt_bytes(v)))
    a.set_minor_formatter(NullFormatter())
    (ax.set_ylim if axis == "y" else ax.set_xlim)(lo, hi)


def end_label(ax, x, y, text, dx=6, dy=0, ha="left"):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", va="center", ha=ha,
                fontsize=9, color=TXT)


def paper_kw(color):
    return dict(color=color, linestyle="--", linewidth=1.5, marker="o", markersize=8,
                markerfacecolor=BG, markeredgecolor=color, markeredgewidth=1.8)


def save(fig, out, name, caption):
    os.makedirs(out, exist_ok=True)
    png = os.path.join(out, name + ".png")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(out, name + ".svg"), bbox_inches="tight")
    plt.close(fig)
    CAPTIONS[name] = caption
    print(f"[fig] {png}\n      caption: {caption}")


def not_available(name, why):
    msg = f"{name}: not yet available ({why})"
    CAPTIONS[name] = "NOT YET AVAILABLE: " + why
    print(f"[skip] {msg}")


# ------------------------------------------------------------------ E1 loading
def load_e1(res):
    rows, files = [], []
    for fn in sorted(os.listdir(res)) if os.path.isdir(res) else []:
        if fn.startswith("e1_sweep") and fn.endswith(".jsonl"):
            recs = read_jsonl(os.path.join(res, fn)) or []
            for r in recs:
                r["_file"] = fn
            rows += recs
            files.append(fn)
    if not rows:
        return None, None, files
    df = pd.json_normalize(rows, max_level=1)
    df["N"] = df["N"].astype(int)
    if "writer_kind" in df:
        df["writer_eff"] = np.where(df["system"] == "nano", df.get("writer", df["writer_kind"]), "n/a")
    allc = df.groupby(["N", "system", "status"]).size().rename("n").reset_index()
    ok = df[df["status"] == "ok"].copy()
    # dedupe re-runs of the same combo (keep latest)
    key = ["history_idx", "N", "system"]
    before = len(ok)
    ok = ok.sort_values("timestamp").drop_duplicates(key + (["writer_eff"] if "writer_eff" in ok else []),
                                                      keep="last")
    if len(ok) != before:
        quality(f"E1: {before - len(ok)} duplicate (history, N, system) records; kept the latest")
    return ok, allc, files


def e1_quality(ok):
    if ok is None:
        return
    d = (ok["history_tokens"] - ok["N"]).abs() / ok["N"]
    if (d > 0).any():
        quality(f"E1: history_tokens differs from nominal N in {(d > 0).sum()}/{len(ok)} records "
                f"(max {d.max() * 100:.2f}%; sampler tolerance is +-1%, so 'N' is a bucket, not an exact count)")
    e2e_chk = (ok["e2e_s"] - (ok["prep_s"] + ok["total_s_mean"])).abs().max()
    if e2e_chk > 1e-6:
        quality(f"E1: e2e_s != prep_s + total_s_mean (max diff {e2e_chk:.3g} s)")
    if "tokens_identical_across_reps" in ok and (~ok["tokens_identical_across_reps"].astype(bool)).any():
        quality("E1: some records have tokens_identical_across_reps = false (non-deterministic decode)")
    if "exceeds_max_position" in ok and ok["exceeds_max_position"].fillna(False).astype(bool).any():
        quality("E1: some fullctx records exceed max_position_embeddings")
    for (n, s), g in ok.groupby(["N", "system"]):
        med = g["ttft_s_mean"].median()
        hi = g[g["ttft_s_mean"] > 1.5 * med]
        if len(hi):
            quality(f"E1: TTFT outlier(s) at N={fmt_tokens(n)} {s}: histories {hi['history_idx'].tolist()} "
                    f"> 1.5x median {med * 1e3:.0f} ms")
        med = g["prep_s"].median()
        hi = g[g["prep_s"] > 2 * med]
        if len(hi) and med > 0.2:
            quality(f"E1: prep_s outlier(s) at N={fmt_tokens(n)} {s}: histories {hi['history_idx'].tolist()} "
                    f"({', '.join(f'{v:.2f}' for v in hi['prep_s'])} s vs median {med:.2f} s)")
    nano = ok[ok["system"] == "nano"]
    if len(nano) and "state.serialized_bytes" in nano:
        sb = nano["state.serialized_bytes"].unique()
        if len(sb) > 1:
            quality(f"E1: nano serialized_bytes not constant: {sorted(sb)}")
    if "slice_mode" in ok:
        modes = ok.loc[ok.system == "nano", "slice_mode"].dropna().unique().tolist()
        quality(f"E1 sweep nano slice_mode = {modes}; the validation/probe files used 'messages' slicing, so "
                "fast-writer prep there is not directly comparable with the sweep")
    quality("E1: writer_kind is null on reference-writer records in e1_validate/e1_probe256k; the 'writer' field "
            "is authoritative (writer_kind was added later)")
    quality("E1: restore_s is a pinned-host -> GPU copy (state already in host RAM), not a storage read; "
            "storage restore is E3")


def agg(ok, col):
    g = ok.groupby(["N", "system"])[col]
    return g.agg(["mean", "min", "max", "count"]).reset_index()


def paired_ratio(ok, col):
    a = ok[ok.system == "fullctx"][["history_idx", "N", col]]
    b = ok[ok.system == "nano"][["history_idx", "N", col]]
    m = a.merge(b, on=["history_idx", "N"], suffixes=("_fc", "_nano"))
    if not len(m):
        return pd.DataFrame(columns=["N", "mean", "min", "max", "count"])
    m["r"] = m[col + "_fc"] / m[col + "_nano"]
    return m.groupby("N")["r"].agg(["mean", "min", "max", "count"]).reset_index()


def plot_band_series(ax, d, color, label, marker="o"):
    d = d.sort_values("N")
    ax.fill_between(d["N"], d["min"], d["max"], color=color, alpha=0.15, linewidth=0)
    ax.plot(d["N"], d["mean"], color=color, marker=marker, label=label)


# ------------------------------------------------------------------ E1 figures
def fig_e1_latency(ok, out):
    name = "fig_e1_latency"
    if ok is None or not len(ok):
        return not_available(name, "no E1 sweep records")
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    labels = {"fullctx": "full context (ours, A100, HF)", "nano": "MemT-Nano, fast writer (ours, A100, HF)"}
    cols = {"fullctx": C_KV, "nano": C_MEM}
    a = agg(ok, "e2e_s")
    ymax = a["max"].max()
    for s in ("fullctx", "nano"):
        d = a[a.system == s]
        if len(d):
            plot_band_series(ax, d, cols[s], labels[s])
    for s, c, lab in (("fullctx", C_KV, "full context"), ("nano", C_MEM, "MemT-Nano"), ("lora", C_LORA, "Temp-LoRA")):
        xs, ys = zip(*sorted(P_E2E[s].items()))
        ax.plot(xs, ys, label=f"{lab}, {PAPER}", **paper_kw(c))
        # paper full context sits just above our Nano point at 256K: lift it, push ours below
        end_label(ax, xs[-1], ys[-1], f"{lab} (paper) {ys[-1]:.1f} s", dy=8 if s == "fullctx" else 0)
        ymax = max(ymax, max(ys))
    for s, nm in (("fullctx", "full context"), ("nano", "Nano")):
        d = a[a.system == s].sort_values("N")
        if len(d):
            r = d.iloc[-1]
            end_label(ax, r["N"], r["mean"], f"{nm} (ours) {r['mean']:.1f} s", dy=0 if s == "fullctx" else -9)
    n_axis(ax)
    ax.set_xlim(NS[0] / 1.4, NS[-1] * 4)
    sec_axis(ax, 5, ymax * 1.6)
    ax.set_ylabel("end-to-end latency (history prep + 128-token reply)")
    ax.set_title("E1  End-to-end latency vs history length")
    ax.legend(loc="upper left", fontsize=8.5)
    cov = ", ".join(f"{fmt_tokens(n)}:{int(c)}" for n, c in a.groupby("N")["count"].min().items())
    save(fig, out, name,
         "End-to-end latency (history preparation + 512-token query + forced 128-token decode), mean across "
         "histories with min-max band; HF transformers BF16 on 2x A100-80GB, batch 1. Hollow dashed markers are "
         f"the paper's values ({PAPER}). Histories per N (min over systems): {cov}. Decode runs at ~10-11 tok/s "
         "in HF here, so ~11-12 s of every point is the 128-token reply.")


def fig_e1_ttft(ok, out):
    name = "fig_e1_ttft"
    if ok is None or not len(ok):
        return not_available(name, "no E1 sweep records")
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    a = agg(ok, "ttft_s_mean")
    lo, hi = P_TTFT["nano_range"]
    ax.axhspan(lo, hi, color=C_MEM, alpha=0.10, linewidth=0)
    ax.plot([NS[0], NS[-1]], [(lo + hi) / 2] * 2, color=C_MEM, linestyle="--", linewidth=1.5,
            label=f"MemT-Nano 104-112 ms, {PAPER}")
    end_label(ax, NS[0], hi, "Nano (paper) 104-112 ms", dx=0, dy=8)
    xs, ys = zip(*sorted(P_TTFT["fullctx"].items()))
    ax.plot(xs, ys, label=f"full context, {PAPER}", **paper_kw(C_KV))
    for x, y in zip(xs, ys):
        end_label(ax, x, y, f"full context (paper) {y * 1e3:.0f} ms", dx=8, dy=-8 if x < NS[-1] else 4)
    for s, c, lab in (("fullctx", C_KV, "full context (ours)"), ("nano", C_MEM, "MemT-Nano, fast writer (ours)")):
        d = a[a.system == s]
        if len(d):
            plot_band_series(ax, d, c, lab)
            r = d.sort_values("N").iloc[-1]
            end_label(ax, r["N"], r["mean"], ("full context" if s == "fullctx" else "Nano") +
                      f" (ours) {r['mean'] * 1e3:.0f} ms", dy=0 if s == "fullctx" else -11)
    n_axis(ax)
    ax.set_xlim(NS[0] / 1.4, NS[-1] * 3)
    sec_axis(ax, 0.08, max(2.0, a["max"].max() * 1.5))
    ax.set_ylabel("query TTFT (includes state restore)")
    ax.set_title("E1  Query time-to-first-token vs history length")
    ax.legend(loc="upper left", fontsize=8.5)
    save(fig, out, name,
         "Query TTFT for a 512-token query (restore of the prepared state from pinned host memory + query "
         "prefill + first token), mean across histories with min-max band. Note: HF transformers' MoE expert "
         "loop sets a ~350 ms TTFT floor on A100 for any 512-token prefill, so our absolute TTFTs sit ~3x above "
         "the paper's 104-112 ms Nano figure; compare the shape (flat vs rising) and the ratio, not the level. "
         f"Paper values are hollow/dashed ({PAPER}).")


def mark_crossover(ax, d):
    d = d.sort_values("N").reset_index(drop=True)
    # last point below parity, followed only by points at/above parity that eventually clear it by >=5%
    below = [i for i in range(len(d)) if d.loc[i, "mean"] < 1]
    cands = [below[-1] + 1] if below and below[-1] + 1 < len(d) else []
    for i in cands:
        r0, r1 = d.loc[i - 1, "mean"], d.loc[i, "mean"]
        if d.loc[i:, "mean"].max() >= 1.05:
            x0, x1 = math.log2(d.loc[i - 1, "N"]), math.log2(d.loc[i, "N"])
            xc = 2 ** (x0 + (1 - r0) * (x1 - x0) / (r1 - r0))
            ax.axvline(xc, color=TXT2, linestyle=":", linewidth=1)
            ax.annotate(f"crossover ~{xc / 1024:.0f}K" if xc >= 1024 else f"crossover ~{xc:.0f}",
                        (xc, 1), xytext=(5, -16), textcoords="offset points", fontsize=9, color=TXT)
            return xc
    return None


def fig_e1_speedup(ok, out):
    name = "fig_e1_speedup"
    if ok is None or not len(ok):
        return not_available(name, "no E1 sweep records")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    notes = []
    for ax, col, paper, title in ((axes[0], "e2e_s", P_SPEEDUP, "End-to-end latency"),
                                   (axes[1], "ttft_s_mean", P_TTFT_RATIO, "Query TTFT")):
        d = paired_ratio(ok, col)
        ax.axhline(1, color=TXT2, linewidth=1)
        if len(d):
            plot_band_series(ax, d, C_MEM, "ours (A100, HF), per-history paired ratio")
            r = d.sort_values("N").iloc[-1]
            end_label(ax, r["N"], r["mean"], f"{r['mean']:.2f}x", dy=-10)
            xc = mark_crossover(ax, d)
            notes.append(f"{title}: ours crossover " + (f"~{xc:,.0f} tokens" if xc else
                                                        "not reached (within +-5% of parity or below) in measured range"))
        xs, ys = zip(*sorted(paper.items()))
        ax.plot(xs, ys, label=PAPER, **paper_kw(C_MEM))
        for x, y in zip(xs, ys):
            end_label(ax, x, y, f"{y:.2f}x", dx=-6, dy=10, ha="right")
        n_axis(ax)
        ax.set_title(title)
        ax.set_yscale("log", base=2)
        ax.yaxis.set_major_locator(FixedLocator([0.5, 0.75, 1, 1.5, 2, 3, 4, 6]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}x"))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_ylim(0.6, 6)
    axes[0].set_ylabel("full context / MemT-Nano  (>1x: Nano faster)")
    axes[0].legend(loc="upper left", fontsize=8.5)
    fig.suptitle("E1  MemT-Nano speedup over full context", x=0.01, ha="left", fontweight="bold",
                 color=TXT)
    fig.tight_layout()
    save(fig, out, name,
         "Ratio full-context / MemT-Nano (fast writer) for end-to-end latency and for query TTFT, computed per "
         "history and averaged (band = min-max across histories). Horizontal line = parity. "
         + "; ".join(notes) + f". Paper ratios hollow/dashed ({PAPER}); paper puts e2e parity at ~32K.")


def load_validation(res):
    rows = []
    for fn in ("e1_validate.jsonl", "e1_probe256k.jsonl"):
        recs = read_jsonl(os.path.join(res, fn))
        if recs is None:
            warn(f"{fn} missing")
            continue
        for r in recs:
            r["_file"] = fn
        rows += recs
    if not rows:
        return None
    df = pd.json_normalize(rows, max_level=1)
    df = df[df.status == "ok"].copy()
    df["N"] = df["N"].astype(int)
    df["series"] = np.where(df.system == "fullctx", "fullctx",
                            "nano-" + df["writer"].fillna(df.get("writer_kind")).astype(str))
    return df


def fig_e1_writer_cost(val, out):
    name = "fig_e1_writer_cost"
    if val is None or not len(val):
        return not_available(name, "e1_validate.jsonl / e1_probe256k.jsonl missing")
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    spec = [("nano-reference", C_MEM, "s", ":", "MemT-Nano, reference writer (shipped per-token loop)"),
            ("nano-fast", C_MEM, "o", "-", "MemT-Nano, fast writer"),
            ("fullctx", C_KV, "o", "-", "full context, KV prefill")]
    for key, c, m, ls, lab in spec:
        d = val[val.series == key]
        if not len(d):
            continue
        ax.scatter(d["N"], d["prep_s"], s=18, color=c, marker=m, alpha=0.45, linewidths=0)
        med = d.groupby("N")["prep_s"].agg(["median", "count"]).reset_index()
        ax.plot(med["N"], med["median"], color=c, marker=m, linestyle=ls, label=lab,
                markerfacecolor=c)
        r = med.iloc[-1]
        end_label(ax, r["N"], r["median"], f"{r['median']:.0f} s" if r["median"] >= 10 else f"{r['median']:.1f} s")
    n_axis(ax)
    ax.set_xlim(NS[0] / 1.4, NS[-1] * 2.5)
    sec_axis(ax, 0.2, val["prep_s"].max() * 2)
    ax.set_ylabel("history preparation time")
    ax.set_title("E1  Cost of writing history into memory vs KV prefill")
    ax.legend(loc="upper left", fontsize=8.5)
    nh = val.groupby(["series", "N"])["history_idx"].nunique()
    save(fig, out, name,
         "History-preparation time: MemT-Nano commit_memory with the shipped reference writer vs our fast writer, "
         "and full-context KV prefill. Line = median, dots = individual histories. Source: e1_validate.jsonl + "
         "e1_probe256k.jsonl, which hold only 1-5 histories per point (" +
         ", ".join(f"{s}@{fmt_tokens(n)}:{c}" for (s, n), c in nh.items()) +
         "); nano slices packed by message ('messages' slice mode), the main sweep uses 'dense' slicing. "
         "One fullctx point (history 0 @2K, 9.1 s) is a first-run warm-up outlier; the median ignores it.")


# ------------------------------------------------------------------ E2
def e2_table(ok, val):
    frames = []
    for src, df in (("sweep", ok), ("validation", val)):
        if df is None or not len(df):
            continue
        d = df.copy()
        d["_src"] = src
        frames.append(d)
    if not frames:
        return None
    d = pd.concat(frames, ignore_index=True)
    rows = []
    for n in sorted(d.N.unique()):
        g = d[d.N == n]
        nano = g[g.system == "nano"]["state.serialized_bytes"].dropna()
        fc = g[g.system == "fullctx"]
        row = {"N": n}
        row["nano_bytes"] = nano.mean() if len(nano) else np.nan
        if len(fc) and "state.serialized_bytes" in fc:
            ser = fc["state.serialized_bytes"]
            delta = fc["state.gdn_recurrent_bytes_if_fp32"] - fc["state.gdn_recurrent_bytes"]
            row["fc_bf16_bytes"] = ser.mean()
            row["fc_fp32_bytes"] = (ser + delta).mean()
            row["kv_bytes_per_token"] = fc["state.kv_bytes_per_token"].mean()
            row["fc_n"] = len(fc)
            row["src"] = "+".join(sorted(fc["_src"].unique()))
        rows.append(row)
    t = pd.DataFrame(rows)
    return t


def fit_fc(t, col):
    d = t.dropna(subset=[col])
    if len(d) < 2:
        return None
    a, b = np.polyfit(d["N"].astype(float), d[col].astype(float), 1)
    return a, b


def fig_e2_storage(t, out):
    name = "fig_e2_storage"
    if t is None or not len(t) or "fc_bf16_bytes" not in t:
        return not_available(name, "no E1 state-size fields")
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    d = t.dropna(subset=["fc_bf16_bytes"])
    ax.plot(d["N"], d["fc_bf16_bytes"], color=C_KV, marker="o", label="full context, measured (BF16 GDN state, as HF stores it)")
    ax.plot(d["N"], d["fc_fp32_bytes"], color=C_KV, marker="^", linestyle=":",
            label="full context, FP32 GDN state (paper's accounting)")
    dn = t.dropna(subset=["nano_bytes"])
    ax.plot(dn["N"], dn["nano_bytes"], color=C_MEM, marker="o", label="MemT-Nano snapshot, measured")
    # paper
    ax.plot([131072, 262144], [P_NANO_BYTES] * 2, label=f"MemT-Nano 40 MiB, {PAPER}", **paper_kw(C_MEM))
    ax.plot([131072, 262144], [paper_fc_bytes(131072), P_FC_256K],
            label=f"full context 20 KiB/token + 62.5 MiB, {PAPER}", **paper_kw(C_KV))
    end_label(ax, 262144, P_FC_256K, "5.061 GiB (paper)", dy=10)
    ax.axhline(P_LORA_BYTES, color=C_LORA, linestyle="--", linewidth=1.5, label=f"Temp-LoRA 172.5 MiB, {PAPER}")
    end_label(ax, NS[0], P_LORA_BYTES, "Temp-LoRA 172.5 MiB (paper)", dx=0, dy=8)
    # ratio annotation at 128K
    n = 131072
    nano = dn["nano_bytes"].mean() if len(dn) else P_NANO_BYTES
    txt = []
    for col, lab in (("fc_bf16_bytes", "BF16 GDN"), ("fc_fp32_bytes", "FP32 GDN")):
        meas = t.loc[t.N == n, col].dropna()
        if len(meas):
            v, how = meas.iloc[0], "measured"
        else:
            f = fit_fc(t, col)
            if f is None:
                continue
            v, how = f[0] * n + f[1], "linear fit"
        txt.append(f"{lab}: {v / nano:.0f}x ({how})")
    if txt:
        ax.annotate("at 128K, full context / Nano\n" + "\n".join(txt) + "\npaper: ~66x",
                    (n, paper_fc_bytes(n)), xytext=(20000, 60 * MIB), textcoords="data", fontsize=9,
                    color=TXT, arrowprops=dict(arrowstyle="-", color=TXT2, linewidth=0.8))
    n_axis(ax)
    ax.set_xlim(NS[0] / 1.4, NS[-1] * 1.6)
    bytes_axis(ax, 20 * MIB, 12 * GIB)
    ax.set_ylabel("serialized per-trajectory state (safetensors)")
    ax.set_title("E2  Persistent state per conversation vs history length")
    ax.legend(loc="upper left", fontsize=8)
    save(fig, out, name,
         "Serialized per-trajectory state after history prep (safetensors bytes). Full context: KV of the 10 "
         "attention layers (20 KiB/token) + 30 Gated-DeltaNet layers' recurrent and conv state. HF stores the "
         "GDN recurrent state in BF16 (~30 MiB); the paper counts it in FP32 (~60 MiB), shown dotted (derived "
         "from the measured record by swapping the recurrent-state dtype). Nano: 10 x FP32 [1,16,256,256] "
         f"memory = 40 MiB, constant. Paper values hollow/dashed ({PAPER}). Sources: E1 sweep + validation/probe "
         "records (state bytes are deterministic given the token count).")


# ------------------------------------------------------------------ E3
def load_e3(res):
    recs = read_jsonl(os.path.join(res, "e3_persist_io.jsonl"))
    if not recs:
        return None, None
    df = pd.json_normalize(recs, max_level=0)
    df = df[df.op != "skipped"].copy() if "op" in df else df
    skipped = [r for r in recs if r.get("op") == "skipped"]
    for r in skipped:
        quality(f"E3: {r['target']} {r['label']} K={r.get('concurrency')} skipped: {r.get('reason')}")
    rc = read_jsonl(os.path.join(res, "e3_recheck_rdma_small.jsonl"))
    replaced = []
    if rc:
        r = pd.json_normalize(rc, max_level=0)
        r = r.sort_values("ts").drop_duplicates(["target", "label", "op", "concurrency"], keep="last")
        k = ["target", "label", "op", "concurrency"]
        keys = set(map(tuple, r[k].values.tolist()))
        mask = df[k].apply(tuple, axis=1).isin(keys)
        newv = r.set_index(k)["agg_gbps_median"]
        for _, o in df[mask & (df.op == "conc_read_cold")].iterrows():
            nv = newv.get((o.target, o.label, o.op, o.concurrency))
            if nv is None or abs(nv - o.agg_gbps_median) < 0.25 * nv:
                continue
            quality(f"E3: original full-run cell {o.target} {o.label} K={o.concurrency} read "
                    f"{o.agg_gbps_median:.2f} GB/s, p95 {o.lat_p95_s * 1e3:.0f} ms; recheck reads {nv:.2f} GB/s "
                    "(e3_recheck_rdma_small.jsonl, used instead)")
        replaced = sorted({f"{a} {b} {c} K={d}" for a, b, c, d in keys})
        df = pd.concat([df[~mask], r.assign(_recheck=True)], ignore_index=True)
    df["_recheck"] = df.get("_recheck", False)
    df["_recheck"] = df["_recheck"].fillna(False).astype(bool)
    return df, replaced


def obj_order(label):
    if label == "memt":
        return (0, 0)
    if label == "lora":
        return (2, 0)
    return (1, int(label.split("@")[1]))


def obj_name(label):
    if label.startswith("kv@"):
        return "kv@" + fmt_tokens(int(label.split("@")[1]))
    return label


def fig_e3_restore_latency(e3, out):
    name = "fig_e3_restore_latency"
    if e3 is None:
        return not_available(name, "results/e3_persist_io.jsonl missing")
    d = e3[(e3.op == "read_cold") & (e3.concurrency == 1)]
    if not len(d):
        return not_available(name, "no read_cold records")
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for t in ("rdma", "tcp", "local"):
        g = d[d.target == t].sort_values("bytes")
        if not len(g):
            continue
        kv = g[g.object == "kv"]
        ax.plot(kv["bytes"], kv["median_s"], color=C_T[t], marker=M_T[t], label=f"{t} (KV objects)")
        oth = g[g.object != "kv"]
        zs = {"rdma": (75, 3), "tcp": (50, 4), "local": (30, 5)}[t]  # nested so overlapping points stay visible
        ax.scatter(oth["bytes"], oth["median_s"], color=C_T[t], marker=M_T[t], s=zs[0], zorder=zs[1],
                   edgecolors=BG, linewidths=1)
    # object labels at max over targets
    for lab, g in d.groupby("label"):
        y = g["median_s"].max()
        x = g["bytes"].iloc[0]
        ax.annotate(obj_name(lab), (x, y), xytext=(0, 9), textcoords="offset points", ha="center",
                    fontsize=8.5, color=TXT, fontweight="bold" if lab in ("memt", "lora") else "normal")
    bytes_axis(ax, 25 * MIB, 8 * GIB, axis="x")
    ax.set_xlabel("object size (memt = MemT-Nano snapshot, kv@N = full-context state at N, lora = Temp-LoRA)")
    sec_axis(ax, 0.004, 2.0)
    ax.set_ylabel("cold restore time (O_DIRECT read, median)")
    ax.set_title("E3  Single-session cold restore from FlashBlade and local NVMe")
    ax.legend(loc="upper left", fontsize=8.5, title="target", title_fontsize=9)
    save(fig, out, name,
         "One session (K=1) reading its state object with O_DIRECT (client page cache bypassed; storage-side "
         "cache may be warm), median of 5 reps. rdma = FlashBlade NFSv3 over RDMA, tcp = FlashBlade NFSv4.1 over "
         "TCP (both nconnect=16, 200 GbE), local = NVMe RAID ext4. Isolated markers are memt (40 MiB) and lora "
         "(172.5 MiB). kv@N objects are synthetic with the paper's FP32-GDN layout (e3_persist_io.py).")


def conc_cells(e3, labels):
    d = e3[(e3.op == "conc_read_cold") & (e3.label.isin(labels))].copy()
    return d


def fig_e3_concurrency(e3, out):
    name, name2 = "fig_e3_concurrency", "fig_e3_concurrency_latency"
    if e3 is None:
        not_available(name, "results/e3_persist_io.jsonl missing")
        return not_available(name2, "results/e3_persist_io.jsonl missing")
    labels = ["memt", "kv@32768", "kv@262144"]
    d = conc_cells(e3, labels)
    if not len(d):
        not_available(name, "no conc_read_cold records")
        return not_available(name2, "no conc_read_cold records")
    ks = sorted(d.concurrency.unique())
    rc_note = ""
    if d["_recheck"].any():
        rc_note = " rdma memt cells come from e3_recheck_rdma_small.jsonl (re-measurement of the full run)."
    # aggregate GB/s
    fig, axes = plt.subplots(1, len(labels), figsize=(11, 3.9), sharey=True)
    for ax, lab in zip(axes, labels):
        for t in ("rdma", "tcp", "local"):
            g = d[(d.target == t) & (d.label == lab)].sort_values("concurrency")
            if len(g):
                ax.plot(g["concurrency"], g["agg_gbps_median"], color=C_T[t], marker=M_T[t], label=t)
                r = g.iloc[-1]
                end_label(ax, r["concurrency"], r["agg_gbps_median"], f"{r['agg_gbps_median']:.1f}")
        ax.set_title(f"{obj_name(lab)} ({fmt_bytes(d[d.label == lab]['bytes'].iloc[0], 0)}/session)")
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(FixedLocator(ks))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlim(min(ks) / 1.3, max(ks) * 2.2)
        ax.set_xlabel("concurrent sessions K")
    axes[0].set_ylabel("aggregate restore throughput (GB/s)")
    axes[0].set_ylim(0, max(26, d["agg_gbps_median"].max() * 1.1))
    axes[0].legend(loc="upper left", fontsize=8.5, title="target", title_fontsize=9)
    fig.suptitle("E3  Aggregate cold-restore throughput vs concurrent sessions", x=0.01, ha="left",
                 fontweight="bold", color=TXT)
    fig.tight_layout()
    save(fig, out, name,
         "K processes each read a distinct state file with O_DIRECT, released together by a barrier; "
         "aggregate GB/s = K x bytes / wall time, median over reps (GB = 1e9 B). Client is one DGX A100." + rc_note)
    # latency
    fig, axes = plt.subplots(1, len(labels), figsize=(11, 3.9))
    for ax, lab in zip(axes, labels):
        for t in ("rdma", "tcp", "local"):
            g = d[(d.target == t) & (d.label == lab)].sort_values("concurrency")
            if len(g):
                ax.plot(g["concurrency"], g["lat_p50_s"], color=C_T[t], marker=M_T[t], label=f"{t} p50")
                ax.plot(g["concurrency"], g["lat_p95_s"], color=C_T[t], marker=M_T[t], linestyle=":",
                        linewidth=1.5, markersize=6, markerfacecolor=C_T[t], alpha=0.55, label=f"{t} p95")
        ax.set_title(f"{obj_name(lab)}")
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(FixedLocator(ks))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlim(min(ks) / 1.3, max(ks) * 1.3)
        ax.set_xlabel("concurrent sessions K")
        sub = d[d.label == lab]
        sec_axis(ax, max(0.003, sub["lat_p50_s"].min() / 1.6), sub["lat_p95_s"].max() * 1.6)
    axes[0].set_ylabel("per-session restore latency")
    axes[0].legend(loc="upper left", fontsize=7.5, ncol=1)
    fig.suptitle("E3  Per-session cold-restore latency (p50 solid, p95 dotted) vs concurrent sessions",
                 x=0.01, ha="left", fontweight="bold", color=TXT)
    fig.tight_layout()
    save(fig, out, name2,
         "Per-session wall time from barrier release to last byte, p50 (solid) and p95 (dotted) across sessions "
         "and reps; one y-scale per panel (log)." + rc_note)


def fig_e3_loader_overhead(e3, out, target="rdma"):
    name = "fig_e3_loader_overhead"
    if e3 is None:
        return not_available(name, "results/e3_persist_io.jsonl missing")
    d = e3[(e3.target == target) & (e3.op.isin(["read_cold", "st_load_cold"])) & (e3.concurrency == 1)]
    if not len(d):
        return not_available(name, "no read_cold/st_load_cold records")
    p = d.pivot_table(index="label", columns="op", values="median_s").dropna()
    p = p.loc[sorted(p.index, key=obj_order)]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    y = np.arange(len(p))[::-1]
    for (_, r), yy in zip(p.iterrows(), y):
        ax.plot([r["read_cold"], r["st_load_cold"]], [yy, yy], color=GRID, linewidth=3, zorder=1)
        ax.annotate(f"{r['st_load_cold'] / r['read_cold']:.1f}x", (r["st_load_cold"], yy), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9, color=TXT)
    ax.scatter(p["read_cold"], y, color=C_T["rdma"], s=55, zorder=3, label="raw O_DIRECT read")
    ax.scatter(p["st_load_cold"], y, color=C_T["tcp"], marker="D", s=50, zorder=3,
               label="O_DIRECT read + bytes() copy + safetensors.numpy.load")
    ax.set_yticks(y)
    ax.set_yticklabels([obj_name(l) for l in p.index])
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.7, len(p) - 0.3)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([0.01, 0.1, 1, 10]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_seconds(v)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(0.004, p["st_load_cold"].max() * 3)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([obj_name(l) for l in p.index]))
    ax.set_xlabel("time to restore one object (K=1, median)")
    ax.set_title(f"E3  Loader overhead: raw read vs safetensors load ({target})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8.5)
    save(fig, out, name,
         f"FlashBlade {target}, one session. Both paths read with O_DIRECT; the library path then copies into a "
         "Python bytes object and parses with safetensors.numpy.load, which caps it near 0.67 GB/s on every "
         "target (CPU-bound copy, not storage). Label = slowdown factor.")


def load_conc_sweep(res):
    recs = read_jsonl(os.path.join(res, "e3_conc_sweep_small.jsonl"))
    if not recs:
        return None
    d = pd.json_normalize(recs, max_level=0)
    if "op" not in d:
        return None
    d = d[d.op == "conc_read_cold"].copy()
    if not len(d):
        return None
    d = d.sort_values("ts").drop_duplicates(["target", "label", "concurrency"], keep="last") if "ts" in d else d
    d["concurrency"] = d["concurrency"].astype(int)
    return d


def fig_e3_conc_sweep(cs, out):
    name = "fig_e3_conc_sweep"
    if cs is None:
        return not_available(name, "results/e3_conc_sweep_small.jsonl missing")
    labels = [l for l in ("memt", "kv@512") if l in set(cs.label)] or sorted(cs.label.unique(), key=obj_order)
    ks = sorted(cs.concurrency.unique())
    fig, axes = plt.subplots(2, len(labels), figsize=(5.4 * len(labels), 7.0), sharex=True, squeeze=False)
    gmax = max(20, cs["agg_gbps_max"].max() * 1.1)
    def place(ax, items, close):
        # items: (x, y, text); nudge apart end labels whose values nearly coincide
        items = sorted(items, key=lambda it: it[1])
        for i, (x, y, txt) in enumerate(items):
            dy = 0
            if i > 0 and close(items[i - 1][1], y):
                dy = 7
            elif i + 1 < len(items) and close(y, items[i + 1][1]):
                dy = -7
            end_label(ax, x, y, txt, dy=dy)

    for j, lab in enumerate(labels):
        top, bot = axes[0][j], axes[1][j]
        tl, bl = [], []
        for t in ("rdma", "tcp", "local"):
            g = cs[(cs.label == lab) & (cs.target == t)].sort_values("concurrency")
            if not len(g):
                continue
            c, m = C_T[t], M_T[t]
            top.fill_between(g["concurrency"], g["agg_gbps_min"], g["agg_gbps_max"], color=c, alpha=0.15,
                             linewidth=0)
            top.plot(g["concurrency"], g["agg_gbps_median"], color=c, marker=m, label=f"{t} median (band = min-max)")
            bot.plot(g["concurrency"], g["lat_p95_s"], color=c, marker=m, label=f"{t} p95")
            r = g.iloc[-1]
            tl.append((r["concurrency"], r["agg_gbps_median"], f"{t} {r['agg_gbps_median']:.1f}"))
            bl.append((r["concurrency"], r["lat_p95_s"], f"{t} {r['lat_p95_s'] * 1e3:.0f} ms"))
        place(top, tl, lambda a, b: abs(b - a) < 0.08 * gmax)
        place(bot, bl, lambda a, b: b / a < 1.35)
        b = cs[cs.label == lab]["bytes"].iloc[0] if "bytes" in cs else None
        top.set_title(f"{obj_name(lab)}" + (f" ({fmt_bytes(b, 1)}/session)" if b else ""))
        top.set_ylim(0, gmax)
        top.set_ylabel("aggregate restore GB/s" if j == 0 else "")
        sub = cs[cs.label == lab]
        sec_axis(bot, max(0.003, sub["lat_p95_s"].min() / 1.6), sub["lat_p95_s"].max() * 1.8)
        bot.set_ylabel("p95 per-session latency" if j == 0 else "")
        for ax in (top, bot):
            ax.set_xscale("log", base=2)
            ax.xaxis.set_major_locator(FixedLocator(ks))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_xlim(min(ks) / 1.3, max(ks) * 3.2)
        bot.set_xlabel("concurrent sessions K")
    axes[0][0].legend(loc="upper left", fontsize=8)
    axes[1][0].legend(loc="upper left", fontsize=8)
    fig.suptitle("E3  Small-object cold-restore concurrency sweep (FlashBlade, idle storage path)", x=0.01,
                 ha="left", fontweight="bold", color=TXT)
    fig.tight_layout()
    runs = ", ".join(sorted(cs["run_id"].dropna().unique())) if "run_id" in cs else "?"
    reps = int(cs["reps"].median()) if "reps" in cs else "?"
    save(fig, out, name,
         f"Source e3_conc_sweep_small.jsonl (run {runs}), {reps} reps per point, run with the storage path "
         "otherwise idle. K processes each read a distinct file with O_DIRECT behind a barrier; top = aggregate "
         "GB/s (median, band = min-max over reps), bottom = p95 per-session latency. This sweep supersedes the "
         "K=8/32 small-object cells of e3_recheck_rdma_small.jsonl for the question of whether the RDMA dip is "
         "real: tcp stays flat at ~16-17 GB/s from K=8 up, while rdma drops at K=32 (memt median 5.2 GB/s, but "
         "max 14.4 GB/s, so the dip is intermittent across reps). The fig_e3_concurrency figures keep their "
         "original sources.")


def conc_sweep_table(cs):
    rows = []
    for lab in sorted(cs.label.unique(), key=obj_order):
        for t in ("rdma", "tcp", "local"):
            g = cs[(cs.label == lab) & (cs.target == t)].sort_values("concurrency")
            for _, r in g.iterrows():
                rows.append({"object": obj_name(lab), "target": t, "K": int(r["concurrency"]),
                             "GB/s median [min-max]": f"{r['agg_gbps_median']:.2f} [{r['agg_gbps_min']:.2f}-"
                                                      f"{r['agg_gbps_max']:.2f}]",
                             "p50 ms": r["lat_p50_s"] * 1e3, "p95 ms": r["lat_p95_s"] * 1e3,
                             "per-session GB/s": r.get("per_session_gbps_p50", np.nan)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ E4
def load_e4_ttft(res):
    """Return (df, source_desc, feasibility_flag)."""
    p = os.path.join(res, "e4_pkva_resume.jsonl")
    if os.path.exists(p):
        recs = read_jsonl(p) or []
        keep = ("cold", "hot", "restored")
        rows, other = [], {}
        for r in recs:
            lab = r.get("label")
            if lab not in keep:
                other[lab] = other.get(lab, 0) + 1
                continue
            if r.get("ttft_s") is None or r.get("N") is None:
                continue
            if lab == "restored" and r.get("valid") is not True:
                other["restored (valid != True)"] = other.get("restored (valid != True)", 0) + 1
                continue
            rows.append({k: r.get(k) for k in ("run_id", "label", "N", "history_idx", "history_tokens",
                                               "prompt_tokens", "ttft_s", "total_s", "cached_tokens",
                                               "expected_hit_tokens", "sample", "bytes_written",
                                               "files_written")})
        if rows:
            df = pd.DataFrame(rows)
            df["N"] = df["N"].astype(int)
            df["n_hist"] = df["N"]
            df["config"] = "B"  # single vLLM config in this run (checkpoint every 16 blocks)
            E4_NOTES.append("e4_pkva_resume.jsonl run " + ", ".join(sorted(df["run_id"].dropna().unique()))
                            + f": plotted {', '.join(f'{k}={v}' for k, v in df.label.value_counts().items())}; "
                            "excluded from plots: " + ", ".join(f"{k}={v}" for k, v in other.items())
                            + " (filler_* evict the kva host pool, first_load_after_kva_start is the first "
                            "load after a kva_server restart, end holds store_final)")
            fin = [r.get("store_final") for r in recs if r.get("label") == "end" and r.get("store_final")]
            if fin:
                E4_NOTES.append(f"pKVA store at end of run: {fmt_bytes(fin[-1]['bytes'], 2)} in "
                                f"{fin[-1]['files']} files (includes fillers)")
            return df, None, "e4_pkva_resume.jsonl", False
    frames, stores = [], []
    for cfg in ("A", "B"):
        recs = read_jsonl(os.path.join(res, f"e4_feasibility_config{cfg}.jsonl"))
        if not recs:
            continue
        s = [dict(r, config=cfg) for r in recs if "ttft_s" in r]
        frames.append(pd.json_normalize(s, max_level=0))
        stores += [dict(r["kvstore"], config=cfg) for r in recs if "kvstore" in r]
    if not frames:
        return None, None, None, True
    return pd.concat(frames, ignore_index=True), stores, "e4_feasibility_config{A,B}.jsonl", True


COND_STYLE = {"cold": (C_T["tcp"], "s", "cold (prefill computed)"),
              "hot": (C_T["rdma"], "o", "hot (GPU prefix-cache hit)"),
              "reset": (C_T["local"], "^", "GPU cache reset, kva_server retained"),
              "restored": (C_KV, "D", "restored from FlashBlade (servers restarted)")}


def fig_e4_pkva_ttft(e4, src, feas, out):
    name = "fig_e4_pkva_ttft"
    if e4 is None:
        return not_available(name, "no E4 files")
    cfgs = list(dict.fromkeys(e4["config"]))
    fig, axes = plt.subplots(1, len(cfgs), figsize=(5.2 * len(cfgs), 4.3), sharey=True, squeeze=False)
    ns = sorted(e4["n_hist"].unique())
    for ax, cfg in zip(axes[0], cfgs):
        g = e4[e4.config == cfg]
        for cond, (c, m, lab) in COND_STYLE.items():
            h = g[g.label == cond]
            if not len(h):
                continue
            med = h.groupby("n_hist")["ttft_s"].agg(["median", "min", "max"]).reset_index()
            med["ttft_s"] = med["median"]
            if h.groupby("n_hist").size().max() > 1:
                ax.fill_between(med["n_hist"], med["min"], med["max"], color=c, alpha=0.15, linewidth=0)
            ax.plot(med["n_hist"], med["ttft_s"], color=c, marker=m, label=lab)
            r = med.iloc[-1]
            end_label(ax, r["n_hist"], r["ttft_s"], fmt_seconds(round(r["ttft_s"], 2)) if r["ttft_s"] >= 1
                      else f"{r['ttft_s'] * 1e3:.0f} ms")
        n_axis(ax, ns)
        ax.set_xlim(ns[0] / 1.4, ns[-1] * 2.6)
        ax.set_title(f"config {cfg}" + (" (checkpoint every chunk)" if cfg == "A" else
                                         " (checkpoint every 16 blocks)" if cfg == "B" else "")
                     if feas else "pKVA on FlashBlade RDMA, config B (MNBT 16896)")
        sec_axis(ax, 0.04, max(20, e4["ttft_s"].max() * 1.5))
    axes[0][0].set_ylabel("TTFT, 512-token query (vLLM TP2 + pKVA)")
    axes[0][0].legend(loc="upper left", fontsize=8)
    tag = "FEASIBILITY (1 history, 1 sample per point) " if feas else ""
    if not feas:
        cnt = e4.groupby("n_hist")["history_idx"].nunique()
        tag = ""
    fig.suptitle(f"E4  {tag}Full-context TTFT on vLLM with KV persisted to FlashBlade", x=0.01, ha="left",
                 fontweight="bold", color=TXT)
    fig.tight_layout()
    if not feas:
        save(fig, out, name,
             f"Source {src}. Qwen3.6-35B-A3B on vLLM 0.32 TP2 + pKVA 1.3.0, KV store on FlashBlade NFSv3/RDMA, "
             "512-token query. Median per N with min-max band over "
             + ", ".join(f"{fmt_tokens(n)}:{c}" for n, c in cnt.items()) + " histories. cold = first request "
             "(prefill computed, KV written); hot = repeat with GPU prefix-cache hit; restored = after a kva_server "
             "restart plus a filler history to evict host/GPU caches, so the KV is read back from FlashBlade "
             "(only samples with valid == True, i.e. cached tokens matched the expected hit).")
        return
    save(fig, out, name,
         f"Source {src}. Qwen3.6-35B-A3B on vLLM TP2 with PureKVConnector writing KV to FlashBlade. "
         "cold = no cache; hot = GPU prefix cache; reset = POST /reset_prefix_cache with kva_server alive "
         "(may hit kva's host pool); restored = after restarting vLLM/kva_server, so KV comes from FlashBlade. "
         + ("Feasibility data: config B's 'restored' rows re-read the h1 objects written under config A; the "
            "config A 8K 'restored' point (1.98 s) is the first request after restart." if feas else ""))


def e4_bytes(e4, stores):
    if e4 is not None and "bytes_written" in e4 and e4["bytes_written"].notna().any():
        c = e4[(e4.label == "cold") & e4["bytes_written"].notna()].copy()
        c["bpt"] = c["bytes_written"] / c["history_tokens"]
        g = c.groupby(["config", "N"]).agg(prompt_tokens=("history_tokens", "median"),
                                           delta=("bytes_written", "median"), files=("files_written", "median"),
                                           bytes_per_token=("bpt", "median"), n_hist=("history_idx", "nunique"))
        g = g.reset_index().rename(columns={"N": "n"})
        g.attrs["tokens"] = "history tokens"
        return g
    if not stores:
        return None
    s = pd.DataFrame(stores)
    s = s[s["cond"] == "cold"].copy()
    s["delta"] = s["after"] - s["before"]
    if e4 is not None and "prompt_tokens" in e4:
        pt = e4[e4.label == "cold"].groupby(["config", "n_hist"])["prompt_tokens"].first().rename("prompt_tokens")
        s = s.merge(pt.reset_index(), left_on=["config", "n"], right_on=["config", "n_hist"], how="left")
    else:
        s["prompt_tokens"] = s["n"]
    s["bytes_per_token"] = s["delta"] / s["prompt_tokens"]
    s["files"] = s["files_after"] - s["files_before"]
    s = s[["config", "n", "prompt_tokens", "delta", "files", "bytes_per_token"]]
    s.attrs["tokens"] = "prompt tokens"
    return s


def fig_e4_bytes(bt, feas, out):
    name = "fig_e4_bytes_per_token"
    if bt is None or not len(bt):
        return not_available(name, "no kvstore records in E4 files")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ns = sorted(bt["n"].unique())
    styles = {"A": (C_KV, "o", "-"), "B": (C_KV, "s", ":")}
    lab = {"A": "config A (checkpoint every chunk)", "B": "config B (every 16 blocks)"}
    if not feas:
        lab = {"B": "pKVA config B (every 16 blocks), median over histories"}
        styles = {"B": (C_KV, "o", "-")}
    for cfg, g in bt.groupby("config"):
        c, m, ls = styles.get(cfg, (C_KV, "o", "-"))
        g = g.sort_values("n")
        axes[0].plot(g["n"], g["bytes_per_token"], color=c, marker=m, linestyle=ls, label=lab.get(cfg, cfg))
        axes[1].plot(g["n"], g["delta"], color=c, marker=m, linestyle=ls, label=lab.get(cfg, cfg))
        r = g.iloc[-1]
        end_label(axes[0], r["n"], r["bytes_per_token"], f"{r['bytes_per_token'] / KIB:.1f} KiB/tok")
        end_label(axes[1], r["n"], r["delta"], fmt_bytes(r["delta"], 2))
    axes[0].axhline(20 * KIB, color=TXT2, linestyle="-", linewidth=1.2, label="attention KV only, 20 KiB/token")
    end_label(axes[0], ns[0], 20 * KIB, "20 KiB/token (attention-only KV)", dx=0, dy=-10)
    axes[0].set_yscale("log")
    axes[0].yaxis.set_major_locator(FixedLocator([10 * KIB, 20 * KIB, 50 * KIB, 100 * KIB]))
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / KIB:.0f} KiB"))
    axes[0].yaxis.set_minor_formatter(NullFormatter())
    axes[0].set_ylim(10 * KIB, 160 * KIB)
    axes[0].set_ylabel(f"bytes on FlashBlade per {bt.attrs.get('tokens', 'prompt token').rstrip('s')}")
    axes[0].set_title("Per token")
    xs = np.array([ns[0], ns[-1]], dtype=float)
    axes[1].plot(xs, 20 * KIB * xs, color=TXT2, linewidth=1.2, label="20 KiB/token x N")
    axes[1].axhline(40 * MIB, color=C_MEM, linewidth=2, label="MemT-Nano snapshot, 40 MiB")
    end_label(axes[1], ns[0], 40 * MIB, "MemT-Nano 40 MiB", dx=0, dy=8)
    bytes_axis(axes[1], 20 * MIB, 40 * GIB)
    axes[1].set_ylabel("bytes on FlashBlade per session")
    axes[1].set_title("Per session")
    for ax in axes:
        n_axis(ax, ns)
        ax.set_xlim(ns[0] / 1.4, ns[-1] * 2.6)
        ax.legend(loc="upper left", fontsize=8)
    tag = "FEASIBILITY " if feas else ""
    fig.suptitle(f"E4  {tag}pKVA bytes written to FlashBlade by the cold request", x=0.01, ha="left",
                 fontweight="bold", color=TXT)
    fig.tight_layout()
    src = ("bytes_written / files_written of the cold request in e4_pkva_resume.jsonl, divided by history "
           "tokens, median over histories" if not feas else
           "Growth of the pKVA store directory during the cold request (feasibility kvstore records), divided by "
           "prompt tokens (history + 512-token query + template)")
    save(fig, out, name,
         src + ". The 20 KiB/token line is the attention-only KV "
         "of Qwen3.6-35B-A3B; anything above it is GDN/mamba state checkpoints. Right panel: same data per "
         "session with the 40 MiB MemT-Nano snapshot for reference.")


def fig_e4_memt_resume(res, out):
    name = "fig_e4_memt_resume"
    p = os.path.join(res, "e4_memt_resume.jsonl")
    recs = read_jsonl(p)
    if not recs:
        return not_available(name, "results/e4_memt_resume.jsonl not present yet (E4 agent running)")
    df = pd.json_normalize([r for r in recs if isinstance(r, dict)], max_level=0)
    ncol = next((c for c in ("N", "n_hist", "history_len", "length", "n") if c in df), None)
    scol = next((c for c in ("system", "kind", "path", "object", "label") if c in df), None)
    stages = []
    for want in ("read", "h2d", "prefill"):
        c = next((c for c in df.columns if c.lower().startswith(want) and c.endswith("_s")
                  and not isinstance(df[c].iloc[0], list)), None)
        if c:
            stages.append((want, c))
    if ncol is None or scol is None or not stages:
        return not_available(name, f"e4_memt_resume.jsonl schema not recognised (fields: {list(df.columns)[:30]})")
    if "status" in df:
        df = df[df.status.fillna("ok") == "ok"]
    if "label" in df and scol != "label":
        df = df[df["label"].astype(str).str.startswith("odirect")]
    if not len(df):
        return not_available(name, "no odirect samples in e4_memt_resume.jsonl yet")
    g = df.groupby([scol, ncol])[[c for _, c in stages]].median().reset_index()
    order = {"nano": 0, "memt": 0, "fullctx": 1}
    systems = sorted(dict.fromkeys(g[scol]), key=lambda x: order.get(str(x), 9))
    ns = sorted(g[ncol].unique())
    sysname = {"nano": "MemT-Nano snapshot", "fullctx": "HF full-context KV"}
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    width = 0.8 / max(1, len(systems))
    shades = {"read": 1.0, "h2d": 0.6, "prefill": 0.3}
    for i, s in enumerate(systems):
        base = C_MEM if "memt" in str(s).lower() or "nano" in str(s).lower() else C_KV
        h = g[g[scol] == s].set_index(ncol).reindex(ns)
        bottom = np.zeros(len(ns))
        x = np.arange(len(ns)) + (i - (len(systems) - 1) / 2) * width
        for st, c in stages:
            v = h[c].fillna(0).values
            ax.bar(x, v, width * 0.92, bottom=bottom, color=base, alpha=shades[st],
                   edgecolor=BG, linewidth=1, label=f"{sysname.get(s, s)}: {st}")
            bottom += v
    ax.set_xticks(np.arange(len(ns)))
    ax.set_xticklabels([fmt_tokens(n) for n in ns])
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("history length N (tokens)")
    ax.set_ylabel("seconds (median)")
    ax.set_title("E4  Resume stage breakdown: MemT snapshot vs HF KV restore")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    cnt = df.groupby([scol, ncol]).size()
    save(fig, out, name, "Resume of a saved session from FlashBlade (NFSv3/RDMA) on HF transformers, 2x A100: "
         "O_DIRECT read into a pinned host pool, host-to-device copy, then the 512-token query prefill (first "
         "token). Median over 'odirect*' samples ("
         + ", ".join(f"{s}@{fmt_tokens(n)}:{c}" for (s, n), c in cnt.items()) +
         f"); pagecache_warm samples excluded. Source e4_memt_resume.jsonl, fields {[c for _, c in stages]}.")


# ------------------------------------------------------------------ summary
def build_summary(args, ok, allc, files, val, e2, e3, replaced, e4, e4src, e4feas, bt, cs=None):
    L = []
    now = _dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    L.append("# MemT-1.0 reproduction: results summary\n")
    L.append(f"_Generated {now} by `analysis/make_figures.py` from `results/*.jsonl`. Do not edit by hand; "
             "rerun the script._\n")
    L.append("Paper reference values are from the MemT-1.0 paper (hardware unspecified). Our platform: DGX A100 "
             "(2x A100-80GB used, HF transformers BF16, batch 1), FlashBlade over NFS (RDMA and TCP).\n")

    # coverage
    L.append("## Coverage\n")
    if allc is not None:
        cov = allc.pivot_table(index="N", columns=["system", "status"], values="n", fill_value=0)
        cov.columns = [f"{s} {st}" for s, st in cov.columns]
        cov = cov.reset_index()
        cov["N"] = cov["N"].map(fmt_tokens)
        L.append(f"E1 sweep files: {', '.join(files)}. Run ids: "
                 f"{', '.join(sorted(ok['run_id'].unique()))}. Expected 35 histories per (N, system).\n")
        L.append(md_table(cov))
        miss = [fmt_tokens(n) for n in NS if n not in set(ok.N)]
        if miss:
            L.append(f"\nNo completed E1 records yet for N = {', '.join(miss)}.\n")
    else:
        L.append("No E1 sweep files found.\n")
    if val is not None:
        L.append(f"\nValidation/probe run ids: {', '.join(sorted(val['run_id'].unique()))}.\n")
    if e3 is not None:
        L.append(f"E3 run ids: {', '.join(sorted(e3['run_id'].unique()))}."
                 + (f" Cells taken from e3_recheck_rdma_small.jsonl: {'; '.join(replaced)}." if replaced else "")
                 + "\n")
    L.append(f"E4 source: {e4src or 'none'}" + (" (feasibility only)" if e4feas and e4src else "") + ".\n")

    # E1
    L.append("\n## E1  Efficiency (HF, paper protocol)\n")
    if ok is not None and len(ok):
        rows = []
        for (n, s), g in ok.groupby(["N", "system"]):
            rows.append({"N": fmt_tokens(n), "system": s, "n": len(g),
                         "e2e mean s": g.e2e_s.mean(), "e2e min s": g.e2e_s.min(), "e2e max s": g.e2e_s.max(),
                         "TTFT mean ms": g.ttft_s_mean.mean() * 1e3, "TTFT min ms": g.ttft_s_mean.min() * 1e3,
                         "TTFT max ms": g.ttft_s_mean.max() * 1e3, "prep mean s": g.prep_s.mean(),
                         "restore mean ms": g.restore_s_mean.mean() * 1e3,
                         "decode tok/s": g.decode_tok_s_mean.mean()})
        t = pd.DataFrame(rows)
        ff = {c: "{:.2f}" for c in t.columns}
        ff.update({c: "{:.0f}" for c in t.columns if "TTFT" in c})
        ff["restore mean ms"] = "{:.1f}"
        L.append(md_table(t, ff))
        L.append("\nSpeedup (full context / Nano), per-history paired ratio, mean [min-max]:\n")
        e = paired_ratio(ok, "e2e_s").set_index("N")
        tt = paired_ratio(ok, "ttft_s_mean").set_index("N")
        rows = []
        for n in NS:
            r = {"N": fmt_tokens(n)}
            if n in e.index:
                r["pairs"] = int(e.loc[n, "count"])
                r["e2e ours"] = f"{e.loc[n, 'mean']:.2f}x [{e.loc[n, 'min']:.2f}-{e.loc[n, 'max']:.2f}]"
            r["e2e paper"] = f"{P_SPEEDUP[n]:.2f}x" if n in P_SPEEDUP else ""
            if n in tt.index:
                r["TTFT ours"] = f"{tt.loc[n, 'mean']:.2f}x [{tt.loc[n, 'min']:.2f}-{tt.loc[n, 'max']:.2f}]"
            r["TTFT paper"] = f"{P_TTFT_RATIO[n]:.2f}x" if n in P_TTFT_RATIO else ""
            rows.append(r)
        L.append(md_table(pd.DataFrame(rows).astype(object).fillna("")))
        L.append("\nPaper absolutes: e2e full context 30.07 s (128K) / 94.47 s (256K), Nano 16.19 / 23.69 s, "
                 "Temp-LoRA 700.82 / 1400.88 s; TTFT Nano 104-112 ms, full context ~257 / ~445 ms. HF's MoE "
                 "expert loop puts a ~350 ms floor under our TTFT on A100.\n")
    else:
        L.append("_No E1 sweep data yet._\n")
    if val is not None and len(val):
        L.append("\n### Writer cost and 256K probe (validation files, 1-5 histories per point)\n")
        v = val.groupby(["series", "N"]).agg(n=("history_idx", "nunique"), prep_med_s=("prep_s", "median"),
                                             prep_min_s=("prep_s", "min"), prep_max_s=("prep_s", "max"),
                                             e2e_med_s=("e2e_s", "median"),
                                             ttft_med_ms=("ttft_s_mean", lambda x: x.median() * 1e3)).reset_index()
        v["N"] = v["N"].map(fmt_tokens)
        L.append(md_table(v, {"prep_med_s": "{:.2f}", "prep_min_s": "{:.2f}", "prep_max_s": "{:.2f}",
                              "e2e_med_s": "{:.2f}", "ttft_med_ms": "{:.0f}"}))
        L.append("\nValidation nano runs use 'messages' slice packing; the sweep uses 'dense'.\n")

    # E2
    L.append("\n## E2  Storage per trajectory (measured, safetensors)\n")
    if e2 is not None and len(e2):
        t = e2.copy()
        t["paper full ctx"] = t["N"].map(lambda n: fmt_bytes(paper_fc_bytes(n), 1))
        t["paper ratio"] = t["N"].map(lambda n: f"{paper_fc_bytes(n) / P_NANO_BYTES:.1f}x")
        t["ratio BF16"] = (t["fc_bf16_bytes"] / t["nano_bytes"]).map(lambda x: f"{x:.1f}x" if x == x else "")
        t["ratio FP32"] = (t["fc_fp32_bytes"] / t["nano_bytes"]).map(lambda x: f"{x:.1f}x" if x == x else "")
        for c in ("nano_bytes", "fc_bf16_bytes", "fc_fp32_bytes"):
            t[c] = t[c].map(lambda x: fmt_bytes(x, 2) if x == x else "")
        t["kv_bytes_per_token"] = t["kv_bytes_per_token"].map(lambda x: f"{x:.0f}" if x == x else "")
        t["N"] = t["N"].map(fmt_tokens)
        t = t.rename(columns={"nano_bytes": "Nano", "fc_bf16_bytes": "full ctx (BF16 GDN)",
                              "fc_fp32_bytes": "full ctx (FP32 GDN)", "kv_bytes_per_token": "KV B/token",
                              "fc_n": "fullctx records", "src": "source"})
        L.append(md_table(t[["N", "Nano", "full ctx (BF16 GDN)", "full ctx (FP32 GDN)", "ratio BF16",
                             "ratio FP32", "paper full ctx", "paper ratio", "KV B/token", "fullctx records",
                             "source"]].fillna("")))
        for col, lab in (("fc_bf16_bytes", "BF16 GDN"), ("fc_fp32_bytes", "FP32 GDN")):
            f = fit_fc(e2, col)
            if f:
                L.append(f"- Linear fit, full context {lab}: {f[0]:.0f} B/token + {fmt_bytes(f[1], 1)}; "
                         f"at 128K {fmt_bytes(f[0] * 131072 + f[1], 2)} "
                         f"({(f[0] * 131072 + f[1]) / P_NANO_BYTES:.1f}x Nano).\n")
        L.append("- FP32-GDN column = measured serialized bytes + (FP32 - BF16) recurrent-state bytes. HF stores "
                 "the GDN recurrent state in BF16; the paper counts FP32. Conv state stays BF16 in both "
                 "(HF conv kernel width 4, 1.875 MiB; E3's synthetic kv objects use width 3).\n")
    # E3
    L.append("\n## E3  FlashBlade persistence\n")
    if e3 is not None:
        k1 = e3[(e3.op == "read_cold") & (e3.concurrency == 1)].pivot_table(index="label", columns="target",
                                                                             values="median_s")
        k1 = k1.loc[sorted(k1.index, key=obj_order)] * 1e3
        sz = e3.groupby("label")["bytes"].first()
        k1.insert(0, "MiB", sz.reindex(k1.index) / MIB)
        k1 = k1.reset_index()
        k1["label"] = k1["label"].map(obj_name)
        L.append("### Cold restore, one session (O_DIRECT, median ms)\n")
        L.append(md_table(k1[["label", "MiB"] + [t for t in ("rdma", "tcp", "local") if t in k1]],
                          {c: "{:.1f}" for c in ("MiB", "rdma", "tcp", "local")}))
        c = e3[e3.op == "conc_read_cold"]
        rows = []
        for lab in sorted(c.label.unique(), key=obj_order):
            for t in ("rdma", "tcp", "local"):
                r = {"object": obj_name(lab), "target": t}
                for k in (8, 32):
                    x = c[(c.label == lab) & (c.target == t) & (c.concurrency == k)]
                    if len(x):
                        x = x.iloc[0]
                        r[f"K={k} GB/s"] = x["agg_gbps_median"]
                        r[f"K={k} p50 ms"] = x["lat_p50_s"] * 1e3
                        r[f"K={k} p95 ms"] = x["lat_p95_s"] * 1e3
                        if x["_recheck"]:
                            r["object"] = obj_name(lab) + " (recheck)"
                rows.append(r)
        L.append("\n### Concurrent cold restore (distinct files, one process each)\n")
        ct = pd.DataFrame(rows)
        L.append(md_table(ct, {col: ("{:.2f}" if "GB/s" in col else "{:.0f}") for col in ct.columns}))
        # derived
        rows = []
        for lab in sorted(c.label.unique(), key=obj_order):
            b = sz[lab]
            r = {"object": obj_name(lab), "bytes/session": fmt_bytes(b, 1), "sessions per TB": 1e12 / b}
            for t in ("rdma", "tcp"):
                x1 = e3[(e3.op == "read_cold") & (e3.label == lab) & (e3.target == t) & (e3.concurrency == 1)]
                x32 = c[(c.label == lab) & (c.target == t) & (c.concurrency == 32)]
                if len(x1):
                    r[f"{t} K=1 restore ms"] = x1["median_s"].iloc[0] * 1e3
                if len(x32):
                    g = x32["agg_gbps_median"].iloc[0]
                    r[f"{t} K=32 ms/session (amortised)"] = b / (g * 1e9) * 1e3
                    r[f"{t} K=32 sessions/s"] = g * 1e9 / b
            rows.append(r)
        L.append("\n### Derived: capacity and restore cost per session\n")
        dt = pd.DataFrame(rows)
        L.append(md_table(dt, {col: "{:,.0f}" if "per TB" in col else "{:.1f}" for col in dt.columns}))
        L.append("\nsessions per TB = 1e12 / bytes; amortised ms/session = bytes / aggregate K=32 throughput.\n")
        # first buffered read
        f = e3[(e3.op.isin(["read_cold", "read_buffered_first", "read_cached"])) & (e3.concurrency == 1)
               & (e3.label.isin(["memt", "kv@32768", "kv@262144"]))]
        p = f.pivot_table(index=["label", "target"], columns="op", values="gbps_median").reset_index()
        p["label"] = p["label"].map(obj_name)
        p = p.rename(columns={"read_cold": "O_DIRECT GB/s", "read_buffered_first": "1st buffered GB/s",
                              "read_cached": "2nd buffered (cached) GB/s"})
        L.append("\n### First buffered read after write (NFS close-to-open)\n")
        L.append(md_table(p, {c: "{:.2f}" for c in p.columns}))
        L.append("\nOn NFS the first buffered read after write+close revalidates and refetches through the "
                 "page cache at ~0.55 GB/s, 10x slower than O_DIRECT; on local ext4 it is a page-cache hit. "
                 "tcp kv@2K's 1st buffered read (3.78 GB/s) is an exception where the client kept its cache.\n")
    else:
        L.append("_No E3 data._\n")
    if cs is not None and len(cs):
        L.append("\n### Small-object concurrency sweep (cold O_DIRECT, idle path)\n")
        L.append(f"Source e3_conc_sweep_small.jsonl (run {', '.join(sorted(cs['run_id'].dropna().unique()))}, "
                 f"{int(cs['reps'].median())} reps); supersedes the recheck's K=8/32 small-object cells for the "
                 "RDMA-dip question. GB/s = aggregate (1e9 B/s).\n")
        L.append(md_table(conc_sweep_table(cs), {"p50 ms": "{:.1f}", "p95 ms": "{:.1f}",
                                                 "per-session GB/s": "{:.2f}"}))

    # E4
    L.append("\n## E4  vLLM + pKVA production path\n")
    if e4 is not None and not e4feas:
        L.append(f"Source {e4src} (vLLM TP2 + pKVA config B, KV store on FlashBlade RDMA).\n")
        rows = []
        for n, g in e4.groupby("N"):
            r = {"N": fmt_tokens(n)}
            for lab in ("cold", "hot", "restored"):
                h = g[g.label == lab]["ttft_s"] * 1e3
                r[f"{lab} TTFT ms"] = f"{h.median():.0f} [{h.min():.0f}-{h.max():.0f}] n={len(h)}" if len(h) else ""
            if bt is not None and n in set(bt["n"]):
                b = bt[bt.n == n].iloc[0]
                r["KiB/token"] = f"{b['bytes_per_token'] / KIB:.1f}"
                r["bytes/history"] = fmt_bytes(b["delta"], 2)
                r["files/history"] = f"{b['files']:.0f}"
            rows.append(r)
        L.append("pKVA TTFT by N, median [min-max] n (restored: valid samples only); bytes written by the cold "
                 "request, median per history (KiB/token = bytes_written / history_tokens):\n")
        L.append(md_table(pd.DataFrame(rows).astype(object).fillna("")))
        for nt in E4_NOTES:
            L.append(f"- {nt}\n")
    if e4 is not None and e4feas:
        if e4feas:
            L.append("**Feasibility data only** (1 history, 1 sample per condition; e4_pkva_resume.jsonl "
                     "not yet available).\n")
        t = e4.pivot_table(index=["config", "n_hist"], columns="label", values="ttft_s", aggfunc="median")
        t = (t * 1e3).reset_index()
        t["n_hist"] = t["n_hist"].map(fmt_tokens)
        L.append("TTFT, ms (median per cell):\n")
        L.append(md_table(t, {c: "{:.0f}" for c in t.columns}))
        if bt is not None:
            b = bt.copy()
            b["n"] = b["n"].map(fmt_tokens)
            b["delta"] = b["delta"].map(lambda x: fmt_bytes(x, 2))
            b["KiB/token"] = b["bytes_per_token"] / KIB
            L.append("\nBytes written to the pKVA store by the cold request:\n")
            L.append(md_table(b[["config", "n", "prompt_tokens", "delta", "files", "KiB/token"]],
                              {"KiB/token": "{:.1f}"}))
    elif e4 is None:
        L.append("_No E4 data._\n")
    mr = read_jsonl(os.path.join(args.results, "e4_memt_resume.jsonl"))
    if not mr:
        L.append("\ne4_memt_resume.jsonl: not yet available.\n")
    else:
        L.append("\n### MemT snapshot vs HF KV resume from FlashBlade (e4_memt_resume.jsonl)\n")
        m = pd.DataFrame(mr)
        need = {"N", "system", "label"}
        if not need <= set(m.columns):
            L.append(f"_schema not recognised: {list(m.columns)[:30]}_\n")
        else:
            save_b = m[m.label == "save"].groupby(["N", "system"])["file_bytes"].median() \
                if "file_bytes" in m else pd.Series(dtype=float)
            od = m[m.label.astype(str).str.startswith("odirect")]
            rows = []
            for (n, s), g in od.groupby(["N", "system"]):
                r = {"N": fmt_tokens(n), "system": s, "n": len(g),
                     "histories": g["history_idx"].nunique() if "history_idx" in g else ""}
                for c, lab in (("read_s", "read ms"), ("h2d_s", "h2d ms"), ("prefill_s", "prefill ms"),
                               ("ttft_s", "TTFT ms")):
                    r[lab] = g[c].median() * 1e3 if c in g else np.nan
                r["total s"] = g["total_s"].median() if "total_s" in g else np.nan
                fb = save_b.get((n, s), np.nan)
                if fb != fb and "bytes" in g:
                    fb = g["bytes"].median()
                r["file_bytes"] = fmt_bytes(fb, 2) if fb == fb else ""
                rows.append(r)
            t = pd.DataFrame(rows)
            if len(t):
                t["_o"] = t["system"].map({"nano": 0, "fullctx": 1})
                t = t.sort_values(["_o"], kind="stable").drop(columns="_o")
                t = t.sort_values("N", key=lambda c: c.map(lambda x: dict((fmt_tokens(v), i) for i, v in
                                                                          enumerate(NS)).get(x, 99)), kind="stable")
            L.append("Medians over O_DIRECT ('odirect*') samples; TTFT = restore (read + h2d + pool) + query "
                     "prefill; total = TTFT + 128-token decode.\n")
            L.append(md_table(t, {"read ms": "{:.1f}", "h2d ms": "{:.1f}", "prefill ms": "{:.0f}",
                                  "TTFT ms": "{:.0f}", "total s": "{:.2f}"}))
            pw = m[m.label.astype(str).str.startswith("pagecache_warm")]
            if len(pw) and "read_s" in pw:
                L.append(f"\npagecache_warm samples (excluded above): median read {pw['read_s'].median() * 1e3:.1f} ms "
                         f"(n={len(pw)}).\n")
            done = od.groupby(["system"])["N"].max().to_dict()
            L.append(f"Largest N with odirect samples so far: {done}. Run ids: "
                     f"{', '.join(sorted(m['run_id'].dropna().unique())) if 'run_id' in m else '?'}.\n")

    # captions + quality
    L.append("\n## Figures and captions\n")
    for k, v in CAPTIONS.items():
        L.append(f"- **{k}**: {v}\n")
    L.append("\n## Data-quality notes (auto-detected + known)\n")
    for q in QUALITY + WARNINGS:
        L.append(f"- {q}\n")
    return "\n".join(L)


# ------------------------------------------------------------------ main
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join(root, "results"))
    ap.add_argument("--out", default=os.path.join(root, "figures"))
    ap.add_argument("--summary", default=os.path.join(root, "results", "summary.md"))
    args = ap.parse_args()
    setup_style()
    res = args.results

    ok, allc, files = load_e1(res)
    e1_quality(ok)
    if allc is not None:
        print("[E1 coverage] ok records per (N, system):")
        print(ok.groupby(["N", "system"]).size().unstack(fill_value=0).to_string())
    val = load_validation(res)
    if val is not None:
        o = val[(val.system == "fullctx") & (val.prep_s > 3 * val.groupby(["system", "N"])["prep_s"]
                                             .transform("median"))]
        for _, r in o.iterrows():
            quality(f"e1_validate: fullctx prep_s outlier history {r.history_idx} @N={fmt_tokens(r.N)}: "
                    f"{r.prep_s:.2f} s (run {r.run_id}; likely first-run warm-up)")
    e3, replaced = load_e3(res)
    if e3 is not None:
        c = e3[e3.op == "conc_read_cold"]
        for (t, lab), g in c.groupby(["target", "label"]):
            g = g.set_index("concurrency")
            if 8 in g.index and 1 in g.index and g.loc[8, "agg_gbps_median"] < 0.8 * g.loc[1, "agg_gbps_median"] * 1:
                quality(f"E3: {t} {lab} aggregate restore at K=8 ({g.loc[8, 'agg_gbps_median']:.1f} GB/s) is below "
                        f"K=1 ({g.loc[1, 'agg_gbps_median']:.1f} GB/s)")
            if 32 in g.index and 8 in g.index and g.loc[32, "agg_gbps_median"] < 0.6 * g.loc[8, "agg_gbps_median"]:
                quality(f"E3: {t} {lab} aggregate restore drops from {g.loc[8, 'agg_gbps_median']:.1f} GB/s (K=8) to "
                        f"{g.loc[32, 'agg_gbps_median']:.1f} GB/s (K=32), p95 {g.loc[32, 'lat_p95_s'] * 1e3:.0f} ms"
                        + (" [recheck]" if bool(g.loc[32, "_recheck"]) else ""))
        # cells far below the same target/K on comparable objects (kv >= 8K)
        big = c[c.label.str.startswith("kv@") & (c.bytes >= 200 * MIB)]
        ref = big.groupby(["target", "concurrency"])["agg_gbps_median"].median()
        for _, x in big.iterrows():
            m = ref.get((x.target, x.concurrency))
            if m and x.concurrency > 1 and x.agg_gbps_median < 0.6 * m:
                quality(f"E3: {x.target} {x.label} K={x.concurrency} aggregate restore {x.agg_gbps_median:.1f} GB/s "
                        f"vs {m:.1f} GB/s median for kv objects >=8K at the same K/target (p50 "
                        f"{x.lat_p50_s * 1e3:.0f} ms); single-rep outlier, worth a recheck")
        quality("E3: 'st_load_cold' is O_DIRECT read + bytes() copy + safetensors.numpy.load, i.e. the loader's "
                "CPU cost on top of a cold read; it is ~0.67 GB/s on every target, so it measures the CPU path, "
                "not storage")
        quality("E3: kv@N objects are synthetic (random BF16-entropy data) with FP32 GDN recurrent state and a "
                "width-3 conv state, ~0.5 MB smaller than the HF-measured FP32-GDN accounting")
    e2 = e2_table(ok, val)
    cs = load_conc_sweep(res)
    e4, stores, e4src, e4feas = load_e4_ttft(res)
    bt = e4_bytes(e4, stores) if e4 is not None else None
    if e4feas and e4 is not None:
        quality("E4 feasibility: config B file's 'restored' rows (tag h1) re-read objects written under config A; "
                "config A's 8K 'restored' TTFT (1.98 s) exceeds its cold TTFT (0.91 s): first request after restart")
        quality("E4 feasibility: config A cold TTFT is ~2x config B's at 128K (16.7 vs 8.6 s): per-chunk "
                "checkpointing is on the prefill critical path")

    jobs = [
        lambda: fig_e1_latency(ok, args.out),
        lambda: fig_e1_ttft(ok, args.out),
        lambda: fig_e1_speedup(ok, args.out),
        lambda: fig_e1_writer_cost(val, args.out),
        lambda: fig_e2_storage(e2, args.out),
        lambda: fig_e3_restore_latency(e3, args.out),
        lambda: fig_e3_concurrency(e3, args.out),
        lambda: fig_e3_loader_overhead(e3, args.out),
        lambda: fig_e3_conc_sweep(cs, args.out),
        lambda: fig_e4_pkva_ttft(e4, e4src, e4feas, args.out),
        lambda: fig_e4_bytes(bt, e4feas, args.out),
        lambda: fig_e4_memt_resume(res, args.out),
    ]
    if not os.path.exists(os.path.join(res, "e4_pkva_resume.jsonl")):
        print("[info] e4_pkva_resume.jsonl: not yet available; E4 TTFT/bytes figures use the feasibility files")
    failures = 0
    for j in jobs:
        try:
            j()
        except Exception:  # keep going; one broken figure must not block the rest
            failures += 1
            traceback.print_exc()
            plt.close("all")
    md = build_summary(args, ok, allc, files, val, e2, e3, replaced, e4, e4src, e4feas, bt, cs)
    os.makedirs(os.path.dirname(os.path.abspath(args.summary)), exist_ok=True)
    with open(args.summary, "w") as f:
        f.write(md)
    print(f"[summary] {args.summary}")
    if QUALITY:
        print("[data quality]")
        for q in QUALITY:
            print("  - " + q)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
