# Analysis and figures

One command regenerates every figure in `figures/` and the tables in `results/summary.md`
from the JSONL files in `results/`. It only reads `results/*.jsonl`; it never writes to them.

```bash
cd $M1T_ROOT   # repo root
.venv-analysis/bin/python analysis/make_figures.py --out figures/ --summary results/summary.md
```

Rerun it whenever new results land (E1 sweep finished, `e4_pkva_resume.jsonl`, `e4_memt_resume.jsonl`).
The script tolerates missing or partial inputs: half-written trailing lines of live JSONL files are
skipped, and a figure whose inputs are absent prints `not yet available` instead of failing.
The exit code is non-zero only if a figure raised an exception (the traceback is printed and the
other figures are still produced).

## Environment

A separate CPU-only venv (do not use `.venv`, which the GPU sweep uses):

```bash
UV=uv   # or the full path to your uv binary
$UV venv --python /usr/bin/python3.12 .venv-analysis
$UV pip install --python .venv-analysis/bin/python -r env/requirements-analysis.txt
```

## Inputs -> outputs

| input | used for |
|---|---|
| `e1_sweep_shard*.jsonl` (merged, `status == ok`, de-duplicated) | fig_e1_latency, fig_e1_ttft, fig_e1_speedup, E1 tables, coverage |
| `e1_validate.jsonl`, `e1_probe256k.jsonl` (1-5 histories per point) | fig_e1_writer_cost, writer table, E2 bytes at 32K/256K |
| E1 + validation `state.*` fields | fig_e2_storage, E2 table |
| `e3_persist_io.jsonl` (+ `e3_recheck_rdma_small.jsonl` overrides matching cells) | fig_e3_restore_latency, fig_e3_concurrency, fig_e3_concurrency_latency, fig_e3_loader_overhead, E3 tables |
| `e4_pkva_resume.jsonl` if present, else `e4_feasibility_config{A,B}.jsonl` | fig_e4_pkva_ttft, fig_e4_bytes_per_token, E4 tables |
| `e4_memt_resume.jsonl` (when present) | fig_e4_memt_resume |

Every figure is written as PNG (200 dpi, 2x) and SVG. Captions are printed to stdout and collected
in `results/summary.md`, together with auto-detected data-quality notes.

`e4_pkva_resume.jsonl` (when present) supersedes the feasibility files: x = `N`, series = `label` in
{cold, hot, restored} with restored limited to `valid == True`; filler_*, first_load_after_kva_start and
end records are excluded from plots and counted in a note. Bytes come from `bytes_written` /
`files_written` on cold records (B/token = bytes_written / history_tokens, median over histories).

`e4_memt_resume.jsonl` uses only `odirect*` samples (pagecache_warm reported separately; save records
give file_bytes) and is parsed loosely: N from one of `N / n_hist / history_len / length`, the
system from `system / kind / path / object / label`, and stages from the first `read*_s`, `h2d*_s`,
`prefill*_s` scalar fields. If the schema differs, the figure is skipped with the field list printed;
adjust `fig_e4_memt_resume` accordingly.

## Conventions

- Colors: full context / KV `#2a78d6`, MemT-Nano `#eb6834`, Temp-LoRA `#1baf7a`; E3 targets
  rdma `#6250d6`, tcp `#e34948`, local `#eda100` (validated all-pairs, with direct labels).
- Paper values are hollow markers with dashed lines in the same hue, labelled
  "paper (hardware unspecified)".
- Speedups are per-history paired ratios (full context / Nano) averaged across histories, with a
  min-max band. A crossover is marked only when the ratio clears parity by >= 5% and stays above it.
- E2 "FP32 GDN" = measured serialized bytes + the extra bytes the GDN recurrent state would take in
  FP32 (HF stores it in BF16; the paper counts FP32).
