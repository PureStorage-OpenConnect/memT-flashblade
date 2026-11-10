# MemT-1.0 on a DGX A100 with FlashBlade: an independent reproduction

An independent reproduction of the efficiency and storage claims in *MemT-1.0: Towards Memory-Native
Evolution* ([arXiv:2610.26825](https://arxiv.org/abs/2610.26825), MemTensor team, October 2026),
run on a DGX A100 with session state persisted to a Pure Storage FlashBlade over NFS. The paper
measures compute and bytes. This repository measures the same things on named hardware, then adds
what the paper leaves out: what it costs to put each kind of session state on shared storage and
bring it back.

Everything here was produced in one day (2026-11-09) by a Claude Code session orchestrating the
experiments; the person running it reviewed the plan, the results, and this text.

- Blog post: `docs/blog.md`
- Plain-language explainer with architecture diagrams: `docs/explainer.html`
- Experiment plan and scope: `docs/PLAN.md`
- All numbers, auto-generated from the raw results: `results/summary.md`

## What the paper claims and what we found

The model under test is **MemT-1.0-Nano**: Qwen3.6-35B-A3B (a 36B-parameter mixture-of-experts
model with 3B active parameters, 30 Gated DeltaNet layers and 10 full-attention layers) plus a
trained memory block on the 10 attention layers. Memory is a fixed 40 MiB FP32 state per session
(10 layers x 16 heads x 256 x 256). The paper compares it with the same backbone reading the full
conversation history as context.

| Claim | Paper (hardware unspecified) | This repo (2x A100-80GB, HF transformers, BF16, batch 1) |
|---|---|---|
| C1 end-to-end latency (history prep + 128-token reply), Nano vs full context | parity at 32K; 1.20x at 64K; 1.86x at 128K; 3.99x at 256K | 0.83x at 32K; 0.95x at 64K; 1.41x at 128K; 2.68x at 256K (35 histories, paired means) |
| C2 query TTFT | Nano flat at 104-112 ms; full context 257 ms at 128K, 445 ms at 256K | Nano flat at 366 to 370 ms; full context 767 ms at 128K, 1,253 ms at 256K |
| C3 state per session | Nano 40 MiB constant; full context ~20 KiB/token (5.06 GiB at 256K) | 40.00 MiB constant; 20,480 B/token attention KV + 32 MiB GDN state (4.99 GiB at 256K) |

Beyond the paper, with the state on FlashBlade:

| Measurement | Result |
|---|---|
| Cold restore of one 40 MiB MemT snapshot (O_DIRECT, NFS over RDMA) | 8 ms; one 256K full-context KV state (5.2 GB) 849 ms |
| Aggregate cold-read throughput, 32 concurrent sessions | ~23 GB/s RDMA, ~18 GB/s TCP for large states |
| MemT resume end to end (read + copy to GPU + query prefill to first token) | 375 ms, of which storage is 8 ms |
| vLLM + pKVA full-context resume from FlashBlade at 256K | 1.54 s TTFT vs 26.9 s cold recompute, 23.9 KiB/token stored |
| Loader overhead | a generic safetensors load is ~8x slower than a raw O_DIRECT read for the 40 MiB object |

Two differences from the paper matter when reading the E1 numbers. First, the checkpoint's shipped
memory-write path is a per-token Python loop; `bench/memt_fast_writer.py` replaces it with the
equivalent chunked GDN-2 kernel from `flash-linear-attention` (4-6x faster to write a history),
and the sweep uses that writer. With the shipped loop, Nano is slower than full context at every
length on this hardware. Second, HF's MoE expert dispatch puts a ~350 ms floor under every query on
A100, so our TTFT curves sit higher than the paper's and the TTFT ratio is compressed. Curve shapes
and the storage numbers reproduce; absolute latencies do not, and the paper gives no hardware to
compare against.

## Repository layout

```
bench/                  HF-side experiments (MemT-1.0-Nano and the backbone as full context)
  e1_efficiency_sweep.py   E1/E2: paper protocol (Sec 6.5/6.6), 35 BEAM histories x 7 lengths
  memt_fast_writer.py      chunked-kernel memory write path (patches MemTMemory at run time)
  test_fast_writer.py      numerical check of the fast writer against the shipped loop
  e3_persist_io.py         E3: save/restore of state objects on FlashBlade RDMA, TCP, local NVMe
  e4_memt_resume.py        E4 part 2: resume a MemT session (and an HF KV cache) from FlashBlade
  e4_build_ids.py          shared token ids for E4 so both stacks see identical prompts
  e4_summarize.py          markdown tables for E4
  beam_histories.py        BEAM 1M-split loader and token-length slicing
  smoke_memt_hf.py, diag_decode_recall.py   smoke test and recall diagnostics
vllm_kva/               full-context baseline on vLLM 0.32 + Pure Storage KV Accelerator 1.3.0
  start.sh, stop.sh        launch recipe (flags that make the hybrid GDN/attention model work)
  kva-memt-a*.json         kva_server configs (FlashBlade-backed store, host pool sizes)
  e4_pkva_resume.py        E4 part 1: cold / hot / restored-from-FlashBlade TTFT
  bench_ttft.py, run_conditions.sh   feasibility measurements
analysis/make_figures.py  regenerates figures/ and results/summary.md from results/*.jsonl
results/                raw JSONL, one record per measurement, plus the generated summary
figures/                PNG and SVG figures
docs/                   plan, blog post, explainer page
env/                    frozen requirements and environment notes
```

## Platform

- DGX A100: 8x A100-SXM4-80GB (two used per experiment), 2x AMD EPYC 7742, 2 TB RAM, Ubuntu 24.04,
  driver 595.112, CUDA 13.2.
- FlashBlade over NFS, 200 GbE: NFSv3 over RDMA (`nconnect=16`, 512 KiB rsize/wsize) and
  NFSv4.1 over TCP (`nconnect=16`). Local NVMe RAID (ext4) as a reference target.
- HF side: torch 2.13.0+cu130, transformers 5.12.1 (pinned by the checkpoint), fla 0.6.0.
- vLLM side: vLLM 0.32.0, purekva-plugin 1.3.0, kva_server 1.3.0.

Paths in the scripts default to the layout of the machine this ran on (`M1T_ROOT`, `HF_HOME`, the
mount points in `bench/e3_persist_io.py --targets`); override them with the environment variables
and flags documented in each script's header.

## Reproducing

### Environment

```bash
export HF_HOME=/path/to/hf-cache HF_HUB_OFFLINE=1   # after a one-time download of the two models and BEAM
uv venv --python python3.12 .venv
uv pip install --python .venv/bin/python -r env/requirements-hf.txt
uv venv --python python3.12 .venv-analysis
uv pip install --python .venv-analysis/bin/python -r env/requirements-analysis.txt
```

`env/README.md` explains the version pins (transformers 5.12.1 is required by `modeling_memt.py`)
and the MoE expert-dispatch settings that matter for A100.

### E1 and E2: paper protocol

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python bench/e1_efficiency_sweep.py \
    --histories 0-34 --lengths 512,2048,8192,32768,65536,131072,262144 \
    --systems nano,fullctx --reps 5 --writer fast --out results/e1_sweep.jsonl
```

One record per (history, length, system) with prep time, per-query TTFT and end-to-end time,
decode rate, and the serialized size of the per-session state (E2). `--writer reference` uses the
checkpoint's shipped per-token loop. The run took about 7 hours per 18 histories on two A100s;
we ran two shards on GPUs 0-1 and 2-3. Validation on two histories first:
`--histories 0,1 --lengths 512,32768,262144 --reps 2`.

### E3: storage persistence (no GPU)

```bash
.venv/bin/python bench/e3_persist_io.py --targets rdma,tcp,local --objects memt,kv,lora \
    --lengths 512,2048,8192,32768,65536,131072,262144 --concurrency 1,8,32 --reps 5 \
    --phases raw,st,conc --out results/e3_persist_io.jsonl
```

Materializes byte-exact safetensors images of each state object, then times buffered and
O_DIRECT writes and reads, the safetensors library load path, and K concurrent cold reads of
distinct files. Mount points are passed as `--targets name=path`.

### E4: resume from FlashBlade

Part 1, vLLM + pKVA (see `vllm_kva/start.sh` for the flags; `--mamba-cache-mode align` and
`--enable-prefix-caching` are what let the hybrid model's KV be offloaded):

```bash
cd vllm_kva && KVA_CONFIG=$PWD/kva-memt-a-e4.json ./start.sh all   # ~17 min to ready
python e4_pkva_resume.py --lengths 2048,8192,32768,65536,131072,262144 --histories 15,16,17 \
    --out ../results/e4_pkva_resume.jsonl
```

Part 2, HF MemT snapshot and HF KV-cache resume via O_DIRECT, staged timings:

```bash
CUDA_VISIBLE_DEVICES=4,5 .venv/bin/python bench/e4_memt_resume.py --systems nano,fullctx
```

### Figures and summary

```bash
.venv-analysis/bin/python analysis/make_figures.py --out figures/ --summary results/summary.md
```

## Scope

Reproduced: Section 6.5 (efficiency) and 6.6 (storage) for MemT-1.0-Nano. Not reproduced: the
memory-quality benchmarks (Table 2, needs a GPT-4.1-mini judge), the training-side MoS
microbenchmark (Table 1, no code released), the agentic-coding study (Section 7), and
MemT-1.0-Flash (576 GB of weights). See `docs/PLAN.md`.

## License

Code: Apache-2.0 (`LICENSE`). Model weights, datasets and the authors' serving code are not
included; see `THIRD_PARTY_NOTICES.md` for their licenses and for what was and was not used.
