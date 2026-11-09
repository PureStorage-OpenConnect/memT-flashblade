# Reproduction plan: MemT-1.0 on a DGX A100 with FlashBlade

Paper: *MemT-1.0: Towards Memory-Native Evolution* (MemTensor team, arXiv:2610.26825, Oct 2026).
Code: https://github.com/MemTensor/MemT-1.0 · Weights: https://huggingface.co/MemTensor/MemT-1.0-Nano

## 1. Claims under test

| ID | Paper claim (MemT-1.0-Nano vs full-context Qwen3.6-35B-A3B) | Where |
|----|---------------------------------------------------------------|-------|
| C1 | End-to-end latency (history prep + 128-token reply): parity at 32K history, 1.20x faster at 64K, 1.86x at 128K, 3.99x at 256K | Sec 6.5, Fig 8 left, Fig 3 |
| C2 | Query TTFT flat at ~104-112 ms for Nano; full context rises to ~257 ms (128K) and ~445 ms (256K), i.e. 2.44x / 4.19x | Sec 6.5, Fig 8 right |
| C3 | Per-trajectory persistent state: Nano constant 40 MiB (10,485,760 FP32); full context grows ~20 KiB/token (5.06 GiB at 256K), ~66x smaller at 128K | Sec 6.6, Fig 9, Fig 3, Table 6 |

Out of scope and why:
- Table 2 memory-quality benchmarks: need a GPT-4.1-mini judge and 11 benchmark harnesses. No OpenAI key on this host.
- Table 1 / Appendix F MoS windowed-execution microbenchmark: training-side code, not released.
- Section 7 agentic coding: SWE-bench at 3 runs per task; cost far beyond a reproduction.
- MemT-1.0-Flash: 576 GB BF16, does not fit on 8x80 GB.

## 2. Platform

- DGX A100: 8x A100-SXM4-80GB (NVLink), 2x AMD EPYC 7742, 2 TB RAM, Ubuntu 24.04, driver 595.112, CUDA 13.2.
- FlashBlade over NFS: `/mnt/data` (NFSv3, RDMA, nconnect=16) and `/mnt/data_tcp` (NFSv4.1, TCP, nconnect=16), 200 GbE. O_DIRECT works on both mounts.
- Local reference: NVMe RAID (ext4) at `/`.
- The paper reports no GPU model, interconnect, or software versions. Absolute numbers are therefore not expected to match; ratios and curve shapes are.

## 3. Software tracks

| Track | Stack | Role |
|-------|-------|------|
| HF | PyTorch + transformers, checkpoint `modeling_memt.py` (session API: `commit_memory`, `save_state`), 2 GPUs, batch 1 | Paper-faithful reproduction of C1-C3 (the paper used HF inference for these) |
| vLLM + pKVA | vLLM 0.32, `PureKVConnector_V1` 1.3.0, KV store on FlashBlade | Production full-context baseline: KV cache persisted to and restored from FlashBlade |
| SGLang (optional POC) | authors' `memt_sglang` 0.5.16 server with `--state-directory` on FlashBlade | Timing of memory-snapshot export/import through the authors' serving path |

## 4. Experiments

### E1  Efficiency sweep (paper-faithful)
Protocol from Sec 6.5: 35 histories, lengths {512, 2K, 8K, 32K, 64K, 128K, 256K} tokens, fixed 512-token query plus chat template, forced 128-token decode (ignore EOS), BF16, 2 GPUs, concurrency 1. Prepare history once, one warm-up query, average 5 timed queries. Nano writes history in 2048-token slices; full context prefills history into the KV cache and reuses it. Both include state restoration in TTFT.
Outputs: `results/e1_*.jsonl`, `figures/fig8_repro_*.png` (latency and TTFT vs history length, ours vs paper).

### E2  Storage (paper-faithful, measured not computed)
Serialize each system's per-trajectory state after history prep and before the query; record actual bytes on disk (safetensors) per history length. Verify 40 MiB constant vs ~20 KiB/token + constant GDN state.
Outputs: `results/e2_storage.jsonl`, `figures/fig9_repro.png`.

### E3  FlashBlade persistence (new work, beyond the paper)
Save and restore the serialized states from E2 to three targets: FlashBlade RDMA, FlashBlade TCP, local NVMe. Cold reads via O_DIRECT. Concurrency 1, 8, 32 sessions. Then end-to-end *resume* latency on the GPU: restore state from FlashBlade, inject, run the query, measure TTFT, for Nano vs full context at each history length. Derived: sessions per TB of FlashBlade, aggregate restore bandwidth needed per resumed-sessions/sec.
Outputs: `results/e3_*.jsonl`, `figures/e3_resume_*.png`.

### E4  Production full-context path (vLLM + pKVA)
Qwen3.6-35B-A3B on vLLM TP2 with pKVA offloading KV to FlashBlade. TTFT at each history length for cold (compute), hot (GPU prefix hit), and restored-from-FlashBlade (GPU cache cleared, kva_server retained). Bytes on FlashBlade per session. Compared with HF-side Nano snapshot restore from E3.
Outputs: `results/e4_*.jsonl`, `figures/e4_*.png`.

### E5  SGLang snapshot POC (optional)
Authors' server with `--state-directory` on FlashBlade; time `export`/`import` of `memt-memory-v1` snapshots; confirm HF-saved snapshots load.

## 5. Deliverables
- Public GitHub repo: scripts, launch recipes, raw results, figures, this plan.
- Blog post: what transfers, what does not, and what it means for a storage tier that holds session state.
