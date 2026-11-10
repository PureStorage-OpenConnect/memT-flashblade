# MemT-1.0 reproduction: results summary

_Generated 2026-11-10 00:36 PST by `analysis/make_figures.py` from `results/*.jsonl`. Do not edit by hand; rerun the script._

Paper reference values are from the MemT-1.0 paper (hardware unspecified). Our platform: DGX A100 (2x A100-80GB used, HF transformers BF16, batch 1), FlashBlade over NFS (RDMA and TCP).

## Coverage

E1 sweep files: e1_sweep_shard0.jsonl, e1_sweep_shard1.jsonl. Run ids: 20261109T172910-55f776, 20261109T172910-c03489. Expected 35 histories per (N, system).

| N | fullctx ok | nano ok |
|---|---:|---:|
| 512 | 35 | 35 |
| 2K | 35 | 35 |
| 8K | 35 | 35 |
| 32K | 35 | 35 |
| 64K | 35 | 35 |
| 128K | 35 | 35 |
| 256K | 35 | 35 |


Validation/probe run ids: 20261109T163407-aad11a, 20261109T164733-97e5b8, 20261109T165403-b6ae03, 20261109T170754-d2ae33, 20261109T171202-d32fa6, 20261109T171824-e4d8b7.

E3 run ids: e3-full-20261109-1748, e3-recheck-1859. Cells taken from e3_recheck_rdma_small.jsonl: rdma kv@512 conc_read_cold K=1; rdma kv@512 conc_read_cold K=32; rdma kv@512 conc_read_cold K=8; rdma kv@512 conc_write K=1; rdma kv@512 conc_write K=32; rdma kv@512 conc_write K=8; rdma memt conc_read_cold K=1; rdma memt conc_read_cold K=32; rdma memt conc_read_cold K=8; rdma memt conc_write K=1; rdma memt conc_write K=32; rdma memt conc_write K=8.

E4 source: e4_pkva_resume.jsonl.


## E1  Efficiency (HF, paper protocol)

| N | system | n | e2e mean s | e2e min s | e2e max s | TTFT mean ms | TTFT min ms | TTFT max ms | prep mean s | restore mean ms | decode tok/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | fullctx | 35 | 12.01 | 11.79 | 12.42 | 362 | 353 | 378 | 0.39 | 4.1 | 11.28 |
| 512 | nano | 35 | 12.97 | 12.65 | 13.55 | 370 | 357 | 380 | 0.42 | 2.1 | 10.43 |
| 2K | fullctx | 35 | 11.93 | 11.75 | 12.09 | 360 | 353 | 366 | 0.45 | 5.0 | 11.42 |
| 2K | nano | 35 | 13.23 | 13.00 | 13.71 | 367 | 357 | 381 | 0.74 | 2.0 | 10.48 |
| 8K | fullctx | 35 | 12.62 | 12.38 | 13.03 | 377 | 369 | 388 | 1.08 | 9.4 | 11.38 |
| 8K | nano | 35 | 14.61 | 14.32 | 15.03 | 367 | 359 | 377 | 2.09 | 2.0 | 10.45 |
| 32K | fullctx | 35 | 16.62 | 16.32 | 16.89 | 452 | 443 | 462 | 5.02 | 28.6 | 11.38 |
| 32K | nano | 35 | 19.96 | 19.45 | 20.34 | 367 | 358 | 373 | 7.43 | 2.1 | 10.45 |
| 64K | fullctx | 35 | 25.66 | 25.26 | 26.32 | 549 | 542 | 555 | 13.97 | 54.5 | 11.40 |
| 64K | nano | 35 | 27.06 | 26.48 | 28.04 | 366 | 358 | 376 | 14.56 | 2.1 | 10.47 |
| 128K | fullctx | 35 | 58.11 | 57.75 | 59.31 | 767 | 751 | 781 | 46.22 | 105.0 | 11.42 |
| 128K | nano | 35 | 41.25 | 40.34 | 43.08 | 366 | 359 | 377 | 28.75 | 2.1 | 10.46 |
| 256K | fullctx | 35 | 187.16 | 185.89 | 195.42 | 1253 | 1219 | 1280 | 174.70 | 207.0 | 11.33 |
| 256K | nano | 35 | 69.97 | 67.83 | 73.92 | 368 | 359 | 404 | 57.38 | 2.1 | 10.39 |


Speedup (full context / Nano), per-history paired ratio, mean [min-max]:

| N | pairs | e2e ours | e2e paper | TTFT ours | TTFT paper |
|---|---|---|---|---|---|
| 512 | 35 | 0.93x [0.91-0.95] |  | 0.98x [0.95-1.01] |  |
| 2K | 35 | 0.90x [0.87-0.92] |  | 0.98x [0.95-1.00] |  |
| 8K | 35 | 0.86x [0.84-0.88] |  | 1.03x [1.00-1.05] |  |
| 32K | 35 | 0.83x [0.82-0.85] | 1.00x | 1.23x [1.19-1.26] |  |
| 64K | 35 | 0.95x [0.92-0.96] | 1.20x | 1.50x [1.46-1.53] |  |
| 128K | 35 | 1.41x [1.35-1.45] | 1.86x | 2.09x [2.02-2.14] | 2.44x |
| 256K | 35 | 2.68x [2.53-2.86] | 3.99x | 3.40x [3.12-3.51] | 4.19x |


Paper absolutes: e2e full context 30.07 s (128K) / 94.47 s (256K), Nano 16.19 / 23.69 s, Temp-LoRA 700.82 / 1400.88 s; TTFT Nano 104-112 ms, full context ~257 / ~445 ms. HF's MoE expert loop puts a ~350 ms floor under our TTFT on A100.


### Writer cost and 256K probe (validation files, 1-5 histories per point)

| series | N | n | prep_med_s | prep_min_s | prep_max_s | e2e_med_s | ttft_med_ms |
|---|---|---:|---:|---:|---:|---:|---:|
| fullctx | 512 | 4 | 0.40 | 0.38 | 0.41 | 12.11 | 364 |
| fullctx | 2K | 4 | 0.45 | 0.45 | 9.11 | 12.14 | 360 |
| fullctx | 8K | 5 | 1.08 | 0.85 | 1.14 | 12.69 | 373 |
| fullctx | 32K | 2 | 4.81 | 4.79 | 4.83 | 16.61 | 442 |
| fullctx | 256K | 1 | 175.42 | 175.42 | 175.42 | 187.94 | 1235 |
| nano-fast | 512 | 2 | 0.40 | 0.40 | 0.40 | 12.90 | 384 |
| nano-fast | 2K | 2 | 0.88 | 0.85 | 0.92 | 13.44 | 374 |
| nano-fast | 8K | 2 | 2.41 | 2.26 | 2.56 | 14.92 | 370 |
| nano-fast | 32K | 2 | 9.22 | 9.03 | 9.41 | 21.68 | 363 |
| nano-fast | 256K | 1 | 71.20 | 71.20 | 71.20 | 83.69 | 374 |
| nano-reference | 512 | 4 | 0.90 | 0.89 | 0.91 | 13.45 | 372 |
| nano-reference | 2K | 4 | 2.77 | 2.68 | 2.82 | 15.50 | 375 |
| nano-reference | 8K | 5 | 10.15 | 9.66 | 10.26 | 22.79 | 370 |
| nano-reference | 32K | 2 | 39.18 | 38.88 | 39.48 | 51.81 | 365 |
| nano-reference | 256K | 1 | 310.32 | 310.32 | 310.32 | 322.92 | 367 |


Validation nano runs use 'messages' slice packing; the sweep uses 'dense'.


## E2  Storage per trajectory (measured, safetensors)

| N | Nano | full ctx (BF16 GDN) | full ctx (FP32 GDN) | ratio BF16 | ratio FP32 | paper full ctx | paper ratio | KV B/token | fullctx records | source |
|---|---|---|---|---|---|---|---|---|---:|---|
| 512 | 40.00 MiB | 41.94 MiB | 71.94 MiB | 1.0x | 1.8x | 72.5 MiB | 1.8x | 20480 | 39 | sweep+validation |
| 2K | 40.00 MiB | 71.94 MiB | 101.94 MiB | 1.8x | 2.5x | 102.5 MiB | 2.6x | 20480 | 39 | sweep+validation |
| 8K | 40.00 MiB | 191.91 MiB | 221.91 MiB | 4.8x | 5.5x | 222.5 MiB | 5.6x | 20480 | 40 | sweep+validation |
| 32K | 40.00 MiB | 670.35 MiB | 700.35 MiB | 16.8x | 17.5x | 702.5 MiB | 17.6x | 20480 | 37 | sweep+validation |
| 64K | 40.00 MiB | 1.28 GiB | 1.31 GiB | 32.7x | 33.5x | 1.3 GiB | 33.6x | 20480 | 35 | sweep |
| 128K | 40.00 MiB | 2.51 GiB | 2.54 GiB | 64.3x | 65.1x | 2.6 GiB | 65.6x | 20480 | 35 | sweep |
| 256K | 40.00 MiB | 4.99 GiB | 5.02 GiB | 127.8x | 128.5x | 5.1 GiB | 129.6x | 20480 | 36 | sweep+validation |

- Linear fit, full context BF16 GDN: 20312 B/token + 34.3 MiB; at 128K 2.51 GiB (64.3x Nano).

- Linear fit, full context FP32 GDN: 20312 B/token + 64.3 MiB; at 128K 2.54 GiB (65.1x Nano).

- FP32-GDN column = measured serialized bytes + (FP32 - BF16) recurrent-state bytes. HF stores the GDN recurrent state in BF16; the paper counts FP32. Conv state stays BF16 in both (HF conv kernel width 4, 1.875 MiB; E3's synthetic kv objects use width 3).


## E3  FlashBlade persistence

### Cold restore, one session (O_DIRECT, median ms)

| label | MiB | rdma | tcp | local |
|---|---:|---:|---:|---:|
| memt | 40.0 | 8.2 | 8.5 | 7.9 |
| kv@512 | 71.4 | 13.9 | 14.6 | 16.6 |
| kv@2K | 101.4 | 19.6 | 19.9 | 20.8 |
| kv@8K | 221.4 | 42.6 | 39.2 | 42.4 |
| kv@32K | 701.4 | 124.7 | 102.1 | 135.6 |
| kv@64K | 1341.4 | 241.6 | 215.8 | 256.6 |
| kv@128K | 2621.4 | 454.7 | 380.3 | 469.6 |
| kv@256K | 5181.4 | 848.6 | 724.7 | 658.6 |
| lora | 172.5 | 33.6 | 25.7 | 34.6 |


### Concurrent cold restore (distinct files, one process each)

| object | target | K=8 GB/s | K=8 p50 ms | K=8 p95 ms | K=32 GB/s | K=32 p50 ms | K=32 p95 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| memt (recheck) | rdma | 17.07 | 18 | 24 | 5.10 | 85 | 264 |
| memt | tcp | 16.55 | 20 | 21 | 16.34 | 78 | 82 |
| memt | local | 6.80 | 48 | 49 | 6.67 | 196 | 200 |
| kv@512 (recheck) | rdma | 18.38 | 31 | 34 | 8.81 | 187 | 272 |
| kv@512 | tcp | 17.48 | 34 | 36 | 17.29 | 132 | 140 |
| kv@512 | local | 6.65 | 86 | 90 | 7.26 | 310 | 354 |
| kv@2K | rdma | 18.07 | 44 | 48 | 11.21 | 169 | 305 |
| kv@2K | tcp | 17.59 | 45 | 56 | 17.90 | 186 | 198 |
| kv@2K | local | 7.09 | 109 | 126 | 7.09 | 442 | 506 |
| kv@8K | rdma | 19.27 | 93 | 105 | 18.85 | 271 | 396 |
| kv@8K | tcp | 19.10 | 96 | 108 | 18.58 | 393 | 411 |
| kv@8K | local | 10.98 | 168 | 271 | 8.24 | 836 | 1104 |
| kv@32K | rdma | 20.84 | 268 | 311 | 22.81 | 910 | 1035 |
| kv@32K | tcp | 18.46 | 292 | 325 | 18.16 | 1285 | 1388 |
| kv@32K | local | 8.78 | 653 | 822 | 8.20 | 2567 | 3460 |
| kv@64K | rdma | 20.54 | 531 | 565 | 23.07 | 1783 | 1933 |
| kv@64K | tcp | 18.81 | 589 | 617 | 16.61 | 2656 | 2840 |
| kv@64K | local | 9.74 | 1133 | 1330 | 7.21 | 5711 | 6251 |
| kv@128K | rdma | 20.50 | 1062 | 1210 | 23.28 | 3544 | 3750 |
| kv@128K | tcp | 9.62 | 2251 | 2612 | 19.81 | 4424 | 4550 |
| kv@128K | local | 9.70 | 2164 | 2317 | 7.18 | 11646 | 12408 |
| kv@256K | rdma | 19.41 | 2207 | 2288 | 23.71 | 7079 | 7347 |
| kv@256K | tcp | 15.14 | 2785 | 3035 | 18.29 | 9480 | 9708 |
| kv@256K | local | 10.45 | 4080 | 4188 |  |  |  |
| lora | rdma | 18.19 | 75 | 85 | 15.61 | 314 | 373 |
| lora | tcp | 16.76 | 82 | 267 | 17.85 | 316 | 331 |
| lora | local | 7.40 | 171 | 220 | 9.68 | 552 | 689 |


### Derived: capacity and restore cost per session

| object | bytes/session | sessions per TB | rdma K=1 restore ms | rdma K=32 ms/session (amortised) | rdma K=32 sessions/s | tcp K=1 restore ms | tcp K=32 ms/session (amortised) | tcp K=32 sessions/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| memt | 40.0 MiB | 23,841 | 8.2 | 8.2 | 121.6 | 8.5 | 2.6 | 389.5 |
| kv@512 | 71.4 MiB | 13,354 | 13.9 | 8.5 | 117.6 | 14.6 | 4.3 | 230.9 |
| kv@2K | 101.4 MiB | 9,404 | 19.6 | 9.5 | 105.4 | 19.9 | 5.9 | 168.3 |
| kv@8K | 221.4 MiB | 4,307 | 42.6 | 12.3 | 81.2 | 39.2 | 12.5 | 80.0 |
| kv@32K | 701.4 MiB | 1,360 | 124.7 | 32.2 | 31.0 | 102.1 | 40.5 | 24.7 |
| kv@64K | 1.3 GiB | 711 | 241.6 | 61.0 | 16.4 | 215.8 | 84.7 | 11.8 |
| kv@128K | 2.6 GiB | 364 | 454.7 | 118.1 | 8.5 | 380.3 | 138.8 | 7.2 |
| kv@256K | 5.1 GiB | 184 | 848.6 | 229.1 | 4.4 | 724.7 | 297.0 | 3.4 |
| lora | 172.5 MiB | 5,529 | 33.6 | 11.6 | 86.3 | 25.7 | 10.1 | 98.7 |


sessions per TB = 1e12 / bytes; amortised ms/session = bytes / aggregate K=32 throughput.


### First buffered read after write (NFS close-to-open)

| label | target | 1st buffered GB/s | 2nd buffered (cached) GB/s | O_DIRECT GB/s |
|---|---|---:|---:|---:|
| kv@256K | local | 6.71 | 6.74 | 8.25 |
| kv@256K | rdma | 0.57 | 7.43 | 6.40 |
| kv@256K | tcp | 0.53 | 7.30 | 7.50 |
| kv@32K | local | 5.46 | 5.79 | 5.42 |
| kv@32K | rdma | 0.57 | 6.27 | 5.90 |
| kv@32K | tcp | 0.53 | 5.95 | 7.20 |
| memt | local | 5.24 | 5.24 | 5.32 |
| memt | rdma | 0.55 | 5.44 | 5.09 |
| memt | tcp | 0.51 | 4.75 | 4.93 |


On NFS the first buffered read after write+close revalidates and refetches through the page cache at ~0.55 GB/s, 10x slower than O_DIRECT; on local ext4 it is a page-cache hit. tcp kv@2K's 1st buffered read (3.78 GB/s) is an exception where the client kept its cache.


### Small-object concurrency sweep (cold O_DIRECT, idle path)

Source e3_conc_sweep_small.jsonl (run e3-conc-sweep-2301, 5 reps); supersedes the recheck's K=8/32 small-object cells for the RDMA-dip question. GB/s = aggregate (1e9 B/s).

| object | target | K | GB/s median [min-max] | p50 ms | p95 ms | per-session GB/s |
|---|---|---:|---|---:|---:|---:|
| memt | rdma | 1 | 5.30 [4.86-5.88] | 7.9 | 8.5 | 5.30 |
| memt | rdma | 4 | 11.35 [6.49-17.20] | 14.0 | 25.7 | 3.00 |
| memt | rdma | 8 | 13.87 [12.08-15.00] | 22.2 | 27.1 | 1.89 |
| memt | rdma | 16 | 13.29 [11.33-16.47] | 47.0 | 57.3 | 0.89 |
| memt | rdma | 32 | 5.18 [5.00-14.42] | 92.4 | 263.4 | 0.45 |
| memt | rdma | 64 | 10.08 [9.83-10.20] | 158.0 | 267.3 | 0.27 |
| memt | tcp | 1 | 5.68 [5.01-6.08] | 7.4 | 8.3 | 5.68 |
| memt | tcp | 4 | 11.51 [10.69-14.71] | 14.2 | 15.6 | 2.96 |
| memt | tcp | 8 | 16.05 [13.54-17.05] | 20.0 | 23.2 | 2.10 |
| memt | tcp | 16 | 16.08 [15.27-16.71] | 39.8 | 43.0 | 1.05 |
| memt | tcp | 32 | 16.51 [14.77-16.86] | 78.0 | 88.7 | 0.54 |
| memt | tcp | 64 | 16.68 [7.77-16.97] | 150.6 | 338.1 | 0.28 |
| kv@512 | rdma | 1 | 5.24 [5.02-5.53] | 14.3 | 14.9 | 5.24 |
| kv@512 | rdma | 4 | 16.74 [6.45-17.23] | 17.5 | 40.4 | 4.27 |
| kv@512 | rdma | 8 | 15.81 [14.50-18.23] | 35.2 | 40.5 | 2.13 |
| kv@512 | rdma | 16 | 14.26 [12.46-15.73] | 78.2 | 91.9 | 0.96 |
| kv@512 | rdma | 32 | 8.82 [8.60-8.83] | 214.4 | 274.2 | 0.35 |
| kv@512 | rdma | 64 | 16.34 [15.78-17.80] | 267.0 | 295.2 | 0.28 |
| kv@512 | tcp | 1 | 5.87 [4.86-5.94] | 12.8 | 15.0 | 5.87 |
| kv@512 | tcp | 4 | 14.28 [9.89-15.81] | 20.1 | 24.4 | 3.73 |
| kv@512 | tcp | 8 | 16.61 [15.61-17.39] | 34.1 | 37.6 | 2.20 |
| kv@512 | tcp | 16 | 17.07 [16.89-17.84] | 66.6 | 69.5 | 1.12 |
| kv@512 | tcp | 32 | 17.28 [16.80-17.84] | 130.8 | 139.7 | 0.57 |
| kv@512 | tcp | 64 | 17.12 [16.25-17.33] | 263.5 | 283.1 | 0.28 |


## E4  vLLM + pKVA production path

Source e4_pkva_resume.jsonl (vLLM TP2 + pKVA config B, KV store on FlashBlade RDMA).

pKVA TTFT by N, median [min-max] n (restored: valid samples only); bytes written by the cold request, median per history (KiB/token = bytes_written / history_tokens):

| N | cold TTFT ms | hot TTFT ms | restored TTFT ms | KiB/token | bytes/history | files/history |
|---|---|---|---|---|---|---|
| 2K | 162 [160-173] n=3 | 54 [52-55] n=6 | 84 [76-95] n=9 | 51.6 | 103.13 MiB | 4 |
| 8K | 334 [324-338] n=3 | 65 [64-67] n=6 | 125 [112-130] n=9 | 28.4 | 226.88 MiB | 16 |
| 32K | 1267 [1261-1269] n=3 | 113 [107-118] n=6 | 255 [240-285] n=9 | 24.0 | 763.13 MiB | 62 |
| 64K | 3161 [3123-3186] n=3 | 190 [177-202] n=6 | 433 [414-488] n=9 | 24.0 | 1.49 GiB | 124 |
| 128K | 8676 [8631-8800] n=3 | 332 [318-380] n=6 | 789 [743-984] n=9 | 23.9 | 2.98 GiB | 248 |
| 256K | 26886 [26716-26895] n=3 | 636 [591-672] n=6 | 1539 [1454-1752] n=9 | 23.9 | 5.92 GiB | 492 |

- e4_pkva_resume.jsonl run 20261109T191824-dba88c: plotted restored=54, hot=36, cold=18; excluded from plots: filler_cold=2, first_load_after_kva_start=1, filler_restore=3, end=1 (filler_* evict the kva host pool, first_load_after_kva_start is the first load after a kva_server restart, end holds store_final)

- pKVA store at end of run: 40.30 GiB in 3330 files (includes fillers)


### MemT snapshot vs HF KV resume from FlashBlade (e4_memt_resume.jsonl)

Medians over O_DIRECT ('odirect*') samples; TTFT = restore (read + h2d + pool) + query prefill; total = TTFT + 128-token decode.

| N | system | n | histories | read ms | h2d ms | prefill ms | TTFT ms | total s | file_bytes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 2K | nano | 9 | 3 | 8.3 | 2.4 | 373 | 382 | 12.49 | 40.00 MiB |
| 2K | fullctx | 6 | 3 | 13.0 | 9.1 | 351 | 377 | 11.51 | 71.94 MiB |
| 8K | nano | 9 | 3 | 7.7 | 2.3 | 363 | 372 | 12.49 | 40.00 MiB |
| 8K | fullctx | 6 | 3 | 27.2 | 13.7 | 364 | 413 | 11.55 | 191.94 MiB |
| 32K | nano | 9 | 3 | 8.0 | 2.3 | 362 | 372 | 12.55 | 40.00 MiB |
| 32K | fullctx | 6 | 3 | 117.0 | 35.0 | 423 | 583 | 11.76 | 667.60 MiB |
| 64K | nano | 9 | 3 | 8.0 | 2.3 | 364 | 380 | 12.59 | 40.00 MiB |
| 64K | fullctx | 6 | 3 | 209.2 | 62.9 | 515 | 791 | 12.04 | 1.28 GiB |
| 128K | nano | 9 | 3 | 8.1 | 2.3 | 366 | 378 | 12.60 | 40.00 MiB |
| 128K | fullctx | 6 | 3 | 427.0 | 117.0 | 709 | 1268 | 12.44 | 2.52 GiB |
| 256K | nano | 9 | 3 | 8.0 | 2.4 | 361 | 376 | 12.56 | 40.00 MiB |
| 256K | fullctx | 6 | 3 | 890.7 | 232.1 | 1114 | 2246 | 13.47 | 4.98 GiB |


pagecache_warm samples (excluded above): median read 7.7 ms (n=36).

Largest N with odirect samples so far: {'fullctx': 262144, 'nano': 262144}. Run ids: 20261109T222029-ef3c1b.


## Figures and captions

- **fig_e1_latency**: End-to-end latency (history preparation + 512-token query + forced 128-token decode), mean across histories with min-max band; HF transformers BF16 on 2x A100-80GB, batch 1. Hollow dashed markers are the paper's values (paper (hardware unspecified)). Histories per N (min over systems): 512:35, 2K:35, 8K:35, 32K:35, 64K:35, 128K:35, 256K:35. Decode runs at ~10-11 tok/s in HF here, so ~11-12 s of every point is the 128-token reply.

- **fig_e1_ttft**: Query TTFT for a 512-token query (restore of the prepared state from pinned host memory + query prefill + first token), mean across histories with min-max band. Note: HF transformers' MoE expert loop sets a ~350 ms TTFT floor on A100 for any 512-token prefill, so our absolute TTFTs sit ~3x above the paper's 104-112 ms Nano figure; compare the shape (flat vs rising) and the ratio, not the level. Paper values are hollow/dashed (paper (hardware unspecified)).

- **fig_e1_speedup**: Ratio full-context / MemT-Nano (fast writer) for end-to-end latency and for query TTFT, computed per history and averaged (band = min-max across histories). Horizontal line = parity. End-to-end latency: ours crossover ~70,851 tokens; Query TTFT: ours crossover ~3,580 tokens. Paper ratios hollow/dashed (paper (hardware unspecified)); paper puts e2e parity at ~32K.

- **fig_e1_writer_cost**: History-preparation time: MemT-Nano commit_memory with the shipped reference writer vs our fast writer, and full-context KV prefill. Line = median, dots = individual histories. Source: e1_validate.jsonl + e1_probe256k.jsonl, which hold only 1-5 histories per point (fullctx@512:4, fullctx@2K:4, fullctx@8K:5, fullctx@32K:2, fullctx@256K:1, nano-fast@512:2, nano-fast@2K:2, nano-fast@8K:2, nano-fast@32K:2, nano-fast@256K:1, nano-reference@512:4, nano-reference@2K:4, nano-reference@8K:5, nano-reference@32K:2, nano-reference@256K:1); nano slices packed by message ('messages' slice mode), the main sweep uses 'dense' slicing. One fullctx point (history 0 @2K, 9.1 s) is a first-run warm-up outlier; the median ignores it.

- **fig_e2_storage**: Serialized per-trajectory state after history prep (safetensors bytes). Full context: KV of the 10 attention layers (20 KiB/token) + 30 Gated-DeltaNet layers' recurrent and conv state. HF stores the GDN recurrent state in BF16 (~30 MiB); the paper counts it in FP32 (~60 MiB), shown dotted (derived from the measured record by swapping the recurrent-state dtype). Nano: 10 x FP32 [1,16,256,256] memory = 40 MiB, constant. Paper values hollow/dashed (paper (hardware unspecified)). Sources: E1 sweep + validation/probe records (state bytes are deterministic given the token count).

- **fig_e3_restore_latency**: One session (K=1) reading its state object with O_DIRECT (client page cache bypassed; storage-side cache may be warm), median of 5 reps. rdma = FlashBlade NFSv3 over RDMA, tcp = FlashBlade NFSv4.1 over TCP (both nconnect=16, 200 GbE), local = NVMe RAID ext4. Isolated markers are memt (40 MiB) and lora (172.5 MiB). kv@N objects are synthetic with the paper's FP32-GDN layout (e3_persist_io.py).

- **fig_e3_concurrency**: K processes each read a distinct state file with O_DIRECT, released together by a barrier; aggregate GB/s = K x bytes / wall time, median over reps (GB = 1e9 B). Client is one DGX A100. rdma memt cells come from e3_recheck_rdma_small.jsonl (re-measurement of the full run).

- **fig_e3_concurrency_latency**: Per-session wall time from barrier release to last byte, p50 (solid) and p95 (dotted) across sessions and reps; one y-scale per panel (log). rdma memt cells come from e3_recheck_rdma_small.jsonl (re-measurement of the full run).

- **fig_e3_loader_overhead**: FlashBlade rdma, one session. Both paths read with O_DIRECT; the library path then copies into a Python bytes object and parses with safetensors.numpy.load, which caps it near 0.67 GB/s on every target (CPU-bound copy, not storage). Label = slowdown factor.

- **fig_e3_conc_sweep**: Source e3_conc_sweep_small.jsonl (run e3-conc-sweep-2301), 5 reps per point, run with the storage path otherwise idle. K processes each read a distinct file with O_DIRECT behind a barrier; top = aggregate GB/s (median, band = min-max over reps), bottom = p95 per-session latency. This sweep supersedes the K=8/32 small-object cells of e3_recheck_rdma_small.jsonl for the question of whether the RDMA dip is real: tcp stays flat at ~16-17 GB/s from K=8 up, while rdma drops at K=32 (memt median 5.2 GB/s, but max 14.4 GB/s, so the dip is intermittent across reps). The fig_e3_concurrency figures keep their original sources.

- **fig_e4_pkva_ttft**: Source e4_pkva_resume.jsonl. Qwen3.6-35B-A3B on vLLM 0.32 TP2 + pKVA 1.3.0, KV store on FlashBlade NFSv3/RDMA, 512-token query. Median per N with min-max band over 2K:3, 8K:3, 32K:3, 64K:3, 128K:3, 256K:3 histories. cold = first request (prefill computed, KV written); hot = repeat with GPU prefix-cache hit; restored = after a kva_server restart plus a filler history to evict host/GPU caches, so the KV is read back from FlashBlade (only samples with valid == True, i.e. cached tokens matched the expected hit).

- **fig_e4_bytes_per_token**: bytes_written / files_written of the cold request in e4_pkva_resume.jsonl, divided by history tokens, median over histories. The 20 KiB/token line is the attention-only KV of Qwen3.6-35B-A3B; anything above it is GDN/mamba state checkpoints. Right panel: same data per session with the 40 MiB MemT-Nano snapshot for reference.

- **fig_e4_memt_resume**: Resume of a saved session from FlashBlade (NFSv3/RDMA) on HF transformers, 2x A100: O_DIRECT read into a pinned host pool, host-to-device copy, then the 512-token query prefill (first token). Median over 'odirect*' samples (fullctx@2K:6, fullctx@8K:6, fullctx@32K:6, fullctx@64K:6, fullctx@128K:6, fullctx@256K:6, nano@2K:9, nano@8K:9, nano@32K:9, nano@64K:9, nano@128K:9, nano@256K:9); pagecache_warm samples excluded. Source e4_memt_resume.jsonl, fields ['read_s', 'h2d_s', 'prefill_s'].


## Data-quality notes (auto-detected + known)

- E1: history_tokens differs from nominal N in 294/490 records (max 1.00%; sampler tolerance is +-1%, so 'N' is a bucket, not an exact count)

- E1 sweep nano slice_mode = ['dense']; the validation/probe files used 'messages' slicing, so fast-writer prep there is not directly comparable with the sweep

- E1: writer_kind is null on reference-writer records in e1_validate/e1_probe256k; the 'writer' field is authoritative (writer_kind was added later)

- E1: restore_s is a pinned-host -> GPU copy (state already in host RAM), not a storage read; storage restore is E3

- e1_validate: fullctx prep_s outlier history 0 @N=2K: 9.11 s (run 20261109T163407-aad11a; likely first-run warm-up)

- E3: local kv@262144 K=32 skipped: needs 161.9 GiB, free 155.9 GiB minus reserve 20.0 GiB

- E3: original full-run cell rdma memt K=8 read 1.97 GB/s, p95 573 ms; recheck reads 17.07 GB/s (e3_recheck_rdma_small.jsonl, used instead)

- E3: rdma kv@512 aggregate restore drops from 18.4 GB/s (K=8) to 8.8 GB/s (K=32), p95 272 ms [recheck]

- E3: rdma memt aggregate restore drops from 17.1 GB/s (K=8) to 5.1 GB/s (K=32), p95 264 ms [recheck]

- E3: tcp kv@131072 K=8 aggregate restore 9.6 GB/s vs 18.5 GB/s median for kv objects >=8K at the same K/target (p50 2251 ms); single-rep outlier, worth a recheck

- E3: 'st_load_cold' is O_DIRECT read + bytes() copy + safetensors.numpy.load, i.e. the loader's CPU cost on top of a cold read; it is ~0.67 GB/s on every target, so it measures the CPU path, not storage

- E3: kv@N objects are synthetic (random BF16-entropy data) with FP32 GDN recurrent state and a width-3 conv state, ~0.5 MB smaller than the HF-measured FP32-GDN accounting
