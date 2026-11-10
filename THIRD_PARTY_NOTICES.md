# Third-party notices

The code in this repository is licensed under Apache-2.0 (see `LICENSE`). It depends on, measures,
or is derived from the following external works. Nothing listed here is redistributed in this
repository unless stated.

## Model weights and model code (downloaded at run time, not redistributed)

| Work | License | How it is used |
|---|---|---|
| [MemTensor/MemT-1.0-Nano](https://huggingface.co/MemTensor/MemT-1.0-Nano) (weights and `modeling_memt.py`) | Apache-2.0 | Loaded with `transformers` (`trust_remote_code`) in the HF-side benchmarks (`bench/`). `bench/memt_fast_writer.py` monkeypatches the shipped `MemTMemory.forward` write path at run time; it reimplements the published GDN-2 recurrence with a `flash-linear-attention` kernel and does not copy code from the checkpoint. |
| [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | see model card | Backbone of MemT-1.0-Nano; served by vLLM as the full-context baseline (`vllm_kva/`). |

## Upstream MemT-1.0 serving code (not used)

The authors' repository [MemTensor/MemT-1.0](https://github.com/MemTensor/MemT-1.0) contains an
SGLang serving layer (`inference/memt_sglang`) licensed under **PolyForm Noncommercial 1.0.0**
(with an MIT-licensed vendored `fla`). No code from that repository is included or derived here.
The HF-side scripts only rely on the Apache-2.0 model code shipped with the checkpoint, and the
`memt-memory-v1` snapshot format is read and written with `safetensors` directly.

## Datasets (not redistributed)

| Work | License | How it is used |
|---|---|---|
| [Mohammadta/BEAM](https://huggingface.co/datasets/Mohammadta/BEAM) (Tavakoli et al., arXiv:2510.27246) | CC BY-SA 4.0 | The 35 conversations of the 1M split are the session histories for E1/E2/E4, as in the paper. The scripts read the dataset from a local HF cache; the result files record conversation indices and token counts, not text. |

## Python packages (installed with pip, not vendored)

| Package | License |
|---|---|
| torch | BSD-3-Clause |
| transformers, accelerate, safetensors, huggingface_hub | Apache-2.0 |
| flash-linear-attention (`fla`, `fla-core`) 0.6.0 | MIT |
| triton | MIT |
| vLLM 0.32 | Apache-2.0 |
| numpy, pandas, pyarrow, matplotlib, requests | BSD-3-Clause / Apache-2.0 / PSF-style (see each package) |

Exact versions are frozen in `env/requirements-hf.txt` and `env/requirements-analysis.txt`.

## Pure Storage KV Accelerator (pKVA)

`vllm_kva/` launches the `kva_server` binary and the `purekva_plugin` vLLM connector from a
separately installed Pure Storage KV Accelerator 1.3.0 package. Neither is included here; obtain
them from Pure Storage under their own terms. The scripts only pass configuration and read the
connector's log output.

## Paper

MemTensor team, *MemT-1.0: Towards Memory-Native Evolution*, arXiv:2610.26825 (2026). Reference
values quoted in `results/summary.md`, `docs/`, and the figures are transcribed from the paper
and labelled as such.
