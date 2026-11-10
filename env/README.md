# MemT-1.0-Nano HF-transformers environment

Venv: `/mnt/data/robert-scripts/m1t-test/.venv` (Python 3.12.3, system interpreter).
Frozen package list: `env/requirements-hf.txt`.

## Create

```bash
cd /mnt/data/robert-scripts/m1t-test
UV=/mnt/data/robert-scripts/gpu-scratch/uvbin/uv
export UV_CACHE_DIR=/tmp/claude-1000/-mnt-data-robert-scripts-m1t-test/d52341ba-cd8c-4b9e-83de-e2ab4ba1debb/scratchpad/hf/uv-cache   # any local dir
$UV venv --python /usr/bin/python3.12 .venv
# torch 2.13.0 on PyPI is the CUDA 13.0 build (2.13.0+cu130); same source as gpu-scratch/venv
$UV pip install --python .venv/bin/python \
    torch==2.13.0 transformers==5.12.1 accelerate==1.15.0 safetensors==0.8.0 \
    huggingface_hub numpy flash-linear-attention
# exact reproduction:
# $UV pip install --python .venv/bin/python -r env/requirements-hf.txt
```

Key versions: torch 2.13.0+cu130, triton 3.7.1, transformers 5.12.1 (pinned by the
model card and by the header of `modeling_memt.py`), accelerate 1.15.0, safetensors 0.8.0,
flash-linear-attention / fla-core 0.6.0 (Triton GDN kernels), huggingface-hub 1.33.0.

Notes:
- `modeling_memt.py` imports only torch, safetensors and transformers internals
  (`transformers.models.qwen3_5_moe.modeling_qwen3_5_moe`). It reaches into module-level
  names (`chunk_gated_delta_rule`, `fused_recurrent_gated_delta_rule`, `causal_conv1d_fn`,
  `torch_*` fallbacks) that exist in 5.12.1; do not upgrade transformers without checking.
- `fla` is what gives the 30 Gated-DeltaNet layers a fast chunked kernel. Without it HF falls
  back to `torch_chunk_gated_delta_rule` (pure PyTorch, slow and memory hungry).
- `causal-conv1d` is NOT installed (no prebuilt wheel for torch 2.13/cu130). Both HF and the
  MemT code fall back to `F.conv1d` (prefill) and `torch_causal_conv1d_update` (decode).
  The conv is a 4-tap depthwise conv, so this is a minor cost.

## Run

```bash
cd /mnt/data/robert-scripts/m1t-test
export HF_HOME=/mnt/data/robert-scripts/hf-cache HF_HUB_OFFLINE=1
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python bench/smoke_memt_hf.py > smoke.jsonl
```

## Performance notes (A100, measured 2026-11-09)

- HF picks `experts_implementation="grouped_mm"` by default. On sm80, torch's grouped_mm is a host
  loop over all 256 experts (~20k `aten::mm` + `.item()` per decode step), so decode runs at
  ~6 tok/s. `batched_mm` gives ~11 tok/s for decode and short prompts; keep `grouped_mm` for long
  prefills (`batched_mm` gathers one expert-weight copy per token x top-k). The smoke test switches
  per forward with `model.set_experts_implementation(...)` (cheap, takes effect immediately).
- First use of the fla Triton kernels autotunes (~40 s for the first 2K prefill, ~15 s for the first
  "continue from cached GDN state" prefill). Always warm both paths before timing.
- Cold model load from FlashBlade: ~160 s; warm (page cache): ~20 s. Weights: 32.7 GiB per GPU
  with `device_map="auto"` (layers 0-19 on cuda:0, 20-39 on cuda:1).
