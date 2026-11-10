#!/bin/bash
# Qwen3.6-35B-A3B (hybrid GDN + full attention), TP2 on GPUs 4,5, vLLM 0.32 + pKVA 1.3.0, KV store on FlashBlade.
# Usage: ./start.sh [all|kva|vllm]   (vllm = restart only vLLM, keeping kva_server and its store)
# Env overrides: MAXLEN (default 262144), PORT (8101), GPUS (4,5), HOST (127.0.0.1), MNBT (16896), EXTRA_ARGS
# MNBT = --max-num-batched-tokens. With mamba align mode each prefill step ends on a 1056-token block boundary and
# pKVA stores a full GDN state checkpoint at every step end. vLLM default (2048) => 1 step per block => ~80 KiB/token
# on FlashBlade; 16896 (16 blocks) => ~24 KiB/token and 2x faster cold prefill.
set -e
R=/mnt/data/robert-scripts/kvshowdown
V=$R/venvs/vllm0.32-pkva1.3.0
KVA=$R/pkva/v1.3.0/debcontent/usr/local/bin/kva_server
D=$(cd "$(dirname "$0")" && pwd)
L=${LOGDIR:-/tmp/claude-1000/-mnt-data-robert-scripts-m1t-test/d52341ba-cd8c-4b9e-83de-e2ab4ba1debb/scratchpad/vllm}
mkdir -p $L
GPUS=${GPUS:-4,5}; PORT=${PORT:-8101}; MAXLEN=${MAXLEN:-262144}
SOCK=/tmp/kva-memt-a.sock
export PATH=$V/bin:/usr/local/cuda-13/bin:$PATH CUDA_HOME=/usr/local/cuda-13
export HF_HOME=/mnt/data/robert-scripts/hf-cache
export LD_LIBRARY_PATH=$R/pkva/v1.3.0/debcontent/usr/local/lib/cuobj:$LD_LIBRARY_PATH

start_kva() {
  rm -f $SOCK
  CUDA_VISIBLE_DEVICES=$GPUS PUREKV_LOG_LEVEL=${PUREKV_LOG_LEVEL:-INFO} setsid nohup $KVA --config $D/kva-memt-a.json \
    > $L/kva-memt-a.log 2>&1 & echo $! > $L/kva-memt-a.pid
  for i in $(seq 60); do [ -S $SOCK ] && break; sleep 1; done
  [ -S $SOCK ] || { echo "kva_server did not create $SOCK; see $L/kva-memt-a.log"; exit 1; }
  echo "kva_server up: $SOCK"
}
start_vllm() {
  # --mamba-cache-mode align is required by purekva_plugin for hybrid models (it raises otherwise);
  # it is also vLLM's default when prefix caching is enabled, set explicitly here.
  # VLLM_SERVER_DEV_MODE=1 enables POST /reset_prefix_cache (used to empty the GPU prefix cache without a restart);
  # because dev endpoints are unauthenticated, bind to localhost only.
  VLLM_SERVER_DEV_MODE=1 CUDA_VISIBLE_DEVICES=$GPUS PUREKVA_SERVER_ADDR=unix:$SOCK PUREKV_CHUNK_SIZE=512 \
  PUREKV_LOG_LEVEL=${PUREKV_LOG_LEVEL:-INFO} PUREKVA_LOG_LEVEL=${PUREKVA_LOG_LEVEL:-INFO} \
    setsid nohup $V/bin/vllm serve Qwen/Qwen3.6-35B-A3B --served-model-name qwen3.6-35b-a3b \
      --host ${HOST:-127.0.0.1} --port $PORT --tensor-parallel-size 2 --dtype bfloat16 \
      --gpu-memory-utilization 0.90 --max-model-len $MAXLEN \
      --enable-prefix-caching --mamba-cache-mode align --enable-prompt-tokens-details --language-model-only --max-num-batched-tokens ${MNBT:-16896} ${EXTRA_ARGS} \
      --kv-transfer-config '{"kv_connector":"PureKVConnector_V1","kv_role":"kv_both","kv_connector_module_path":"purekva_plugin"}' \
      >> $L/vllm-memt-a.log 2>&1 & echo $! > $L/vllm-memt-a.pid
  echo "vLLM starting on :$PORT (GPUs $GPUS, max-model-len $MAXLEN); log $L/vllm-memt-a.log"
  for i in $(seq 2400); do curl -sf localhost:$PORT/health >/dev/null && { echo "healthy after ${i}s"; return 0; }
    kill -0 $(cat $L/vllm-memt-a.pid) 2>/dev/null || { echo "vLLM exited; see log"; return 1; }; sleep 1; done
  echo "timeout waiting for health"; return 1
}
case ${1:-all} in
  kva) start_kva ;;
  vllm) echo "=== vLLM start $(date -Is) ===" >> $L/vllm-memt-a.log; start_vllm ;;
  all) echo "=== vLLM start $(date -Is) ===" >> $L/vllm-memt-a.log; start_kva; start_vllm ;;
esac
