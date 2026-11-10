#!/bin/bash
# Qwen3.6-35B-A3B (hybrid GDN + full attention), TP2 on GPUs 4,5, vLLM 0.32 + pKVA 1.3.0, KV store on FlashBlade.
# Usage: ./start.sh [all|kva|vllm]   (vllm = restart only vLLM, keeping kva_server and its store)
# Env overrides: MAXLEN (default 262144), PORT (8101), GPUS (4,5), HOST (127.0.0.1), MNBT (16896), EXTRA_ARGS
# MNBT = --max-num-batched-tokens. With mamba align mode each prefill step ends on a 1056-token block boundary and
# pKVA stores a full GDN state checkpoint at every step end. vLLM default (2048) => 1 step per block => ~80 KiB/token
# on FlashBlade; 16896 (16 blocks) => ~24 KiB/token and 2x faster cold prefill.
# Paths (env):
#   PKVA_ROOT    required: pKVA 1.3.0 install root, the directory holding usr/local/bin/kva_server and
#                usr/local/lib/cuobj (the unpacked .deb contents; "/" for a system-wide .deb install)
#   VLLM_PY      python of the vLLM 0.32 + purekva_plugin venv (default: python on PATH); its bin/ dir is
#                prepended to PATH and its vllm entry point is used
#   M1T_ROOT     repo root (default: parent of this directory)
#   LOGDIR       logs, pid files and the rendered kva config (default $M1T_ROOT/logs/vllm)
#   KVSTORE_DIR  pKVA localfs store (default $M1T_ROOT/kvstore/memt-a); substituted for the KVSTORE_DIR
#                placeholder in $KVA_CONFIG (default kva-memt-a.json), rendered to $LOGDIR/<config name>
#   HF_HOME      passed through unchanged if set (HF cache holding Qwen/Qwen3.6-35B-A3B)
#   CUDA_HOME    default /usr/local/cuda-13
set -e
: "${PKVA_ROOT:?set PKVA_ROOT to the pKVA 1.3.0 install root (dir containing usr/local/bin/kva_server)}"
KVA=$PKVA_ROOT/usr/local/bin/kva_server
D=$(cd "$(dirname "$0")" && pwd)
M1T_ROOT=${M1T_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
VLLM_PY=${VLLM_PY:-python}
VBIN=$(dirname "$(command -v "$VLLM_PY")")
L=${LOGDIR:-$M1T_ROOT/logs/vllm}
KVSTORE_DIR=${KVSTORE_DIR:-$M1T_ROOT/kvstore/memt-a}
mkdir -p $L $KVSTORE_DIR
GPUS=${GPUS:-4,5}; PORT=${PORT:-8101}; MAXLEN=${MAXLEN:-262144}
SOCK=/tmp/kva-memt-a.sock
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13}
export PATH=$VBIN:$CUDA_HOME/bin:$PATH CUDA_HOME
export LD_LIBRARY_PATH=$PKVA_ROOT/usr/local/lib/cuobj:$LD_LIBRARY_PATH

start_kva() {
  rm -f $SOCK
  local tmpl=${KVA_CONFIG:-$D/kva-memt-a.json} cfg
  cfg=$L/$(basename "$tmpl")
  [ "$(cd "$(dirname "$tmpl")" && pwd)/$(basename "$tmpl")" != "$cfg" ] || { echo "KVA_CONFIG must not live in LOGDIR"; exit 1; }
  sed "s#KVSTORE_DIR#$KVSTORE_DIR#g" "$tmpl" > "$cfg"
  echo "kva config: $cfg (store $KVSTORE_DIR)"
  CUDA_VISIBLE_DEVICES=$GPUS PUREKV_LOG_LEVEL=${PUREKV_LOG_LEVEL:-INFO} setsid nohup $KVA --config "$cfg" \
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
    setsid nohup $VBIN/vllm serve Qwen/Qwen3.6-35B-A3B --served-model-name qwen3.6-35b-a3b \
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
