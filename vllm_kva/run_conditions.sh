#!/bin/bash
# Run cold / hot / restored(after GPU prefix-cache reset) for one history length.
# Usage: run_conditions.sh N [seed_tag] [conditions...]
#   conditions default: "cold hot reset"   (reset = POST /reset_prefix_cache, then repeat request;
#   GPU prefix cache is empty, kva_server keeps running, so the prefix must come from pKVA)
#   "restored" = just repeat the request (use after restarting vLLM and/or kva_server)
# Appends JSON lines to $OUT (default <logdir>/results.jsonl) and prints kvstore size deltas.
set -e
N=$1; TAG=${2:-hist}; shift 2 || true; CONDS=${*:-cold hot reset}
D=$(cd "$(dirname "$0")" && pwd)
M1T_ROOT=${M1T_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
L=${LOGDIR:-$M1T_ROOT/logs/vllm}
OUT=${OUT:-$L/results.jsonl}; PORT=${PORT:-8101}
KVS=${KVSTORE_DIR:-$M1T_ROOT/kvstore/memt-a}   # must match the store start.sh rendered into the kva config
PY=${VLLM_PY:-python}                          # python of the vLLM + pKVA venv (needs transformers for bench_ttft.py)
export HF_HUB_OFFLINE=1                        # HF_HOME is passed through from the caller's environment
for c in $CONDS; do
  if [ $c = reset ]; then
    echo "reset_prefix_cache: $(curl -s -X POST localhost:$PORT/reset_prefix_cache)"
  fi
  b=$(du -sb $KVS | cut -f1); nf0=$(find $KVS -type f | wc -l)
  echo "=== N=$N tag=$TAG cond=$c $(date -Is) ===" | tee -a $L/vllm-memt-a.log >> $L/kva-memt-a.log
  $PY $D/bench_ttft.py --n $N --label "$c" --seed-tag $TAG --port $PORT --out $OUT
  sleep ${SETTLE:-15}   # let async saves land on FlashBlade
  a=$(du -sb $KVS | cut -f1); nf1=$(find $KVS -type f | wc -l)
  echo "kvstore: before=$b after=$a delta=$((a-b)) bytes files=$nf0->$nf1 ($((nf1-nf0)) new)"
  echo "{\"kvstore\":{\"n\":$N,\"tag\":\"$TAG\",\"cond\":\"$c\",\"before\":$b,\"after\":$a,\"files_before\":$nf0,\"files_after\":$nf1}}" >> $OUT
done
