#!/bin/bash
# Stop vLLM first (it holds the kva_server connection), then kva_server. Usage: ./stop.sh [all|vllm|kva]
# start.sh launches each under setsid, so the pid is also the process-group id; signal the whole group
# so vLLM's EngineCore/Worker children do not survive (they hold GPU memory).
M1T_ROOT=${M1T_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
L=${LOGDIR:-$M1T_ROOT/logs/vllm}   # must match the LOGDIR given to start.sh
case ${1:-all} in all) W="vllm kva";; *) W=$1;; esac
for p in $W; do
  f=$L/$p-memt-a.pid; [ -f $f ] || continue
  pid=$(cat $f)
  kill -TERM -- -$pid 2>/dev/null || kill -TERM $pid 2>/dev/null
  for i in $(seq 60); do pgrep -g $pid >/dev/null 2>&1 || break; sleep 1; done
  kill -KILL -- -$pid 2>/dev/null; rm -f $f; echo "stopped $p-memt-a"
done
