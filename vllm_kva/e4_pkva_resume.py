#!/usr/bin/env python3
"""E4 part 1: resume a session from FlashBlade with vLLM + pKVA (full-context KV + GDN checkpoints).

Prereqs: kva_server (2 GiB pinned host pool: kva-memt-a-e4.json) + vLLM (config B, MNBT 16896) up via
  LOGDIR=$M1T_ROOT/logs/e4 KVA_CONFIG=$PWD/kva-memt-a-e4.json ./start.sh all
Token ids come from bench/e4_build_ids.py (identical ids are used by the HF part).

Per (history, N), in order:
  cold      nothing cached anywhere (distinct conversation per N, so no shared stored prefix): compute
  hot x2    immediate repeat: GPU prefix-cache hit
  restored  x R: restart kva_server only (vLLM stays up; empties the pinned host pool -- flooding the pool with
            other prefixes was tried and left 6-18 of 62 target chunks resident because reclaim frees ~430 MB of
            scattered chunks at the 80% watermark, not strict LRU), send a 130K filler request (absorbs vLLM
            re-registration and the first load after a kva start; recorded in "eviction"), reset GPU prefix cache,
            then the target request.
            Valid only if every LoadKV for the request reports resident=0 and cpu=chunks (read from the
            FlashBlade store). Invalid samples are kept (valid=false) and retried.
kvstore bytes / file count are taken before and after each history's cold write (after saves settle).
"""
import argparse, datetime as dt, json, os, re, subprocess, sys, time, uuid
import requests

CHUNK = 1056
HERE = os.path.dirname(os.path.abspath(__file__))
M1T_ROOT = os.environ.get("M1T_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
L_DEFAULT = os.environ.get("LOGDIR") or os.path.join(M1T_ROOT, "logs", "e4")


def log(m):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {m}", file=sys.stderr, flush=True)


class Env:
    def __init__(self, a):
        self.a = a
        self.port = a.port
        self.kva_log = a.kva_log
        self.store = a.kvstore

    # ---- vLLM
    def scrape(self):
        keys = ["vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total",
                "vllm:external_prefix_cache_queries_total", "vllm:external_prefix_cache_hits_total",
                "vllm:prompt_tokens_total"]
        out = {}
        txt = requests.get(f"http://localhost:{self.port}/metrics", timeout=10).text
        for line in txt.splitlines():
            if line.startswith("#"):
                continue
            for k in keys:
                if line.startswith(k + "{") or line.startswith(k + " "):
                    out[k] = out.get(k, 0.0) + float(line.rsplit(" ", 1)[1])
            if line.startswith("vllm:prompt_tokens_by_source_total"):
                out[line.rsplit(" ", 1)[0]] = float(line.rsplit(" ", 1)[1])
        return out

    def reset(self):
        r = requests.post(f"http://localhost:{self.port}/reset_prefix_cache", timeout=60)
        return r.status_code, r.text[:200]

    def request(self, ids, max_tokens=128):
        body = {"model": self.a.model, "prompt": ids, "max_tokens": max_tokens, "temperature": 0,
                "stream": True, "ignore_eos": True, "stream_options": {"include_usage": True}}
        m0 = self.scrape()
        k0 = self.kva_mark()
        t0 = time.perf_counter()
        ts = dt.datetime.now().isoformat(timespec="milliseconds")
        ttft, n, usage, text = None, 0, None, []
        with requests.post(f"http://localhost:{self.port}/v1/completions", json=body, stream=True,
                           timeout=3600) as r:
            r.raise_for_status()
            for raw in r.iter_lines():
                if not raw or not raw.startswith(b"data: "):
                    continue
                if raw[6:] == b"[DONE]":
                    break
                ev = json.loads(raw[6:])
                if ev.get("usage"):
                    usage = ev["usage"]
                ch = ev.get("choices") or []
                if ch and ch[0].get("text"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    n += 1
                    text.append(ch[0]["text"])
        total = time.perf_counter() - t0
        time.sleep(1.0)
        m1 = self.scrape()
        delta = {k: m1.get(k, 0) - m0.get(k, 0) for k in set(m0) | set(m1) if m1.get(k, 0) != m0.get(k, 0)}
        kva = self.kva_since(k0)
        cached = ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
        return {"t_start": ts, "ttft_s": ttft, "total_s": total, "stream_chunks": n, "usage": usage,
                "cached_tokens": cached, "metric_delta": delta, "kva": kva,
                "output_head": "".join(text)[:120]}

    # ---- kva_server log (read by byte offset; never written to)
    def kva_mark(self):
        return os.path.getsize(self.kva_log)

    def kva_since(self, off):
        time.sleep(0.2)
        with open(self.kva_log, "rb") as f:
            f.seek(off)
            lines = f.read().decode(errors="replace").splitlines()
        load = [l for l in lines if "LoadKV" in l]
        comp = []
        for l in load:
            m = re.search(r"LoadKV completed: chunks=(\d+), success=(\w+), resident=(\d+), gpu=(\d+), cpu=(\d+)", l)
            if m:
                comp.append({"ts": l[:23], "chunks": int(m[1]), "success": m[2] == "true", "resident": int(m[3]),
                             "gpu": int(m[4]), "cpu": int(m[5])})
        tss = [l[:23] for l in load]

        def p(s):
            return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f")
        span = (p(tss[-1]) - p(tss[0])).total_seconds() if len(tss) >= 2 else None
        n_save = sum(1 for l in lines if "SaveKV" in l)
        warn = [l for l in lines if "[WARN]" in l or "[ERROR]" in l][:20]
        return {"load_lines": load[:400], "load_completed": comp,
                "load_chunks_sum": sum(c["chunks"] for c in comp),
                "load_resident_sum": sum(c["resident"] for c in comp),
                "load_cpu_sum": sum(c["cpu"] for c in comp), "load_gpu_sum": sum(c["gpu"] for c in comp),
                "load_requests": len(comp), "load_span_s": span,
                "load_first_ts": tss[0] if tss else None, "load_last_ts": tss[-1] if tss else None,
                "savekv_lines": n_save, "warn_err_lines": warn}

    # ---- store
    def store_stat(self):
        b = int(subprocess.check_output(["du", "-sb", self.store]).split()[0])
        nf = int(subprocess.check_output(f"find {self.store} -type f | wc -l", shell=True))
        return {"bytes": b, "files": nf, "ts": dt.datetime.now().isoformat(timespec="seconds")}

    def settle(self, quiet=3, poll=2.0, max_s=300):
        t0, last, same = time.time(), None, 0
        while time.time() - t0 < max_s:
            s = self.store_stat()
            if last and s["bytes"] == last["bytes"] and s["files"] == last["files"]:
                same += 1
                if same >= quiet:
                    return s, time.time() - t0
            else:
                same = 0
            last = s
            time.sleep(poll)
        return last, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default=os.path.join(M1T_ROOT, "scratch_state", "e4", "e4_ids.json"))
    ap.add_argument("--out", default=os.path.join(M1T_ROOT, "results", "e4_pkva_resume.jsonl"))
    ap.add_argument("--port", type=int, default=8101)
    ap.add_argument("--model", default="qwen3.6-35b-a3b")
    ap.add_argument("--kva-log", default=f"{L_DEFAULT}/kva-memt-a.log")
    ap.add_argument("--kvstore", default=os.environ.get("KVSTORE_DIR") or os.path.join(M1T_ROOT, "kvstore", "memt-a"))
    ap.add_argument("--restored", type=int, default=3)
    ap.add_argument("--hot", type=int, default=2)
    ap.add_argument("--max-retry", type=int, default=3)
    ap.add_argument("--only", default="", help="comma list of <h>_<N> keys (default: all)")
    ap.add_argument("--skip-warmup", action="store_true")
    ap.add_argument("--restore-only", action="store_true",
                    help="skip cold/hot (history already in the store); only restored samples")
    ap.add_argument("--kva-config", default=os.path.join(HERE, "kva-memt-a-e4.json"))
    ap.add_argument("--tag", default="main")
    a = ap.parse_args()
    E = Env(a)
    D = json.load(open(a.ids))
    run_id = dt.datetime.now().strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    import importlib.metadata as md
    meta = {"run_id": run_id, "experiment": "e4_pkva_resume", "host": os.uname().nodename,
            "versions": {"vllm": md.version("vllm"), "purekva_plugin": md.version("purekva-plugin"),
                         "kva_server": "1.3.0"},
            "model": "Qwen/Qwen3.6-35B-A3B", "cuda_visible": "4,5",
            "flags": "--tensor-parallel-size 2 --dtype bfloat16 --gpu-memory-utilization 0.90 --max-model-len 262144 "
                     "--enable-prefix-caching --mamba-cache-mode align --enable-prompt-tokens-details "
                     "--language-model-only --max-num-batched-tokens 16896 kv_connector=PureKVConnector_V1 kv_both; "
                     "VLLM_SERVER_DEV_MODE=1",
            "tag": a.tag, "kva_config_file": a.kva_config,
            "kva_config": {"pool": json.load(open(a.kva_config))["pool"],
                           "storage": "localfs " + a.kvstore, "chunk_batch_size": 32, "dynamic": True},
            "storage": "FlashBlade NFSv3 over RDMA (/mnt/data, nconnect=16)", "chunk_tokens": CHUNK,
            "gen_tokens": 128, "ids_file": a.ids}
    fout = open(a.out, "a")

    def emit(rec):
        rec = {**meta, "ts": dt.datetime.now().isoformat(timespec="seconds"), **rec}
        fout.write(json.dumps(rec) + "\n")
        fout.flush()

    fillers = list(D["fillers"].values())
    fi = [0]

    part = [0]

    def restart_kva():
        """Archive the kva log (start.sh truncates it), restart kva_server only; vLLM stays up and re-registers."""
        part[0] += 1
        arch = f"{a.kva_log[:-4]}.{run_id}.p{part[0]:03d}.log"
        subprocess.run(["cp", a.kva_log, arch], check=True)
        env = {**os.environ, "LOGDIR": os.path.dirname(a.kva_log),
               "KVA_CONFIG": a.kva_config, "KVSTORE_DIR": a.kvstore}
        t0 = time.perf_counter()
        subprocess.run([os.path.join(HERE, "stop.sh"), "kva"], env=env, check=True, capture_output=True)
        subprocess.run([os.path.join(HERE, "start.sh"), "kva"], env=env, check=True, capture_output=True)
        return {"kva_restart_s": time.perf_counter() - t0, "kva_log_archived": arch}

    def evict():
        """Guarantee an empty pinned pool: restart kva_server, then a full filler request (absorbs vLLM
        re-registration + any first-load cost after a kva start; recorded), then reset the GPU prefix cache."""
        rs = restart_kva()
        f = fillers[fi[0] % len(fillers)]
        fi[0] += 1
        E.reset()
        r = E.request(f["prompt_ids"])
        E.reset()
        kv = r["kva"]
        return {**rs, "filler": f"{f['history_idx']}_{f['N']}", "filler_ttft_s": r["ttft_s"], "filler_total_s": r["total_s"],
                "filler_cached": r["cached_tokens"], "filler_load_chunks": kv["load_chunks_sum"],
                "filler_load_resident": kv["load_resident_sum"], "filler_load_cpu": kv["load_cpu_sum"],
                "filler_load_span_s": kv["load_span_s"], "filler_warn_err": kv["warn_err_lines"][:6]}

    if not a.skip_warmup:
        log("warm-up: write both fillers cold, then restore each (first load after kva_server start recorded)")
        for k, f in enumerate(fillers):
            s0 = E.store_stat()
            r = E.request(f["prompt_ids"])
            s1, st = E.settle()
            emit({"label": "filler_cold", "history_idx": f["history_idx"], "conv_id": f["conv_id"], "N": f["N"],
                  "history_tokens": f["history_tokens"], "prompt_tokens": f["prompt_tokens"], **r,
                  "store_before": s0, "store_after": s1, "settle_s": st})
            log(f"  filler {k} cold ttft={r['ttft_s']:.2f}s wrote {s1['bytes'] - s0['bytes']} B")
        for k, f in enumerate(fillers + fillers):
            E.reset()
            r = E.request(f["prompt_ids"])
            lbl = "first_load_after_kva_start" if k == 0 else "filler_restore"
            emit({"label": lbl, "seq": k, "history_idx": f["history_idx"], "conv_id": f["conv_id"], "N": f["N"],
                  "history_tokens": f["history_tokens"], "prompt_tokens": f["prompt_tokens"], **r})
            kv = r["kva"]
            log(f"  {lbl} {k}: ttft={r['ttft_s']:.3f}s cached={r['cached_tokens']} chunks={kv['load_chunks_sum']} "
                f"resident={kv['load_resident_sum']} cpu={kv['load_cpu_sum']} span={kv['load_span_s']}")

    def cold_hot(key, H, ids, base):
        E.reset()  # GPU cache empty; store has no chunk of this conversation (distinct conversation per N)
        s0 = E.store_stat()
        r = E.request(ids)
        s1, st = E.settle()
        wrote = {"store_before": s0, "store_after": s1, "settle_s": st,
                 "bytes_written": s1["bytes"] - s0["bytes"], "files_written": s1["files"] - s0["files"]}
        emit({**base, "label": "cold", "sample": 0, **r, **wrote})
        log(f"  cold ttft={r['ttft_s']:.3f}s total={r['total_s']:.2f}s cached={r['cached_tokens']} "
            f"wrote {wrote['bytes_written']} B / {wrote['files_written']} files ({st:.0f}s settle)")
        for i in range(a.hot):
            r = E.request(ids)
            emit({**base, "label": "hot", "sample": i, **r})
            log(f"  hot{i} ttft={r['ttft_s']:.3f}s cached={r['cached_tokens']}")

    keys = [k for k in D["histories"] if not a.only or k in a.only.split(",")]
    for key in keys:
        H = D["histories"][key]
        ids = H["prompt_ids"]
        base = {"history_idx": H["history_idx"], "conv_id": H["conv_id"], "N": H["N"],
                "history_tokens": H["history_tokens"], "prompt_tokens": H["prompt_tokens"],
                "expected_hit_tokens": (H["prompt_tokens"] // CHUNK) * CHUNK}
        log(f"== {key}: history {H['history_tokens']} prompt {H['prompt_tokens']}")
        if not a.restore_only:
            cold_hot(key, H, ids, base)
        good, tries = 0, 0
        while good < a.restored and tries < a.restored + a.max_retry:
            tries += 1
            ev = evict()
            r = E.request(ids)
            kv = r["kva"]
            ext = r["metric_delta"].get("vllm:external_prefix_cache_hits_total", 0)
            loc = r["metric_delta"].get("vllm:prefix_cache_hits_total", 0)
            valid = (kv["load_requests"] > 0 and kv["load_resident_sum"] == 0 and kv["load_gpu_sum"] == 0
                     and kv["load_cpu_sum"] == kv["load_chunks_sum"] and loc == 0 and ext > 0)
            emit({**base, "label": "restored", "sample": good if valid else None, "try": tries, "valid": valid,
                  "eviction": ev, **r})
            log(f"  restored try{tries} valid={valid} ttft={r['ttft_s']:.3f}s total={r['total_s']:.2f}s "
                f"cached={r['cached_tokens']} ext_hits={ext} gpu_hits={loc} chunks={kv['load_chunks_sum']} "
                f"resident={kv['load_resident_sum']} cpu={kv['load_cpu_sum']} span={kv['load_span_s']} "
                f"filler_ttft={ev['filler_ttft_s']:.3f}")
            good += valid
    E.reset()
    emit({"label": "end", "store_final": E.store_stat()})
    log("done")


if __name__ == "__main__":
    main()
