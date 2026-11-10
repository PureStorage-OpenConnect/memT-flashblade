#!/usr/bin/env python3
"""E3: FlashBlade persistence benchmark for per-session LLM state (pure I/O, no GPU).

Measures what it costs to save and restore per-session state objects on
FlashBlade (NFSv3/RDMA, NFSv4.1/TCP) versus local NVMe:

  memt  MemT-Nano memory snapshot ("memt-memory-v1"): 10 x FP32 [1,16,256,256] = 40 MiB, constant.
  kv    Full-context state for Qwen3.6-35B-A3B at history length T:
        10 attention layers x {K,V} BF16 [1,2,T,256]  (= 20 KiB/token)
        + 30 GDN layers x (recurrent FP32 [1,32,128,128] + conv BF16 [1,8192,3])  (~61.4 MiB, constant)
  lora  Temp-LoRA adapter baseline: one 172.5 MiB BF16 blob.
  file:<path>  any real .safetensors file (bytes are used as-is).

Every object is materialised in RAM once as the exact bytes of a .safetensors file
(8-byte LE header length + JSON header + tensor data), and those bytes are what the raw
I/O ops write/read. BF16 payloads are numpy uint16.

Ops (one JSONL record per op x config, aggregating --reps repetitions):
  write          open(O_CREAT|O_TRUNC) + write all bytes (buffered) + fsync + close. Fresh file each rep.
  read_buffered_first  first buffered read after write+close. On NFS, close-to-open revalidation
                 can invalidate the client's own freshly written pages (observed on the RDMA mount),
                 so this is sometimes a server read with kernel readahead. Labelled indeterminate.
  read_cached    second buffered read (page cache primed by read_buffered_first). True page cache.
  read_cold      O_DIRECT read into a page-aligned, pre-faulted mmap buffer (client page cache bypassed).
  st_save        safetensors.numpy.save_file(dict) + fsync (fsync time also reported separately).
  st_load_first  first safetensors.numpy.load_file(path) after st_save (cache state indeterminate).
  st_load_cached second load_file (page cache warm).
  st_load_cold   O_DIRECT read of the file + safetensors.numpy.load(bytes) (what a loader that
                 bypasses the page cache would pay; includes the bytes() copy + parse).
  conc_write     K processes each write+fsync a distinct file simultaneously (once per rep).
  conc_read_cold K processes each O_DIRECT-read a distinct file simultaneously.

See --help and the module-level notes at the bottom of this docstring for semantics.

O_DIRECT tail handling: files are NOT padded. The final request is issued at an aligned offset
with an aligned length that extends past EOF into the (rounded-up) aligned buffer; Linux returns a
short read at EOF on ext4 and on NFS (verified on all three targets on this host). If the kernel
rejects that (EINVAL), the harness falls back to a buffered pread of the unaligned tail and records
tail_mode="buffered_fallback" (the default record value is tail_mode="direct_short_read").

Concurrency uses separate processes (fork), not threads: os.preadv releases the GIL, so threads
would also overlap the I/O, but the per-session resume path is modelled as independent workers each
with its own fd, buffer and address space; processes also keep Python-side work (loop, page faults,
safetensors parsing if enabled) from serialising on one GIL, and fork shares the parent's payload
buffer copy-on-write so K writers don't need K copies of the payload. All workers pre-fault their
read buffer, then wait on a Barrier; timestamps use CLOCK_MONOTONIC, which is system-wide, so the
aggregate GB/s = sum(bytes) / (max(end) - min(start)).

Caveats recorded in each record's "notes":
  * We cannot drop caches (no root). "cold" means the *client* page cache is bypassed via O_DIRECT;
    the FlashBlade (or NVMe device) may still serve from its own DRAM/NVRAM, especially for a file
    written seconds earlier. Use --cold-delay-s to insert a gap if desired.
  * NFS close-to-open consistency: every open() revalidates attributes. After our own write the
    client sometimes concludes the file changed and drops its pages, so the first buffered read is
    not reliably cached; hence read_buffered_first vs read_cached.
  * Large O_DIRECT reads on NFS are split into rsize (512 KiB) RPCs issued in parallel, so a "cold"
    read can be as fast as or faster than a single-threaded page-cache memcpy (~7 GB/s per core).
    cold < cached is therefore NOT a valid invariant on NFS at large sizes.
  * GB/s is decimal (1e9 bytes/s). Sizes are also reported in MiB.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import mmap
import multiprocessing as mp
import os
import platform
import socket
import statistics
import struct
import sys
import time
import uuid
from dataclasses import dataclass, field

import numpy as np

MiB = 1 << 20
GiB = 1 << 30
ALIGN = 4096

DEFAULT_TARGETS = {
    "rdma": "/mnt/data/robert-scripts/m1t-test/scratch_state/e3",
    "tcp": "/mnt/data_tcp/robert-scripts/m1t-test/scratch_state/e3",
    "local": "/home/robert/e3_scratch",
}
DEFAULT_LENGTHS = [512, 2048, 8192, 32768, 65536, 131072, 262144]
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "e3_persist_io.jsonl")

# --------------------------------------------------------------------------------------
# State object specs
# --------------------------------------------------------------------------------------

DTYPE_BYTES = {"F32": 4, "BF16": 2, "F16": 2, "U16": 2, "I64": 8, "F64": 8, "U8": 1, "I32": 4}
NP_DTYPE = {"F32": np.float32, "BF16": np.uint16, "F16": np.float16, "U16": np.uint16,
            "I64": np.int64, "F64": np.float64, "U8": np.uint8, "I32": np.int32}


@dataclass
class TensorSpec:
    name: str
    dtype: str  # safetensors dtype tag
    shape: tuple

    @property
    def nbytes(self) -> int:
        return DTYPE_BYTES[self.dtype] * math.prod(self.shape)


@dataclass
class ObjectSpec:
    kind: str                 # memt | kv | lora | file
    label: str
    tensors: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    history_len: int | None = None
    src_path: str | None = None  # for kind == "file"


def _scaled(n: int, scale: float) -> int:
    return max(1, int(round(n * scale)))


def memt_spec(scale: float = 1.0, layers=None, shape=(1, 16, 256, 256)) -> ObjectSpec:
    layers = layers if layers is not None else list(range(3, 40, 4))  # 3,7,...,39 -> 10 layers
    shp = tuple(shape[:-1]) + (_scaled(shape[-1], scale),)
    ts = [TensorSpec(f"memory.{l}", "F32", shp) for l in layers]
    return ObjectSpec("memt", "memt", ts, {"format": "memt-memory-v1"})


def kv_spec(T: int, scale: float = 1.0, attn_layers: int = 10, kv_heads: int = 2,
            head_dim: int = 256, gdn_layers: int = 30, gdn_heads: int = 32, gdn_k: int = 128,
            gdn_v: int = 128, conv_dim: int = 8192, conv_k: int = 3) -> ObjectSpec:
    hd = _scaled(head_dim, scale)
    ts = []
    # Qwen3.6-35B-A3B hybrid layout: every 4th layer is full attention (3,7,...,39), rest GDN.
    attn_ids = [4 * i + 3 for i in range(attn_layers)]
    gdn_ids = [l for l in range(attn_layers + gdn_layers) if l not in attn_ids][:gdn_layers]
    for l in attn_ids:
        ts.append(TensorSpec(f"past_key_values.{l}.key", "BF16", (1, kv_heads, T, hd)))
        ts.append(TensorSpec(f"past_key_values.{l}.value", "BF16", (1, kv_heads, T, hd)))
    for l in gdn_ids:
        ts.append(TensorSpec(f"gdn.{l}.recurrent_state", "F32",
                             (1, gdn_heads, gdn_k, _scaled(gdn_v, scale))))
        ts.append(TensorSpec(f"gdn.{l}.conv_state", "BF16", (1, _scaled(conv_dim, scale), conv_k)))
    md = {"format": "fullctx-kv-v1", "model": "Qwen3.6-35B-A3B", "history_len": str(T)}
    return ObjectSpec("kv", f"kv@{T}", ts, md, history_len=T)


def lora_spec(scale: float = 1.0, mib: float = 172.5) -> ObjectSpec:
    n = int(mib * MiB * scale) // 2
    return ObjectSpec("lora", "lora", [TensorSpec("lora.blob", "BF16", (n,))],
                      {"format": "temp-lora-blob-v1"})


# --------------------------------------------------------------------------------------
# Synthetic payload (deterministic) and safetensors byte layout
# --------------------------------------------------------------------------------------

def _fill_pool(fill: str, seed: int, nbytes: int) -> np.ndarray:
    """Deterministic byte pool used to fill tensors.

    'normal': N(0,1) float32 values truncated to their BF16 upper halves -> realistic entropy for
              BF16 activations (compressibility similar to real KV, matters for FlashBlade inline
              compression). FP32 tensors are filled from the same pool reinterpreted, which is
              slightly more compressible than true FP32 noise; acceptable for I/O timing.
    'random': uniform random bytes (incompressible).
    'zeros' : all zeros (maximally compressible; for debugging only).
    """
    rng = np.random.default_rng(seed)
    if fill == "zeros":
        return np.zeros(nbytes, dtype=np.uint8)
    if fill == "random":
        return np.frombuffer(rng.bytes(nbytes), dtype=np.uint8).copy()
    if fill == "normal":
        f = rng.standard_normal(nbytes // 2, dtype=np.float32)
        return (f.view(np.uint32) >> 16).astype(np.uint16).view(np.uint8)
    raise ValueError(fill)


def build_header(spec: ObjectSpec) -> tuple[bytes, list[tuple[int, int]]]:
    hdr = {}
    if spec.metadata:
        hdr["__metadata__"] = {str(k): str(v) for k, v in spec.metadata.items()}
    offs = []
    off = 0
    for t in spec.tensors:
        hdr[t.name] = {"dtype": t.dtype, "shape": list(t.shape), "data_offsets": [off, off + t.nbytes]}
        offs.append((off, off + t.nbytes))
        off += t.nbytes
    hb = json.dumps(hdr, separators=(",", ":")).encode()
    hb += b" " * ((-len(hb)) % 8)  # safetensors pads header to 8 bytes with spaces
    return hb, offs


class Payload:
    """The object's complete .safetensors byte image held in an anonymous (page-aligned) mmap."""

    def __init__(self, spec: ObjectSpec, fill: str = "normal", pool_mib: int = 64, seed: int = 1234):
        self.spec = spec
        if spec.kind == "file":
            self._load_file(spec.src_path)
            return
        hb, offs = build_header(spec)
        data_start = 8 + len(hb)
        self.nbytes = data_start + (offs[-1][1] if offs else 0)
        self.buf = mmap.mmap(-1, max(self.nbytes, 1))
        mv = memoryview(self.buf)
        mv[0:8] = struct.pack("<Q", len(hb))
        mv[8:data_start] = hb
        self.data_start = data_start
        self.offsets = offs
        arr = np.frombuffer(self.buf, dtype=np.uint8, count=self.nbytes)
        pool = _fill_pool(fill, seed, pool_mib * MiB)
        P = len(pool)
        # Tile pool over each tensor with a per-tensor rotation so tensors are not byte-identical.
        for i, (a, b) in enumerate(offs):
            dst = arr[data_start + a: data_start + b]
            rot = (i * 7919 * 4096) % P
            pos = 0
            n = b - a
            while pos < n:
                k = min(n - pos, P - rot if pos == 0 else P)
                src_off = rot if pos == 0 else 0
                dst[pos:pos + k] = pool[src_off:src_off + k]
                pos += k
        del arr
        self.sha256 = None

    def _load_file(self, path: str):
        n = os.path.getsize(path)
        self.nbytes = n
        self.buf = mmap.mmap(-1, n)
        with open(path, "rb") as f:
            f.readinto(memoryview(self.buf))
        (hlen,) = struct.unpack("<Q", self.buf[:8])
        hdr = json.loads(bytes(self.buf[8:8 + hlen]))
        md = hdr.pop("__metadata__", {}) or {}
        self.spec.metadata = md
        self.spec.tensors = []
        self.offsets = []
        for name, v in sorted(hdr.items(), key=lambda kv: kv[1]["data_offsets"][0]):
            self.spec.tensors.append(TensorSpec(name, v["dtype"], tuple(v["shape"])))
            self.offsets.append(tuple(v["data_offsets"]))
        self.data_start = 8 + hlen
        self.sha256 = None

    def view(self) -> memoryview:
        return memoryview(self.buf)[: self.nbytes]

    def digest(self) -> str:
        if self.sha256 is None:
            self.sha256 = hashlib.sha256(self.view()).hexdigest()
        return self.sha256

    def tensor_dict(self) -> dict:
        """numpy views (zero-copy) for safetensors.numpy.save_file. BF16 -> uint16 (saved as U16)."""
        out = {}
        for t, (a, b) in zip(self.spec.tensors, self.offsets):
            dt = NP_DTYPE.get(t.dtype, np.uint8)
            out[t.name] = np.frombuffer(self.buf, dtype=dt, count=(b - a) // np.dtype(dt).itemsize,
                                        offset=self.data_start + a).reshape(t.shape)
        return out


# --------------------------------------------------------------------------------------
# Raw I/O primitives
# --------------------------------------------------------------------------------------

def aligned_buffer(nbytes: int, prefault: bool = True) -> mmap.mmap:
    """Anonymous mmap: page-aligned (satisfies O_DIRECT 4 KiB alignment). Rounded up + 1 page."""
    size = (nbytes + ALIGN - 1) // ALIGN * ALIGN + ALIGN
    m = mmap.mmap(-1, size)
    if prefault:
        # Touch every page so page-fault cost is not inside the timed region.
        a = np.frombuffer(m, dtype=np.uint8)
        a[::4096] = 0
        del a
    return m


def write_file(path: str, data: memoryview, chunk: int, do_fsync: bool = True) -> dict:
    t0 = time.perf_counter()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        n = len(data)
        pos = 0
        while pos < n:
            pos += os.write(fd, data[pos:min(n, pos + chunk)])
        t1 = time.perf_counter()
        if do_fsync:
            os.fsync(fd)
        t2 = time.perf_counter()
    finally:
        os.close(fd)
    t3 = time.perf_counter()
    return {"total_s": t3 - t0, "write_s": t1 - t0, "fsync_s": t2 - t1}


def read_buffered(path: str, buf, nbytes: int, chunk: int) -> int:
    fd = os.open(path, os.O_RDONLY)
    try:
        mv = memoryview(buf)
        pos = 0
        while pos < nbytes:
            r = os.preadv(fd, [mv[pos:min(nbytes, pos + chunk)]], pos)
            if r <= 0:
                break
            pos += r
        return pos
    finally:
        os.close(fd)


def read_direct(path: str, buf, nbytes: int, chunk: int, io_threads: int = 1) -> tuple[int, str]:
    """O_DIRECT read of the whole file into aligned buf. Returns (bytes_read, tail_mode).

    chunk must be a multiple of 4 KiB. The last request is aligned-offset/aligned-length and runs
    past EOF (short read). On EINVAL for that request, falls back to buffered pread of the tail.
    io_threads > 1 splits the file into contiguous aligned ranges read by threads (GIL released
    in preadv) to get more NFS RPCs in flight per session.
    """
    assert chunk % ALIGN == 0
    mv = memoryview(buf)
    alen = (nbytes + ALIGN - 1) // ALIGN * ALIGN
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    tail_mode = ["direct_short_read" if nbytes % ALIGN else "aligned"]

    def _range(lo: int, hi: int) -> int:
        got = 0
        pos = lo
        while pos < hi:
            end = min(hi, pos + chunk)
            try:
                r = os.preadv(fd, [mv[pos:end]], pos)
            except OSError as e:
                if e.errno == 22 and end == alen and nbytes % ALIGN:  # EINVAL on EOF-straddling req
                    tail_mode[0] = "buffered_fallback"
                    with open(path, "rb", buffering=0) as f:
                        f.seek(pos)
                        r = f.readinto(mv[pos:nbytes])
                else:
                    raise
            if r <= 0:
                break
            got += r
            pos += r
            if r < end - (pos - r) and (pos >= nbytes or pos % ALIGN):
                break  # EOF (expected for the tail) or unaligned short read (caller checks size)
        return got

    try:
        if io_threads <= 1:
            got = _range(0, alen)
        else:
            from concurrent.futures import ThreadPoolExecutor
            per = (alen + io_threads - 1) // io_threads
            per = (per + chunk - 1) // chunk * chunk
            ranges = [(lo, min(alen, lo + per)) for lo in range(0, alen, per)]
            with ThreadPoolExecutor(len(ranges)) as ex:
                got = sum(ex.map(lambda r: _range(*r), ranges))
    finally:
        os.close(fd)
    return min(got, nbytes), tail_mode[0]


# --------------------------------------------------------------------------------------
# Environment helpers
# --------------------------------------------------------------------------------------

def mount_info(path: str) -> dict:
    rp = os.path.realpath(path)
    best = None
    with open("/proc/mounts") as f:
        for line in f:
            dev, mnt, fstype, opts, *_ = line.split()
            mnt = mnt.replace("\\040", " ")
            if rp == mnt or rp.startswith(mnt.rstrip("/") + "/"):
                if best is None or len(mnt) > len(best["mountpoint"]):
                    best = {"device": dev, "mountpoint": mnt, "fstype": fstype, "options": opts}
    return best or {}


def free_bytes(path: str) -> int:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds")


def stats(xs: list[float]) -> dict:
    xs = sorted(xs)
    return {"median_s": statistics.median(xs), "min_s": xs[0], "max_s": xs[-1],
            "mean_s": statistics.fmean(xs)}


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def gbps(nbytes: int, s: float) -> float:
    return nbytes / s / 1e9 if s > 0 else float("nan")


# --------------------------------------------------------------------------------------
# Concurrency workers (fork)
# --------------------------------------------------------------------------------------

_G = {}  # inherited by forked workers: payload view etc.


def _conc_worker(idx, op, path, nbytes, chunk, io_threads, barrier, q):
    try:
        if op == "conc_read_cold":
            buf = aligned_buffer(nbytes)
            barrier.wait()
            t0 = time.monotonic()
            got, tail = read_direct(path, buf, nbytes, chunk, io_threads)
            t1 = time.monotonic()
            ok = got == nbytes
            if _G.get("verify"):
                ok = ok and hashlib.sha256(memoryview(buf)[:nbytes]).hexdigest() == _G["digest"]
            q.put({"idx": idx, "t0": t0, "t1": t1, "bytes": got, "ok": ok, "tail_mode": tail})
        elif op == "conc_write":
            data = _G["payload"].view()
            barrier.wait()
            t0 = time.monotonic()
            r = write_file(path, data, chunk)
            t1 = time.monotonic()
            q.put({"idx": idx, "t0": t0, "t1": t1, "bytes": nbytes, "ok": True,
                   "fsync_s": r["fsync_s"]})
    except Exception as e:  # report, never hang the barrier
        try:
            barrier.abort()
        except Exception:
            pass
        q.put({"idx": idx, "error": repr(e)})


def run_concurrent(op: str, paths: list[str], nbytes: int, chunk: int, io_threads: int,
                   timeout: float = 3600) -> dict:
    ctx = mp.get_context("fork")
    K = len(paths)
    barrier = ctx.Barrier(K)
    q = ctx.Queue()
    procs = [ctx.Process(target=_conc_worker, args=(i, op, p, nbytes, chunk, io_threads, barrier, q))
             for i, p in enumerate(paths)]
    for p in procs:
        p.start()
    res = [q.get(timeout=timeout) for _ in procs]
    for p in procs:
        p.join()
    errs = [r for r in res if "error" in r]
    if errs:
        raise RuntimeError(f"{op} worker errors: {errs[:3]}")
    t0 = min(r["t0"] for r in res)
    t1 = max(r["t1"] for r in res)
    lat = [r["t1"] - r["t0"] for r in res]
    tot = sum(r["bytes"] for r in res)
    return {"wall_s": t1 - t0, "agg_gbps": gbps(tot, t1 - t0), "lat_s": lat,
            "all_ok": all(r["ok"] for r in res),
            "tail_modes": sorted({r.get("tail_mode", "") for r in res} - {""})}


# --------------------------------------------------------------------------------------
# Benchmark driver
# --------------------------------------------------------------------------------------

class Bench:
    def __init__(self, args):
        self.a = args
        self.run_id = args.run_id or uuid.uuid4().hex[:12]
        self.host = socket.gethostname()
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        self.out = open(args.out, "a")
        self.chunk = args.chunk_mib * MiB
        try:
            import safetensors
            self.st_ver = safetensors.__version__
        except Exception:
            self.st_ver = None
        self.n_records = 0

    def emit(self, rec: dict):
        base = {"experiment": "e3_persist_io", "run_id": self.run_id, "host": self.host,
                "ts": now_iso(), "python": platform.python_version(), "numpy": np.__version__,
                "safetensors": self.st_ver, "chunk_mib": self.a.chunk_mib,
                "io_threads": self.a.io_threads, "fill": self.a.fill, "scale": self.a.scale,
                "quick": self.a.quick}
        base.update(rec)
        self.out.write(json.dumps(base) + "\n")
        self.out.flush()
        self.n_records += 1

    def log(self, *a):
        print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

    # ---- one object on one target ------------------------------------------------------
    def bench_object(self, tname: str, tdir: str, spec: ObjectSpec):
        a = self.a
        payload = Payload(spec, fill=a.fill)
        nb = payload.nbytes
        common = {"target": tname, "target_dir": tdir, "mount": mount_info(tdir),
                  "object": spec.kind, "label": spec.label, "history_len": spec.history_len,
                  "bytes": nb, "mib": nb / MiB, "n_tensors": len(spec.tensors),
                  "src_path": spec.src_path}
        digest = payload.digest() if a.verify else None
        reserve = a.reserve_gib * GiB
        prefix = f"e3_{self.run_id}_{spec.label.replace('@', '_')}"
        self.log(f"{tname}: {spec.label} {nb / MiB:.1f} MiB ({len(spec.tensors)} tensors)")

        if free_bytes(tdir) - reserve < 2 * nb:
            self.emit({**common, "op": "skipped", "reason": "insufficient free space",
                       "free_bytes": free_bytes(tdir)})
            return

        rbuf = aligned_buffer(nb)
        data = payload.view()
        ops = {k: [] for k in ("write", "read_buffered_first", "read_cached", "read_cold")}
        extra = {"write_fsync_s": [], "write_syscalls_s": []}
        tail_modes = set()
        verify_ok = True
        if "raw" in a.phases:
            for rep in range(a.reps):
                p = os.path.join(tdir, f"{prefix}_raw_{rep}.safetensors")
                try:
                    r = write_file(p, data, self.chunk)
                    ops["write"].append(r["total_s"])
                    extra["write_fsync_s"].append(r["fsync_s"])
                    extra["write_syscalls_s"].append(r["write_s"])
                    t0 = time.perf_counter()
                    got = read_buffered(p, rbuf, nb, self.chunk)
                    ops["read_buffered_first"].append(time.perf_counter() - t0)
                    assert got == nb, (got, nb)
                    # The first buffered read may have gone to the server (NFS close-to-open
                    # revalidation can invalidate our own freshly written pages); it has now
                    # populated the page cache, so this second read is a true page-cache read.
                    t0 = time.perf_counter()
                    got = read_buffered(p, rbuf, nb, self.chunk)
                    ops["read_cached"].append(time.perf_counter() - t0)
                    assert got == nb, (got, nb)
                    if a.verify:
                        verify_ok &= hashlib.sha256(memoryview(rbuf)[:nb]).hexdigest() == digest
                    if a.verify:  # make sure cold read really fills the buffer afresh
                        np.frombuffer(rbuf, dtype=np.uint8)[:nb] = 0
                    if a.cold_delay_s:
                        time.sleep(a.cold_delay_s)
                    t0 = time.perf_counter()
                    got, tm = read_direct(p, rbuf, nb, self.chunk, a.io_threads)
                    ops["read_cold"].append(time.perf_counter() - t0)
                    tail_modes.add(tm)
                    assert got == nb, (got, nb)
                    if a.verify:
                        verify_ok &= hashlib.sha256(memoryview(rbuf)[:nb]).hexdigest() == digest
                finally:
                    if os.path.exists(p) and not a.keep:
                        os.unlink(p)
            notes = {
                "write": "buffered write + fsync + close, fresh O_TRUNC file per rep",
                "read_buffered_first": "first buffered pread after write+close (what a naive loader "
                                       "gets); on NFS may or may not hit the client page cache "
                                       "depending on close-to-open attribute revalidation",
                "read_cached": "second buffered pread (after an untimed-for-this-op priming read); "
                               "served from client page cache",
                "read_cold": "O_DIRECT, client page cache bypassed; storage-side cache may be warm",
            }
            for op, xs in ops.items():
                rec = {**common, "op": op, "concurrency": 1, "reps": len(xs), "times_s": xs,
                       **stats(xs), "gbps_median": gbps(nb, statistics.median(xs)),
                       "gbps_best": gbps(nb, min(xs)), "gbps_worst": gbps(nb, max(xs)),
                       "mode": "direct" if op == "read_cold" else "buffered", "notes": notes[op],
                       "label_cache": {"read_cold": "client-cache-bypassed",
                                       "read_cached": "page-cache",
                                       "read_buffered_first": "indeterminate"}.get(op, "n/a")}
                if op == "write":
                    rec["fsync_s"] = extra["write_fsync_s"]
                    rec["write_syscalls_s"] = extra["write_syscalls_s"]
                if op == "read_cold":
                    rec["tail_mode"] = sorted(tail_modes)
                    rec["cold_delay_s"] = a.cold_delay_s
                if a.verify and op != "write":
                    rec["verified"] = verify_ok
                self.emit(rec)
            self.log(f"  raw   write {gbps(nb, statistics.median(ops['write'])):.2f} GB/s  "
                     f"buf1st {gbps(nb, statistics.median(ops['read_buffered_first'])):.2f}  "
                     f"cached {gbps(nb, statistics.median(ops['read_cached'])):.2f}  "
                     f"cold {gbps(nb, statistics.median(ops['read_cold'])):.2f}  verify={verify_ok}")

        if "st" in a.phases:
            self.bench_safetensors(tdir, prefix, payload, rbuf, common)

        if "conc" in a.phases:
            for K in a.concurrency:
                self.bench_concurrency(tdir, prefix, payload, K, common)
        del rbuf

    def bench_safetensors(self, tdir, prefix, payload, rbuf, common):
        a = self.a
        from safetensors.numpy import save_file, load_file, load
        tensors = payload.tensor_dict()
        md = {str(k): str(v) for k, v in payload.spec.metadata.items()} or None
        save_t, fsync_t, first_t, load_t, cold_t, sizes = [], [], [], [], [], []
        ok = True
        for rep in range(a.reps):
            p = os.path.join(tdir, f"{prefix}_st_{rep}.safetensors")
            try:
                t0 = time.perf_counter()
                save_file(tensors, p, metadata=md)
                t1 = time.perf_counter()
                fd = os.open(p, os.O_RDONLY)
                os.fsync(fd)
                os.close(fd)
                t2 = time.perf_counter()
                save_t.append(t2 - t0)
                fsync_t.append(t2 - t1)
                n = os.path.getsize(p)
                sizes.append(n)
                t0 = time.perf_counter()
                d = load_file(p)
                first_t.append(time.perf_counter() - t0)
                del d
                t0 = time.perf_counter()
                d = load_file(p)
                load_t.append(time.perf_counter() - t0)
                if a.verify:
                    k0 = next(iter(tensors))
                    ok &= len(d) == len(tensors) and np.array_equal(d[k0], tensors[k0])
                del d
                t0 = time.perf_counter()
                got, _ = read_direct(p, rbuf, n, self.chunk, a.io_threads)
                d = load(bytes(memoryview(rbuf)[:got]))
                cold_t.append(time.perf_counter() - t0)
                del d
            finally:
                if os.path.exists(p) and not a.keep:
                    os.unlink(p)
        n = sizes[0]
        notes = {
            "st_save": "safetensors.numpy.save_file (buffered, library-internal) + explicit fsync",
            "st_load_first": "first safetensors.numpy.load_file after save (cache state "
                             "indeterminate on NFS, see read_buffered_first)",
            "st_load_cached": "second safetensors.numpy.load_file (page cache warm)",
            "st_load_cold": "O_DIRECT read + bytes() copy + safetensors.numpy.load(bytes)",
        }
        for op, xs in (("st_save", save_t), ("st_load_first", first_t), ("st_load_cached", load_t),
                       ("st_load_cold", cold_t)):
            rec = {**common, "op": op, "bytes": n, "mib": n / MiB, "concurrency": 1,
                   "reps": len(xs), "times_s": xs, **stats(xs),
                   "gbps_median": gbps(n, statistics.median(xs)), "gbps_best": gbps(n, min(xs)),
                   "gbps_worst": gbps(n, max(xs)), "notes": notes[op],
                   "mode": "direct" if op == "st_load_cold" else "buffered"}
            if op == "st_save":
                rec["fsync_s"] = fsync_t
            if a.verify and op == "st_load_cached":
                rec["verified"] = ok
            self.emit(rec)
        self.log(f"  st    save {gbps(n, statistics.median(save_t)):.2f} GB/s  "
                 f"load_cached {gbps(n, statistics.median(load_t)):.2f}  "
                 f"load_cold {gbps(n, statistics.median(cold_t)):.2f}  verify={ok}")

    def bench_concurrency(self, tdir, prefix, payload, K, common):
        a = self.a
        nb = payload.nbytes
        need = K * nb
        if free_bytes(tdir) - a.reserve_gib * GiB < need:
            self.emit({**common, "op": "skipped", "concurrency": K,
                       "reason": f"needs {need / GiB:.1f} GiB, free {free_bytes(tdir) / GiB:.1f} GiB "
                                 f"minus reserve {a.reserve_gib} GiB"})
            self.log(f"  conc K={K}: SKIPPED (free space)")
            return
        _G["payload"] = payload
        _G["verify"] = a.verify
        _G["digest"] = payload.digest() if a.verify else None
        paths = [os.path.join(tdir, f"{prefix}_conc{K}_s{i}.safetensors") for i in range(K)]
        wr, rd = [], []
        try:
            for rep in range(a.reps):
                if a.conc_write_reps is None or rep < a.conc_write_reps:
                    for p in paths:  # fresh files, like the single-session write op
                        if os.path.exists(p):
                            os.unlink(p)
                    wr.append(run_concurrent("conc_write", paths, nb, self.chunk, a.io_threads))
                # else: re-read the files written in an earlier rep
                if a.cold_delay_s:
                    time.sleep(a.cold_delay_s)
                rd.append(run_concurrent("conc_read_cold", paths, nb, self.chunk, a.io_threads))
        finally:
            if not a.keep:
                for p in paths:
                    if os.path.exists(p):
                        os.unlink(p)
        for op, rs in (("conc_write", wr), ("conc_read_cold", rd)):
            if not rs:
                continue
            lat = [x for r in rs for x in r["lat_s"]]
            agg = [r["agg_gbps"] for r in rs]
            walls = [r["wall_s"] for r in rs]
            rec = {**common, "op": op, "concurrency": K, "reps": len(rs),
                   "bytes_per_session": nb, "bytes_total": nb * K,
                   "wall_s": walls, "agg_gbps": agg, "agg_gbps_median": statistics.median(agg),
                   "agg_gbps_min": min(agg), "agg_gbps_max": max(agg),
                   "lat_s": lat, "lat_p50_s": pct(lat, 0.5), "lat_p95_s": pct(lat, 0.95),
                   "lat_max_s": max(lat), "lat_min_s": min(lat),
                   "per_session_gbps_p50": gbps(nb, pct(lat, 0.5)),
                   "mode": "direct" if op == "conc_read_cold" else "buffered",
                   "workers": "processes(fork)+barrier, CLOCK_MONOTONIC"}
            if op == "conc_read_cold":
                rec["verified"] = all(r["all_ok"] for r in rs) if a.verify else None
                rec["tail_mode"] = sorted({m for r in rs for m in r["tail_modes"]})
            self.emit(rec)
            self.log(f"  conc  {op:14s} K={K:<3d} agg {statistics.median(agg):.2f} GB/s  "
                     f"p50 {pct(lat, .5) * 1e3:.1f} ms  p95 {pct(lat, .95) * 1e3:.1f} ms")

    def run(self):
        a = self.a
        specs = []
        for o in a.objects:
            if o == "memt":
                specs.append(memt_spec(a.scale))
            elif o == "lora":
                specs.append(lora_spec(a.scale))
            elif o == "kv":
                specs += [kv_spec(T, a.scale) for T in a.lengths]
            elif o.startswith("file:"):
                pth = o[5:]
                specs.append(ObjectSpec("file", os.path.basename(pth), src_path=pth))
            else:
                raise SystemExit(f"unknown object {o}")
        if a.plan:
            return self.print_plan(specs)
        self.log(f"run_id={self.run_id} out={a.out}")
        for tname in a.targets:
            tdir = a.target_dirs[tname]
            os.makedirs(tdir, exist_ok=True)
            self.log(f"== target {tname}: {tdir} ({mount_info(tdir).get('fstype')})")
            for spec in specs:
                self.bench_object(tname, tdir, spec)
        self.out.close()
        self.log(f"wrote {self.n_records} records to {a.out}")

    def print_plan(self, specs):
        a = self.a
        print(f"{'object':<12}{'MiB':>10}  raw+st bytes moved/target   conc bytes moved/target  "
              f"peak on-disk")
        tot = 0
        for s in specs:
            nb = _spec_bytes(s)
            single = a.reps * nb * (3 + 3) if {"raw", "st"} <= set(a.phases) else a.reps * nb * 3
            cw = min(a.reps, a.conc_write_reps) if a.conc_write_reps is not None else a.reps
            conc = sum(K * nb * (a.reps + cw) for K in a.concurrency) if "conc" in a.phases else 0
            peak = max([nb] + [K * nb for K in a.concurrency])
            tot += single + conc
            print(f"{s.label:<12}{nb / MiB:>10.1f}  {single / GiB:>14.1f} GiB  {conc / GiB:>22.1f} GiB"
                  f"  {peak / GiB:>10.1f} GiB")
        print(f"total bytes moved per target: {tot / GiB:.1f} GiB  x {len(a.targets)} targets")


def _spec_bytes(spec: ObjectSpec) -> int:
    if spec.kind == "file":
        return os.path.getsize(spec.src_path)
    hb, offs = build_header(spec)
    return 8 + len(hb) + (offs[-1][1] if offs else 0)


# --------------------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------------------

def summarize(path: str, run_ids: list[str] | None = None):
    recs = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"<!-- skipping malformed line {i}: {e} -->")
    if run_ids:
        recs = [r for r in recs if r.get("run_id") in run_ids]
    if not recs:
        print("no records")
        return
    # keep the latest record per key
    latest = {}
    for r in recs:
        key = (r["target"], r["label"], r["op"], r.get("concurrency", 1))
        latest[key] = r
    runs = sorted({r["run_id"] for r in recs})
    print(f"# E3 persistence summary\n\nsource: `{path}`  runs: {', '.join(runs)}  "
          f"host: {recs[-1]['host']}  quick={recs[-1].get('quick')} scale={recs[-1].get('scale')}\n")
    print("GB/s = 1e9 B/s, median over reps (raw times, best/worst are in the JSONL). cold = O_DIRECT "
          "(client page cache bypassed); buf-1st = first buffered read after write (cache state "
          "indeterminate on NFS); cached = second buffered read (page cache).\n")

    def order(lbl):
        return (0, 0) if lbl == "memt" else (2, 0) if lbl == "lora" else \
            (1, int(lbl.split("@")[1])) if lbl.startswith("kv@") else (3, 0)

    targets = [t for t in ("rdma", "tcp", "local") if any(k[0] == t for k in latest)] + \
        sorted({k[0] for k in latest} - {"rdma", "tcp", "local"})
    labels = sorted({k[1] for k in latest}, key=order)

    def g(t, l, op, K=1):
        return latest.get((t, l, op, K))

    def fmt_bw(r):
        if not r:
            return "-"
        return f"{r['gbps_median']:.2f}"

    def fmt_ms(r):
        return "-" if not r else f"{r['median_s'] * 1e3:.1f}"

    print("## Single session (K=1)\n")
    print("| target | object | MiB | write GB/s | read cold GB/s | read buf-1st GB/s | read cached GB/s "
          "| cold restore ms | st_save GB/s | st_load 1st GB/s | st_load cached GB/s "
          "| st_load cold GB/s | st_load cold ms |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for t in targets:
        for l in labels:
            w = g(t, l, "write")
            if not w and not g(t, l, "st_save"):
                continue
            base = w or g(t, l, "st_save")
            print(f"| {t} | {l} | {base['mib']:.1f} | {fmt_bw(w)} | {fmt_bw(g(t, l, 'read_cold'))} | "
                  f"{fmt_bw(g(t, l, 'read_buffered_first'))} | "
                  f"{fmt_bw(g(t, l, 'read_cached'))} | {fmt_ms(g(t, l, 'read_cold'))} | "
                  f"{fmt_bw(g(t, l, 'st_save'))} | {fmt_bw(g(t, l, 'st_load_first'))} | "
                  f"{fmt_bw(g(t, l, 'st_load_cached'))} | "
                  f"{fmt_bw(g(t, l, 'st_load_cold'))} | {fmt_ms(g(t, l, 'st_load_cold'))} |")

    Ks = sorted({k[3] for k in latest if k[2] in ("conc_read_cold", "conc_write")})
    if Ks:
        print("\n## Concurrent sessions (distinct files, one process each)\n")
        print("| target | object | K | write agg GB/s | restore agg GB/s | restore p50 ms | "
              "restore p95 ms | restore max ms |")
        print("|---|---|---:|---:|---:|---:|---:|---:|")
        for t in targets:
            for l in labels:
                for K in Ks:
                    r = g(t, l, "conc_read_cold", K)
                    w = g(t, l, "conc_write", K)
                    if not r and not w:
                        continue
                    print(f"| {t} | {l} | {K} | {w['agg_gbps_median']:.2f} | " if w else
                          f"| {t} | {l} | {K} | - | ", end="")
                    if r:
                        print(f"{r['agg_gbps_median']:.2f} | {r['lat_p50_s'] * 1e3:.1f} | "
                              f"{r['lat_p95_s'] * 1e3:.1f} | {r['lat_max_s'] * 1e3:.1f} |")
                    else:
                        print("- | - | - | - |")

    print("\n## Derived (per target)\n")
    kmax = max(Ks) if Ks else None
    print(f"| target | object | bytes/session | sessions per TB (1e12 B) | restore s @K=1 (cold) "
          f"| agg restore GB/s @K={kmax} | sessions/s @K={kmax} |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for t in targets:
        for l in labels:
            r1 = g(t, l, "read_cold") or g(t, l, "conc_read_cold", 1)
            rk = g(t, l, "conc_read_cold", kmax) if kmax else None
            base = r1 or rk or g(t, l, "write")
            if not base:
                continue
            nb = base.get("bytes_per_session", base["bytes"])
            spt = 1e12 / nb
            r1s = (r1.get("median_s") or r1.get("lat_p50_s")) if r1 else None
            r1_txt = f"{r1s:.3f}" if r1s else "-"
            agg = rk["agg_gbps_median"] if rk else None
            agg_txt = f"{agg:.2f}" if agg else "-"
            sps_txt = f"{agg * 1e9 / nb:,.1f}" if agg else "-"
            print(f"| {t} | {l} | {nb:,} ({nb / MiB:.1f} MiB) | {spt:,.0f} | {r1_txt} | "
                  f"{agg_txt} | {sps_txt} |")
    sk = [r for r in recs if r["op"] == "skipped"]
    if sk:
        print("\nSkipped:")
        for r in sk:
            print(f"- {r['target']} {r['label']} K={r.get('concurrency', '-')}: {r['reason']}")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def _ints(s):
    return [int(x) for x in s.split(",") if x]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", default="rdma,tcp,local", help="comma list of rdma,tcp,local")
    ap.add_argument("--rdma-dir", default=DEFAULT_TARGETS["rdma"])
    ap.add_argument("--tcp-dir", default=DEFAULT_TARGETS["tcp"])
    ap.add_argument("--local-dir", default=DEFAULT_TARGETS["local"])
    ap.add_argument("--lengths", type=_ints, default=None, help="KV history lengths (tokens)")
    ap.add_argument("--objects", default=None,
                    help="comma list: memt,kv,lora,file:/path/x.safetensors (default memt,kv,lora)")
    ap.add_argument("--concurrency", type=_ints, default=None, help="K values, default 1,8,32")
    ap.add_argument("--reps", type=int, default=None, help="repetitions (default 5; quick 2)")
    ap.add_argument("--phases", default="raw,st,conc", help="subset of raw,st,conc")
    ap.add_argument("--scale", type=float, default=None,
                    help="shrink factor on the innermost tensor dim (1.0 = real sizes; quick 0.0625)")
    ap.add_argument("--quick", action="store_true",
                    help="validation: scale=1/16, lengths 512,2048, objects memt,kv, K 1,8, reps 2, "
                         "conc-write-reps 1, verify")
    ap.add_argument("--chunk-mib", type=int, default=64, help="bytes per read/write syscall (MiB)")
    ap.add_argument("--io-threads", type=int, default=1,
                    help="threads per session for O_DIRECT reads (contiguous ranges)")
    ap.add_argument("--fill", default="normal", choices=["normal", "random", "zeros"])
    ap.add_argument("--verify", action="store_true", help="sha256-check every read (not timed)")
    ap.add_argument("--cold-delay-s", type=float, default=0.0,
                    help="sleep between write and cold read (storage-side cache ageing)")
    ap.add_argument("--conc-write-reps", type=int, default=2,
                    help="(re)write the K files on only the first N reps; later reps re-read them "
                         "(default 2; pass a value >= --reps to rewrite every rep)")
    ap.add_argument("--reserve-gib", type=float, default=20.0,
                    help="skip a config if it would leave less than this free on the target")
    ap.add_argument("--keep", action="store_true", help="do not delete data files")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--plan", action="store_true", help="print sizes/bytes to move, do no I/O")
    ap.add_argument("--summary", nargs="?", const="__out__", default=None, metavar="JSONL",
                    help="print markdown summary of a JSONL (default --out) and exit")
    ap.add_argument("--summary-run", default=None, help="comma list of run_ids to include")
    a = ap.parse_args(argv)
    q = a.quick
    a.targets = [t for t in a.targets.split(",") if t]
    a.target_dirs = {"rdma": a.rdma_dir, "tcp": a.tcp_dir, "local": a.local_dir}
    for t in a.targets:
        if t not in a.target_dirs:
            ap.error(f"unknown target {t}")
    a.lengths = a.lengths or ([512, 2048] if q else DEFAULT_LENGTHS)
    a.objects = (a.objects or ("memt,kv" if q else "memt,kv,lora")).split(",")
    a.concurrency = a.concurrency or ([1, 8] if q else [1, 8, 32])
    a.reps = a.reps or (2 if q else 5)
    a.scale = a.scale if a.scale is not None else (0.0625 if q else 1.0)
    if q and "--conc-write-reps" not in " ".join(argv if argv is not None else sys.argv):
        a.conc_write_reps = 1
    a.verify = a.verify or q
    a.phases = set(a.phases.split(","))
    if a.chunk_mib * MiB % ALIGN:
        ap.error("chunk must be 4 KiB aligned")
    return a


def main(argv=None):
    a = parse_args(argv)
    if a.summary is not None:
        summarize(a.out if a.summary == "__out__" else a.summary,
                  a.summary_run.split(",") if a.summary_run else None)
        return
    Bench(a).run()


if __name__ == "__main__":
    main()
