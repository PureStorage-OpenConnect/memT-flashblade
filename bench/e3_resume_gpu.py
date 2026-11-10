#!/usr/bin/env python3
"""E3 (part 2): file -> pinned host -> GPU restore timing.  *** STUB, NOT YET RUN ***

TODO(E3-gpu): enable once GPUs are free (other agents own them now). Then:
  1. Restore a state object (memt snapshot / full-context KV / LoRA) written by e3_persist_io.py
     (or by the real HF session API, E2) from a target dir into GPU memory, timing each stage:
       open -> O_DIRECT read into page-aligned host buffer -> cudaHostRegister'd (pinned) ->
       H2D copy (non_blocking, one stream) -> torch.cuda.synchronize()
     and alternatively a buffered/safetensors path (safetensors.torch.load_file(device="cuda")).
  2. Also time the reverse (save): D2H into pinned buffer -> write + fsync.
  3. Feed the restored tensors into the model's session API (E1/E2: commit_memory / KV inject)
     and measure TTFT including restore; that lives in the E1 harness, this script only times I/O.
  4. Emit JSONL like e3_persist_io.py (experiment="e3_resume_gpu") into results/e3_resume_gpu.jsonl.

Design notes:
  * torch is imported lazily (inside functions) so this module imports without torch.
  * Host buffer: page-aligned anonymous mmap from e3_persist_io.aligned_buffer (O_DIRECT needs
    4 KiB alignment); pin it in place with cudaHostRegister rather than allocating with
    torch.empty(pin_memory=True), whose alignment is an allocator detail.
  * GPU tensors are views on one flat device buffer at the safetensors data offsets, so H2D is
    a single large copy (best case) -- per-tensor copies can be timed with --per-tensor.
  * Overlap variant (later): chunked O_DIRECT reads into a double-buffered pinned ring with H2D
    of chunk i overlapping read of chunk i+1.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e3_persist_io as io  # noqa: E402  (numpy + stdlib only)


def _torch():
    import torch  # lazy: the project venv may not exist yet
    return torch


def parse_header(buf, nbytes: int) -> tuple[int, dict]:
    """Return (data_start, header dict) for a safetensors image held in buf."""
    (hlen,) = struct.unpack("<Q", bytes(memoryview(buf)[:8]))
    hdr = json.loads(bytes(memoryview(buf)[8:8 + hlen]))
    return 8 + hlen, hdr


def pin_host_buffer(buf) -> object:
    """cudaHostRegister an existing page-aligned buffer; returns a uint8 CPU tensor view (pinned)."""
    torch = _torch()
    t = torch.frombuffer(buf, dtype=torch.uint8)
    # TODO(E3-gpu): verify return code; unregister in release_host_buffer().
    torch.cuda.cudart().cudaHostRegister(t.data_ptr(), t.numel(), 0)
    return t


def release_host_buffer(t) -> None:
    torch = _torch()
    torch.cuda.cudart().cudaHostUnregister(t.data_ptr())


def restore_to_gpu(path: str, device: str = "cuda:0", direct: bool = True,
                   chunk_mib: int = 64, io_threads: int = 1, per_tensor: bool = False) -> dict:
    """Time file -> pinned host -> GPU. Returns a dict of stage timings (seconds).

    Stages: alloc_pin_s (not part of resume latency if buffers are pooled), read_s, h2d_s,
    total_s (read + h2d), plus bytes and GB/s.
    """
    torch = _torch()
    nbytes = os.path.getsize(path)
    t0 = time.perf_counter()
    host = io.aligned_buffer(nbytes)
    pinned = pin_host_buffer(host)
    dev = torch.empty(nbytes, dtype=torch.uint8, device=device)
    torch.cuda.synchronize(device)
    t1 = time.perf_counter()
    if direct:
        got, tail_mode = io.read_direct(path, host, nbytes, chunk_mib * io.MiB, io_threads)
    else:
        got, tail_mode = io.read_buffered(path, host, nbytes, chunk_mib * io.MiB), "buffered"
    t2 = time.perf_counter()
    if per_tensor:
        data_start, hdr = parse_header(host, nbytes)
        hdr.pop("__metadata__", None)
        for v in hdr.values():
            a, b = v["data_offsets"]
            dev[data_start + a:data_start + b].copy_(pinned[data_start + a:data_start + b],
                                                     non_blocking=True)
    else:
        dev[:got].copy_(pinned[:got], non_blocking=True)
    torch.cuda.synchronize(device)
    t3 = time.perf_counter()
    # TODO(E3-gpu): build typed views (dtype/shape per header; BF16 -> torch.bfloat16) on `dev`.
    release_host_buffer(pinned)
    return {"bytes": got, "tail_mode": tail_mode, "alloc_pin_s": t1 - t0, "read_s": t2 - t1,
            "h2d_s": t3 - t2, "total_s": t3 - t1,
            "read_gbps": io.gbps(got, t2 - t1), "h2d_gbps": io.gbps(got, t3 - t2),
            "total_gbps": io.gbps(got, t3 - t1)}


def restore_safetensors_torch(path: str, device: str = "cuda:0") -> dict:
    """Library path: safetensors.torch.load_file(path, device=...). TODO(E3-gpu)."""
    torch = _torch()
    from safetensors.torch import load_file
    t0 = time.perf_counter()
    d = load_file(path, device=device)
    torch.cuda.synchronize(device)
    return {"total_s": time.perf_counter() - t0, "n_tensors": len(d)}


def save_from_gpu(tensors: dict, path: str) -> dict:
    """D2H into pinned buffer -> write + fsync. TODO(E3-gpu): implement."""
    raise NotImplementedError("TODO(E3-gpu): D2H + write timing")


def main(argv=None):
    ap = argparse.ArgumentParser(description="E3 GPU resume timing (stub; requires free GPUs)")
    ap.add_argument("--path", required=True, help="safetensors state file on a target")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--buffered", action="store_true")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--i-own-the-gpu", action="store_true",
                    help="required: confirms no other job is using --device")
    a = ap.parse_args(argv)
    if not a.i_own_the_gpu:
        sys.exit("refusing to run: GPUs are owned by other jobs. TODO(E3-gpu) enable later "
                 "with --i-own-the-gpu.")
    for _ in range(a.reps):
        print(json.dumps(restore_to_gpu(a.path, a.device, direct=not a.buffered)))


if __name__ == "__main__":
    main()
