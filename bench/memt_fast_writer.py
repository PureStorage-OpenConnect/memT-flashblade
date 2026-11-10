"""Chunked-kernel memory *write* path for MemT-1.0-Nano (HF reference model).

The shipped `MemTMemory.forward(commit=True)` advances the FP32 memory state with a Python
loop over every token. Per layer, per head, with u_i = sqrt(beta_i) * L2Norm(k_i):

    M_i = (I - u_i u_i^T) Diag(exp(lambda_i)) M_{i-1} + u_i (omega_i * v_i)^T     (M is K x V)

This is exactly the GDN-2 recurrence implemented by flash-linear-attention's `chunk_gdn2`
(fla >= 0.6.0, MIT):

    S_t = (I - k_t (b_t * k_t)^T) Diag(exp(g_t)) S_{t-1} + k_t (w_t * v_t)^T       (S is K x V)

with the mapping  k := u,  b := 1,  g := lambda (natural-log decay, per key channel),
v := v (BF16, as the reference holds it),  w := sigmoid(W_omega h) (BF16, as the reference
holds it), q := u (the kernel's output o is discarded), initial_state := M (FP32 [B,H,K,V]).
Decay is applied before the erase in both, and the state layouts agree (K x V).

All kernel inputs are bit-exact copies of the values the reference loop consumes (u is
rounded to BF16 by the reference itself, v and sigmoid(w) are BF16 in the reference, the
reference multiplies them in FP32 and so does the kernel). The only differences are the
kernel's chunked (WY / TF32 tensor-core) arithmetic.

Usage:
    from memt_fast_writer import patch_model, unpatch_model
    patch_model(model)          # every MemTMemory uses the kernel write path
    state = model.commit_memory(ids, state, document=True)
    unpatch_model(model)
"""
from __future__ import annotations

import types

import torch
import torch.nn.functional as F

CHUNK = 64  # fla GDN-2 chunk size (hard-coded in the kernel)

_fla_chunk_gdn2 = None


def _kernel():
    global _fla_chunk_gdn2
    if _fla_chunk_gdn2 is None:
        from fla.ops.gdn2 import chunk_gdn2

        _fla_chunk_gdn2 = chunk_gdn2
    return _fla_chunk_gdn2


# --------------------------------------------------------------------------- write inputs
def write_inputs(module, hidden, projected=None):
    """Return (u, v, w_gate, log_decay) exactly as the reference loop sees them.

    u [B,L,H,K] BF16 (the reference computes key*sqrt_b in FP32, rounds to BF16, then .float()),
    v [B,L,H,V] BF16, w_gate [B,L,H,V] BF16 (= sigmoid(w)), log_decay [B,L,H,K] FP32
    (the reference uses exp(log_decay) as `decay`).
    `projected` may be the reference's F.linear(hidden, in_proj.weight[:3*width]) to reuse it.
    """
    batch, length, _ = hidden.shape
    dim, width, heads = module.head_dim, module.value_dim, module.num_heads
    shape = (batch, length, heads, dim)
    if projected is None:
        projected = F.linear(hidden, module.in_proj.weight[: 3 * width])
    key, value = (item.reshape(shape) for item in projected[..., width:].split(width, -1))
    key = module._normalize(key, True)
    b, w, f = F.linear(hidden, module.in_proj.weight[3 * width : -dim]).split(
        (width, width, dim), -1
    )
    sqrt_b = (0.5 * F.logsigmoid(b.float())).exp().to(b.dtype).reshape(shape)
    u = (key.float() * sqrt_b.float()).to(value.dtype)
    w_gate = w.sigmoid().reshape(shape)
    log_decay = -module.decay_scale_log.float().exp().reshape(heads, 1) * F.softplus(
        module.f_down(f).float() + module.decay_bias.float()
    ).reshape(shape)
    return u, value, w_gate, log_decay


def gdn2_state_update(u, v, w_gate, log_decay, state, *, input_dtype=torch.float32, pad=True):
    """Run the chunked GDN-2 kernel and return the final FP32 state [B,H,K,V].

    Padded tail positions are no-ops: u=0 (no erase, no write) and log_decay=0 (decay 1).
    input_dtype: torch.float32 (default) upcasts u/v/w (lossless) so the kernel's tl.dot run as
    TF32 on FP32 operands (~3x lower error than BF16 operands, +~0.5 ms per layer per 2048 tokens).
    None feeds the BF16 tensors directly (intermediates such as k*exp(g) are rounded to BF16).
    High-precision mode: export TRITON_F32_DEFAULT=tf32x3 for the process (measured at L=2048:
    4.8e-5 rel state error vs FP64 instead of 6e-4, 2.4 ms instead of 1.7 ms per layer, ~4 min
    first-call compile+autotune). It must be set before any Triton kernel compiles: the knob is
    read at compile time and is not part of Triton's in-process kernel cache key, so it cannot be
    toggled per call. It affects every FP32-operand tl.dot in the process (also the backbone's fla
    kernels where they do not pin input_precision).
    """
    chunk_gdn2 = _kernel()
    batch, length, heads, dim = u.shape
    if input_dtype is not None:
        u, v, w_gate = u.to(input_dtype), v.to(input_dtype), w_gate.to(input_dtype)
    g = log_decay.float()
    if pad and length % CHUNK:
        extra = CHUNK - length % CHUNK
        padding = (0, 0, 0, 0, 0, extra)
        u, v, w_gate, g = (F.pad(x, padding) for x in (u, v, w_gate, g))
    u, v, w_gate, g = (x.contiguous() for x in (u, v, w_gate, g))
    ones = torch.ones_like(u)
    h0 = state.to(device=u.device, dtype=torch.float32).contiguous()
    with torch.autocast(device_type=u.device.type, enabled=False):
        _, final = chunk_gdn2(
            q=u, k=u, v=v, g=g, b=ones, w=w_gate, scale=1.0,
            initial_state=h0, output_final_state=True,
        )
    return final


@torch.no_grad()
def fast_commit_state(module, hidden, state, **kw):
    """Kernel equivalent of the reference write loop: returns the new FP32 state."""
    batch = hidden.shape[0]
    if state is None:
        state = hidden.new_zeros((batch, module.num_heads, module.head_dim, module.head_dim),
                                 dtype=torch.float32)
    u, v, w_gate, log_decay = write_inputs(module, hidden)
    return gdn2_state_update(u, v, w_gate, log_decay, state, **kw)


# --------------------------------------------------------------------------- patched forward
def fast_forward(self, hidden, state=None, commit=False, *, write_state=None):
    """Drop-in MemTMemory.forward: identical read path, kernel write path."""
    if not commit:
        return self._memt_reference_forward(hidden, state, commit, write_state=write_state)
    batch, length, _ = hidden.shape
    dim, width = self.head_dim, self.value_dim
    shape = (batch, length, self.num_heads, dim)
    snapshot = (
        hidden.new_zeros((batch, self.num_heads, dim, dim), dtype=torch.float32)
        if state is None
        else state
    )
    # Read path: verbatim from the reference (commit=True branch).
    projected = F.linear(hidden, self.in_proj.weight[: 3 * width])
    query = self._normalize(projected[..., :width].reshape(shape), commit)
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        read = self.read_scale * torch.einsum(
            "blhk,bhkv->blhv", query.float(), snapshot.float()
        )
    gate = self.g_down(F.linear(hidden, self.in_proj.weight[-dim:])).reshape(shape)
    read = self.o_norm(read.to(gate.dtype), gate).reshape(batch, length, width)
    # Write path: chunked GDN-2 kernel.
    u, v, w_gate, log_decay = write_inputs(self, hidden, projected)
    start = snapshot if write_state is None else write_state
    updated = gdn2_state_update(u, v, w_gate, log_decay, start.float(),
                                **getattr(self, "_memt_fast_kw", {}))
    return read, updated


def _memory_modules(model):
    for module in model.modules():
        if type(module).__name__ == "MemTMemory":
            yield module


def patch_model(model, **kw):
    """Route every MemTMemory commit through the kernel. Returns the number patched."""
    count = 0
    for module in _memory_modules(model):
        patch_module(module, **kw)
        count += 1
    return count


def patch_module(module, **kw):
    if "_memt_reference_forward" not in module.__dict__:
        module._memt_reference_forward = module.forward  # bound reference method
    module._memt_fast_kw = kw
    module.forward = types.MethodType(fast_forward, module)
    return module


def unpatch_module(module):
    if "_memt_reference_forward" in module.__dict__:
        del module.__dict__["forward"]
        del module.__dict__["_memt_reference_forward"]
        module.__dict__.pop("_memt_fast_kw", None)
    return module


def unpatch_model(model):
    count = 0
    for module in _memory_modules(model):
        unpatch_module(module)
        count += 1
    return count


# --------------------------------------------------------------------------- reference helpers
@torch.no_grad()
def reference_commit_state(module, hidden, state, dtype=torch.float32):
    """The reference token loop (same arithmetic as MemTMemory.forward), optionally in FP64."""
    u, v, w_gate, log_decay = write_inputs(module, hidden)
    u = u.to(dtype)
    write_value = w_gate.float().to(dtype) * v.float().to(dtype)
    decay = log_decay.to(dtype).exp()
    updated = state.to(dtype)
    for index in range(u.shape[1]):
        ui = u[:, index]
        updated = updated * decay[:, index].unsqueeze(-1)
        erase = (ui.unsqueeze(-1) * updated).sum(-2)
        updated = updated + ui.unsqueeze(-1) * (write_value[:, index] - erase).unsqueeze(-2)
    return updated
