# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RotorQuant-family block-diagonal rotations for KVarN.

KVarN's default rotation is the Sylvester Hadamard — a dense d×d orthogonal
matrix applied to q, k, v by GEMM. This module offers the *block-diagonal*
alternatives from the RotorQuant rotation family (scrya.com/rotorquant.pdf,
March 2026; the production PlanarQuant / IsoQuant variants):

  - ``hadamard`` (default, unchanged): dense Sylvester H / sqrt(d). Symmetric,
    so the inverse is itself — the existing code path, byte-identical.
  - ``planar``: one independent 2D Givens rotation per adjacent element pair —
    d/2 blocks of [[c, -s], [s, c]], random angles (seeded).
  - ``iso``: one independent 4D unit-quaternion left-multiplication per group
    of four — d/4 blocks of the 4×4 isoclinic rotation M(q); the 3-DOF
    "fast" mode of the IsoQuant construction. (Experimental on
    Qwen3.8-27B K4V2: the seed-42 instance scores below hadamard/planar in
    the GSM8K A/B for both this and the mirror orientation — see
    kvarn/README.md.)

Why this works without touching the rest of KVarN:

  - Any orthogonal R keeps the attention invariance: (q·R)(k·R)ᵀ = q·kᵀ and
    (Σᵢ pᵢ vᵢ·R)·Rᵀ = Σᵢ pᵢ vᵢ, so rotating q/k/v in and out-rotating the
    output is exactly what the Hadamard path does.
  - The per-tile Sinkhorn variance normalization and asymmetric RTN are
    orthogonal-invariant (they operate on scales/zeros in the rotated frame),
    so they need no change for any rotation family.
  - The matrices are FIXED (seeded, data-independent): identical across
    process restarts, which the KV cache requires — a flushed int4 tile is
    quantized in the rotated frame, so a different matrix would corrupt
    cached history.

One difference from the Hadamard: Givens / quaternion blocks are NOT
symmetric, so the inverse is the TRANSPOSE. Call sites that un-rotate must
use ``Rt = R.T`` (for the Hadamard, ``.T`` is a no-op, so the same code is
correct for all three families).

The reference sources (scrya-com/rotorquant, ParaMind2025/isoquant) are NOT
copied here — that code is unlicensed; only the constructions described in
their papers are re-implemented.
"""

from __future__ import annotations

import functools
import math

import torch

# ──────────────────────────────────────────────────────────────────────────────
# kind dispatch
# ──────────────────────────────────────────────────────────────────────────────

_ROTATION_KINDS = ("hadamard", "planar", "iso")
_DEFAULT_KIND = "hadamard"


def rotation_kind() -> str:
    """The active rotation family from the KVARN_ROTATION env var.

    ``hadamard`` (default, the historical KVarN behaviour), ``planar``
    (2D Givens blocks) or ``iso`` (4D quaternion blocks). Read at call time
    (cheap) so tests can flip it per call; in serving it is constant.
    """
    kind = os_environ_get("KVARN_ROTATION", _DEFAULT_KIND)
    kind = kind.strip().lower()
    if kind not in _ROTATION_KINDS:
        raise ValueError(
            f"KVARN_ROTATION={kind!r} — expected one of {_ROTATION_KINDS}"
        )
    return kind


def rotation_seed() -> int:
    """Seed for the random planar/iso parameters (the published reference
    uses 42; keep the default so our numbers are comparable to theirs)."""
    try:
        return int(os_environ_get("KVARN_ROTOR_SEED", "42"))
    except ValueError:
        raise ValueError("KVARN_ROTOR_SEED must be an integer")


def os_environ_get(name: str, default: str) -> str:
    import os

    return os.environ.get(name, default)


# ──────────────────────────────────────────────────────────────────────────────
# matrix builders (fp32, on the given device; cached per (d, kind, device))
# ──────────────────────────────────────────────────────────────────────────────


def _sylvester_hadamard(d: int, device: torch.device) -> torch.Tensor:
    """Sylvester Hadamard normalised to orthonormal rows — bit-identical to
    the historical KVarN ``_hadamard_cached`` construction."""
    H = torch.ones(1, 1)
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(d)).to(device).float()


def _planar_cs(d: int, seed: int) -> torch.Tensor:
    """(d//2, 2) fp32 [cos θ, sin θ] — one random angle per 2-element pair,
    the PlanarQuant construction (seeded, so fixed across restarts)."""
    if d % 2:
        raise ValueError(f"planar rotation needs even head dim, got {d}")
    gen = torch.Generator()
    gen.manual_seed(seed)
    angles = torch.rand(d // 2, generator=gen) * (2.0 * math.pi)
    return torch.stack([angles.cos(), angles.sin()], dim=-1)


def _iso_quats(d: int, seed: int) -> torch.Tensor:
    """(d//4, 4) fp32 random UNIT quaternions [w, x, y, z] (normalized
    Gaussian — the IsoQuant construction)."""
    if d % 4:
        raise ValueError(f"iso rotation needs head dim divisible by 4, got {d}")
    gen = torch.Generator()
    gen.manual_seed(seed)
    q = torch.randn(d // 4, 4, generator=gen)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _planar_matrix(d: int, seed: int) -> torch.Tensor:
    """d×d block-diagonal Givens matrix: per pair [[c, -s], [s, c]]."""
    cs = _planar_cs(d, seed)
    c, s = cs[:, 0], cs[:, 1]
    R = torch.zeros(d, d)
    idx = torch.arange(0, d, 2)
    R[idx, idx] = c
    R[idx, idx + 1] = -s
    R[idx + 1, idx] = s
    R[idx + 1, idx + 1] = c
    return R


def _iso_matrix_from_quats(d: int, q: torch.Tensor) -> torch.Tensor:
    """d×d block-diagonal matrix from unit quaternions q [n, 4] (w, x, y, z).

    Row j of each 4×4 block is q·e_j (Hamilton left multiplication), so in
    KVarN's row-vector convention — ``x_rot = x @ R`` — the applied map is
    exactly the published IsoQuant ``T(v) = q·v``. (The column-convention
    matrix, whose rows are q·e_j transposed, would apply right-multiplication
    to row-vector data — the mirror family, not the published one.)
    """
    if d % 4:
        raise ValueError(f"iso rotation needs head dim divisible by 4, got {d}")
    if q.shape != (d // 4, 4):
        raise ValueError(f"expected {(d // 4, 4)} quaternions, got {tuple(q.shape)}")
    w, x, y, z = q.unbind(-1)
    R = torch.zeros(d, d)
    idx = torch.arange(0, d, 4)
    for off in range(4):
        for off2 in range(4):
            # rows q·e0 .. q·e3 = (q, q·i, q·j, q·k), e1·e2 = e3
            table = (
                (w, x, y, z),
                (-x, w, z, -y),
                (-y, -z, w, x),
                (-z, y, -x, w),
            )
            R[idx + off, idx + off2] = table[off][off2]
    return R


def _iso_matrix(d: int, seed: int) -> torch.Tensor:
    """d×d block-diagonal quaternion left-multiplication matrix.

    For a unit quaternion q = (w, x, y, z) the rows are the Hamilton products
    q·e_j (e1·e2 = e3):

        [[  w,   x,   y,   z],
         [ -x,   w,   z,  -y],
         [ -y,  -z,   w,   x],
         [ -z,   y,  -x,   w]]

    In KVarN's row-vector convention (x @ R) this applies T(v) = q·v — the
    published IsoQuant fast mode — with inverse Rᵀ (rows q̄·e_j).
    """
    return _iso_matrix_from_quats(d, _iso_quats(d, seed))


@functools.cache
def make_rotation_pair(
    d: int, kind: str, device_str: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """(R, Rt) fp32 d×d on the given device: the forward rotation and its
    inverse (transpose). For ``hadamard`` both refer to the same (symmetric)
    matrix, so there is no extra memory.

    Cached per (d, kind, device) — one build per deployment.
    """
    kind = (kind or _DEFAULT_KIND).strip().lower()
    if kind not in _ROTATION_KINDS:
        raise ValueError(f"unknown rotation kind {kind!r}")
    device = torch.device(device_str)
    if kind == "hadamard":
        R = _sylvester_hadamard(d, device)
        return R, R
    seed = rotation_seed()
    R = _planar_matrix(d, seed) if kind == "planar" else _iso_matrix(d, seed)
    R = R.to(device)
    return R, R.T


def rotation_matrices(d: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Convenience wrapper for the active KVARN_ROTATION family."""
    return make_rotation_pair(d, rotation_kind(), str(device))
