#!/usr/bin/env python
"""RotorQuant-family rotation tests — CPU-only, no GPU required.

Run with the experiment venv (the code under test is the INSTALLED copy in
site-packages; run `bash kvarn/install.sh` after editing kvarn/files):

    /home/dewa/.venv-vllm-0.27.1-experiment/bin/python kvarn/test_rotorquant.py

Covers, for KVARN_ROTATION in {hadamard, planar, iso} at d in {128, 256, 512}:
  1. orthonormality of R (R Rᵀ = I, det = +1)
  2. un-rotation identity: (x @ R) @ Rᵀ = x  (the property the output
     un-rotation and QK-invariance rely on)
  3. QK invariance in FP16, the way the driver actually computes it
     (GEMM vs GEMM): rotated attention scores match unrotated within
     fp16-GEMM noise — compared against a no-rotation fp16 baseline so the
     rotation adds no EXTRA error
  4. full KVarN tile pipeline (rotate → Sinkhorn → asymmetric RTN → dequant →
     un-rotate) round-trip error vs the Hadamard baseline, on the same random
     tiles — the quality question Phase 1 answers
  5. determinism (seeded builders reproducible) and default-behaviour
     invariance (hadamard bit-identical to the historical Sylvester H)
"""

from __future__ import annotations

import math
import os
import sys

import torch

from vllm.model_executor.layers.quantization.kvarn.rotor import (
    _iso_matrix,
    _iso_matrix_from_quats,
    _iso_quats,
    _planar_matrix,
    _sylvester_hadamard,
    make_rotation_pair,
    rotation_kind,
    rotation_seed,
)

DEVICES = ["cpu"]
DS = [128, 256, 512]
KINDS = ["hadamard", "planar", "iso"]
RNG = torch.Generator().manual_seed(1234)

FAILS = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global FAILS
    mark = "PASS" if cond else "FAIL"
    print(f"  {mark}  {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS += 1


# ──────────────────────────────────────────────────────────────────────────────
# 1. orthonormality
# ──────────────────────────────────────────────────────────────────────────────


def test_orthonormal() -> None:
    print("== 1. orthonormality (R Rᵀ = I, Rᵀ R = I, det ≈ +1)")
    for d in DS:
        for kind in KINDS:
            R, Rt = make_rotation_pair(d, kind, "cpu")
            err_rt = (R @ Rt - torch.eye(d)).abs().max().item()
            err_rrt = (R @ R.T - torch.eye(d)).abs().max().item()
            # fp64: the fp32 det of a 512×512 (pivots ~0.04) underflows to 0
            det = torch.linalg.det(R.double()).item()
            check(
                f"d={d} {kind:8s}",
                err_rt < 1e-5 and err_rrt < 1e-5 and abs(det - 1.0) < 1e-3,
                f"‖R Rᵀ−I‖∞={err_rt:.2e}  ‖R Rᵀ−I‖∞(re)={err_rrt:.2e}  det={det:.4f}",
            )
            # transpose is the true inverse (used by every un-rotation site)
            check(
                f"d={d} {kind:8s} Rt == R.T",
                torch.equal(Rt, R.T),
            )


# ──────────────────────────────────────────────────────────────────────────────
# 2. un-rotation identity
# ──────────────────────────────────────────────────────────────────────────────


def test_unrotation_identity() -> None:
    print("== 2. un-rotation identity: (x @ R) @ Rᵀ == x  (fp32)")
    for d in DS:
        for kind in KINDS:
            R, Rt = make_rotation_pair(d, kind, "cpu")
            x = torch.randn(64, d, generator=RNG)
            err = ((x @ R) @ Rt - x).abs().max().item()
            check(f"d={d} {kind:8s}", err < 1e-4, f"max abs err {err:.2e}")


# ──────────────────────────────────────────────────────────────────────────────
# 3. QK invariance in fp16 (the driver's actual arithmetic)
# ──────────────────────────────────────────────────────────────────────────────


def test_qk_invariance_fp16() -> None:
    print("== 3. QK invariance in fp16 (driver-style GEMMs)")
    print("     (rotated error must not exceed the no-rotation fp16 baseline)")
    n = 128
    for d in DS:
        q = torch.randn(n, d, generator=RNG)
        k = torch.randn(n, d, generator=RNG)
        ref = (q @ k.T).abs()  # fp32 reference
        base_err = (q.half() @ k.half().T).float().sub(ref).abs().max().item()
        for kind in KINDS:
            R, _ = make_rotation_pair(d, kind, "cpu")
            R16 = R.half()
            # same shapes/ops as kvarn_decode_attention: row @ R, then
            # (q·R)(k·R)ᵀ via two GEMMs
            qr = (q.half() @ R16)
            kr = (k.half() @ R16)
            scores = (qr @ kr.T).float()
            err = (scores - ref).abs().max().item()
            rel = err / max(base_err, 1e-8)
            check(
                f"d={d} {kind:8s}",
                rel <= 2.0,
                f"rot err {err:.2e} vs fp16 baseline {base_err:.2e} ({rel:.2f}x)",
            )


# ──────────────────────────────────────────────────────────────────────────────
# 4. KVarN tile pipeline round-trip (rotate → Sinkhorn → RTN → dequant → Rᵀ)
# ──────────────────────────────────────────────────────────────────────────────


def _kvarn_tile_roundtrip(
    x: torch.Tensor, R: torch.Tensor, bits: int, orient: str
) -> float:
    """One KVarN tile's full round-trip, the installed pipeline exactly:

    x [G, D] fp32 (row-vector orientation, as in do_kv_cache_update)
      → x_rot = x @ R                                   (rotate on store)
      → K-view [D, G] / V-view [G, D]                   (KIVI orientation)
      → variance_normalize_batched (Sinkhorn, 16 iters)
      → kvarn_store_tile_{k,v}_batch_from_sinkhorn      (asymmetric RTN)
      → kvarn_dequant_tile_{k,v}                        (decode-kernel math)
      → @ Rᵀ                                           (un-rotate)
    returns relative L2 error vs the original tile.
    """
    from vllm.model_executor.layers.quantization.kvarn.sinkhorn import (
        variance_normalize_batched,
    )
    from vllm.v1.attention.ops.kvarn_decode import (
        kvarn_dequant_tile_k,
        kvarn_dequant_tile_v,
    )
    from vllm.v1.attention.ops.kvarn_store import (
        kvarn_store_tile_k_batch_from_sinkhorn,
        kvarn_store_tile_v_batch_from_sinkhorn,
    )

    G, D = x.shape
    x_rot = x @ R  # [G, D]
    if orient == "K":
        tile = x_rot.T.unsqueeze(0).contiguous()  # [1, D, G]
        bal, sc, sr = variance_normalize_batched(tile, iterations=16)
        out = kvarn_store_tile_k_batch_from_sinkhorn(
            bal, sc.squeeze(1), sr.squeeze(2), bits=bits)
        K_rot_DG = kvarn_dequant_tile_k(
            out["q_packed_uint8"][0], out["s_col_K"][0], out["zp_K"][0],
            out["s_row_K"][0], group=G, bits=bits)           # [D, G] rotated
        xhat = K_rot_DG.T @ R.T                              # [G, D]
    else:
        tile = x_rot.unsqueeze(0).contiguous()               # [1, G, D]
        bal, sc, sr = variance_normalize_batched(tile, iterations=16)
        out = kvarn_store_tile_v_batch_from_sinkhorn(
            bal, sc.squeeze(1), sr.squeeze(2), bits=bits)
        V_rot_GD = kvarn_dequant_tile_v(
            out["q_packed_uint8"][0], out["s_col_V"][0], out["s_row_V"][0],
            out["zp_V"][0], head_dim=D, bits=bits)           # [G, D] rotated
        xhat = V_rot_GD @ R.T

    rel = (xhat - x).norm() / x.norm().clamp_min(1e-12)
    return rel.item()


def test_kvarn_tile_roundtrip() -> None:
    print("== 4. KVarN tile pipeline round-trip (rotate→Sinkhorn→RTN→dequant→Rᵀ)")
    print("     relative L2 error, same random tiles across families")
    G, D = 128, 256  # the k4v2 preset on Qwen3.8-27B
    print("     K (4-bit, [D,G] KIVI orientation) / V (2-bit, [G,D])")
    for bits, orient in ((4, "K"), (2, "V")):
        # one "tile": 128 random vectors in the row-vector frame
        x = torch.randn(G, D, generator=RNG)
        # give it the structure real KV has: one heavy channel (an outlier)
        # plus per-vector scale variation
        x[:, 3] *= 6.0
        x *= torch.exp(torch.randn(G, 1, generator=RNG) * 0.5)
        print(f"     orient={orient} bits={bits}  (rel. L2 error, lower = better)")
        for kind in KINDS:
            R, _ = make_rotation_pair(D, kind, "cpu")
            err = _kvarn_tile_roundtrip(x, R, bits, orient)
            print(f"       {kind:8s} {err * 100:8.3f} %")
        # the assertion: planar/iso within 2x of the hadamard baseline
        # (the families are orthogonal, so RTN behaves the same up to how
        # well the rotation spreads each tile's variance)
        errs = {}
        for kind in KINDS:
            R, _ = make_rotation_pair(D, kind, "cpu")
            errs[kind] = _kvarn_tile_roundtrip(x, R, bits, orient)
        base = errs["hadamard"]
        for kind in ("planar", "iso"):
            check(
                f"orient={orient} bits={bits} {kind} within 2x hadamard",
                errs[kind] <= max(2.0 * base, 0.05),
                f"{kind} {errs[kind]*100:.3f}% vs hadamard {base*100:.3f}%",
            )


# ──────────────────────────────────────────────────────────────────────────────
# 5. iso = true quaternion LEFT-multiplication (published IsoQuant construction)
# ──────────────────────────────────────────────────────────────────────────────


def _ham_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Independent Hamilton product, the reference's 16-FMA form
    ([w, x, y, z]; e1·e2 = e3). Deliberately a different code path from
    _iso_matrix so the two can be cross-checked."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def test_iso_quaternion_fidelity() -> None:
    print("== 5. iso matrix == true quaternion left-multiplication (IsoQuant)")
    print("     (v @ R must equal the Hamilton product q·v in KVarN's")
    print("      row-vector convention; and R(a)@R(b) == R(b·a) — catches")
    print("      sign-flip / transpose bugs in the 4×4 table that")
    print("      orthonormality alone cannot catch)")
    n = 64
    for d in (128, 256):
        # the SAME seed-42 quaternions that build R (R is fixed, not random here)
        qL = _iso_quats(d, 42)
        v = torch.randn(n, d, generator=RNG)
        R = _iso_matrix(d, 42)

        # (a) action fidelity, KVarN's actual data flow: rows of x @ R
        #     must equal q·v computed by an independent 16-FMA product.
        #     R is (d//4)² blocks of 4×4; keep only the (i,i) block diagonal.
        #     einsum: vq (n, i, c-component) · block[i] (row=c, col=a) → out (n,i,a)
        Rb = R.reshape(d // 4, 4, d // 4, 4)  # Rb[i,a,j,c] = R[4i+a, 4j+c]
        Rdiag = torch.diagonal(Rb, dim1=0, dim2=2).permute(2, 0, 1)  # (i, row, col)
        vq = v.reshape(n, d // 4, 4)
        out_m = torch.einsum("nic,ica->nia", vq, Rdiag)  # row-vector matmul
        out_h = _ham_mul(qL, vq)  # (n, d//4, 4)
        err = (out_m - out_h).abs().max().item()
        check(f"d={d} v @ R == q·v (independent product)", err < 1e-5,
              f"max abs err {err:.2e}")

        # (b) group closure, row convention: applying R(a) then R(b) to a row
        #     vector composes the maps L_b∘L_a = L_{b·a}, so the matrix
        #     products run in REVERSED Hamilton order: R(a) @ R(b) = R(b·a).
        q42 = _iso_quats(d, 42)
        qa = torch.randn(d // 4, 4, generator=RNG)
        qa = qa / qa.norm(dim=-1, keepdim=True)
        qb_a = _ham_mul(q42, qa)  # independent Hamilton product, reversed order
        Rqa = _iso_matrix_from_quats(d, qa)
        Rq42 = _iso_matrix_from_quats(d, q42)
        Rqba = _iso_matrix_from_quats(d, qb_a)
        err2 = (Rqa @ Rq42 - Rqba).abs().max().item()
        check(f"d={d} R(a)@R(b) == R(b·a) (closure, row order)", err2 < 1e-4,
              f"max abs err {err2:.2e}")


# ──────────────────────────────────────────────────────────────────────────────
# 6. determinism + default-behaviour invariance
# ──────────────────────────────────────────────────────────────────────────────


def test_determinism_and_defaults() -> None:
    print("== 5. determinism + default-behaviour invariance")
    for d in (256,):
        p1, p2 = _planar_matrix(d, 42), _planar_matrix(d, 42)
        i1, i2 = _iso_matrix(d, 42), _iso_matrix(d, 42)
        check("planar(seed=42) reproducible", torch.equal(p1, p2))
        check("iso(seed=42) reproducible", torch.equal(i1, i2))
        check("different seeds differ", not torch.equal(
            _planar_matrix(d, 42), _planar_matrix(d, 43)))

        h_new, _ = make_rotation_pair(d, "hadamard", "cpu")
        h_old = _sylvester_hadamard(d, torch.device("cpu"))
        check("hadamard pair == historical Sylvester (bit-identical)",
              torch.equal(h_new, h_old))

    # env dispatch
    old = os.environ.get("KVARN_ROTATION")
    try:
        os.environ.pop("KVARN_ROTATION", None)
        check("default kind is hadamard", rotation_kind() == "hadamard")
        os.environ["KVARN_ROTATION"] = "planar"
        check("KVARN_ROTATION=planar", rotation_kind() == "planar")
        os.environ["KVARN_ROTATION"] = "ISO"
        check("KVARN_ROTATION is case-insensitive", rotation_kind() == "iso")
        os.environ["KVARN_ROTATION"] = "nonsense"
        try:
            rotation_kind()
            check("invalid kind rejected", False)
        except ValueError:
            check("invalid kind rejected", True)
    finally:
        if old is None:
            os.environ.pop("KVARN_ROTATION", None)
        else:
            os.environ["KVARN_ROTATION"] = old
    check("rotation_seed default 42", rotation_seed() == 42)


def main() -> None:
    print(f"torch {torch.__version__} on {DEVICES}")
    test_orthonormal()
    test_unrotation_identity()
    test_qk_invariance_fp16()
    test_iso_quaternion_fidelity()
    test_kvarn_tile_roundtrip()
    test_determinism_and_defaults()
    print()
    if FAILS:
        print(f"test_rotorquant: {FAILS} FAILURE(S)")
        sys.exit(1)
    print("test_rotorquant: all PASS")


if __name__ == "__main__":
    main()
