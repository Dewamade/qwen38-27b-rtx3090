# KVarN KV cache, ported to vLLM 0.27.1

[KVarN](https://github.com/huawei-csl/KVarN) (Huawei CSL, Apache-2.0) is a
KV-cache compression scheme — Hadamard rotation, iterative variance
normalization, 4-bit keys / 2-bit values per 128-token tile — shipped as a
native vLLM attention backend inside a fork of vLLM 0.23.0. This directory is
that backend ported onto the vLLM 0.27.1 this repo runs, dense (non-MLA) path
only, and tuned for the Qwen3.8-27B / RTX 3090 setup here.

What's in it:

- `files/vllm/...` — the KVarN modules (backend, Triton kernels, config,
  Sinkhorn reference), copied from KVarN and adapted to the 0.27.1 backend API
  (every change is marked `# port(0.27.1)`; upstream KVarN headers kept).
- `kvarn-0.27.1.patch` — the seven small hunks upstream vLLM needs to know the
  new `kvarn_*` cache dtypes (cache dtype literals, dtype map, backend registry
  + priority, a `KVQuantMode.KVARN`, the KV-cache spec branch in the attention
  layer, and the hybrid-model page alignment branch).
- `install.sh` — copies the modules into `venv/lib/python3.12/site-packages/vllm`
  and applies the patch (safe to re-run).

Port notes, for whoever bumps vLLM next:

- 0.27.1 calls `get_kv_cache_shape(..., cache_dtype_str="auto")` for specs
  whose `kv_quant_mode` is `NONE`; KVarN's shape depends on the preset, so the
  port adds `KVQuantMode.KVARN` and passes it through the (reused)
  `TQFullAttentionSpec`. Without that the engine dies at KV-cache init.
- The impl→builder wiring uses `get_layers_from_vllm_config` instead of
  KVarN's `attention.py` `impl.layer_name` hunk, and a small owner registry so
  the MTP draft layer isn't flushed by two builders.
- Pools are materialized during `profile_run` (forward with
  `attn_metadata=None`) so vLLM's memory profiler charges them correctly —
  no `gpu_worker.py` hunk needed.
- Per-token slot padding to a power of two (KVarN did it for Gemma-4's mixed
  head dims) is off by default here (`KVARN_POW2_SLOT=1` restores it): with
  head_dim 256 that is 840 B/token/layer instead of 1024 (fp8: 2048).
- The hybrid alignment makes the attention block 2048 tokens (page must match
  the 1.63 MB Gated-DeltaNet page); vLLM splits it into 128-token kernel
  tiles, KVarN's invariant `tile == kernel block` holds.
- Small robustness fixes: NaN guards in the online-softmax kernels for
  fully-masked chunks / all-empty split-K rows, no per-context recompiles of
  the packed-KV kernel, verify-plan padding zeroed for CUDA-graph replays.
- Not ported: the MLA path, `TQSlidingWindowSpec` (no sliding-window layers
  here), the Gemma-4 config hunk.

Measured on the 3090 (details in [docs/long-context.md](../docs/long-context.md)): 262k context fits
(420k-token pool at 4 slots vs ~200k with fp8), needle-in-a-haystack correct
at 4k…240k, perplexity +0.16%, decode ~20% slower than fp8 at 100k context,
MTP works, short-request throughput lower (2048-token blocks make each
request cost as much as fp8's 800-token block, and prefill flushes cost time).

## RotorQuant-family rotation (`KVARN_ROTATION`)

The rotation step is selectable: `KVARN_ROTATION=hadamard` (default — the
historical Sylvester H, **byte-identical** to before this addition) or the
block-diagonal families from the RotorQuant paper
([scrya.com/rotorquant.pdf](https://www.scrya.com/rotorquant.pdf), March 2026):

- `planar` — one random 2D Givens rotation per element pair (SO(2) blocks),
- `iso` — one random 4D unit-quaternion left-multiplication per group of four
  (the isoclinic SO(4) subgroup, the "fast" IsoQuant mode). **Experimental**
  on this model (see the A/B note below).

Angles/quaternions are seeded (`KVARN_ROTOR_SEED`, default 42 — the seed the
published reference uses, so numbers stay comparable) and therefore fixed
across process restarts, which the KV cache requires: a flushed int4 tile is
quantized in the rotated frame.

Why it touches so little: the backend applies one fixed orthogonal D×D matrix
in exactly five places (K and V on store, Q on read, attention output out, and
the two dequant-read paths). Any orthogonal R keeps `QKᵀ` invariant, so the
only non-trivial change is that the *un-rotation* sites use Rᵀ instead of the
old `@ H` (Hadamard is symmetric, Givens/quaternion are not). The Sinkhorn
variance normalization, RTN, packing, allocator, flush, spec-decode, prefix
cache and CUDA-graph machinery are all rotation-agnostic and untouched.

Files (site-packages paths, minus the leading `vllm/`):

- `model_executor/layers/quantization/kvarn/rotor.py` — the matrix builders
  (`make_rotation_pair(d, kind, device) -> (R, R.T)` fp32, cached per
  (d, kind, device)) and the env dispatch. New in this addition.
- `v1/attention/backends/kvarn_attn.py` — `_H_fp16` (name kept) now caches R
  for the active family, a new `_Ht_fp16` caches Rᵀ (the same object for
  hadamard); `_hadamard()` survives as an alias of `_rotation()`; every
  un-rotation site uses `_rotation_inv()`.
- `v1/attention/ops/triton_kvarn_decode.py` — `kvarn_decode_attention` takes
  `rotation` / `rotation_inv` instead of `hadamard`; its Q-rotation GEMM uses
  R, its output un-rotation uses Rᵀ; `kvarn_verify_attention` likewise.

Validation: `test_rotorquant.py` (CPU-only, run with any venv python that has
the installed tree): orthonormality, the un-rotation identity, fp16 QK
invariance vs the no-rotation fp16 baseline (the rotation adds no extra
error), the iso 4×4 table's fidelity to true Hamilton left-multiplication
(cross-checked against an independent 16-FMA quaternion product, plus group
closure `R(a)@R(b) = R(b·a)` in the row-vector convention — this is what
caught a table that was orthonormal but the *mirror* construction), the full
rotate→Sinkhorn→RTN→dequant→Rᵀ tile round-trip on identical random tiles
across families (K 4-bit: planar ≈ 0.4% and iso ≈ 0.1% *lower* error than
hadamard; V 2-bit: all three within ~0.5%), determinism, and default
invariance. The model-level answer (PPL / IFBench / needle per family) is
measured like everything else in this repo: start the server with
`KVARN_ROTATION=...` and run `python bench/quality_battery.py <tag>` /
`bash bench/run_benchmarks.sh single`.

Model-level A/B (GSM8K-200 greedy, thinking off, `kvarn_k4v2_g128`, 262k
ctx): hadamard 95.5%, planar 95.0% — quality-neutral, as expected of
orthogonal-invariant quantization. Iso is the odd one out at **69.0%**
(answers ~50% longer, still wrong). That number was first hit with a *buggy*
table — the original 4×4 was written in the column convention, so the
row-vector pipeline was actually applying the mirror (right-multiplication)
family, 67.0–67.5% — but the corrected table (verified to ~5e-7 against the
reference Hamilton product) reproduces the weakness, so it is a property of
the construction in this configuration, not of the bug: at head_dim 256 the
3-DOF quaternion blocks (64 independent) quantize worse than the 1-DOF
Givens (128) or the dense Hadamard. `iso` is therefore shipped
**experimental**; `hadamard` stays the default, `planar` is validated
neutral.

**License note:** the reference implementations (scrya-com/rotorquant,
ParaMind2025/isoquant, the llama.cpp fork branch) carry **no license file**;
nothing was copied from them. Only the constructions described in their
papers are re-implemented here (repo license: Apache-2.0). The paper PDF and
source snapshots for reading live in `.rotref/` (git-ignored) — keep it that
way.

### Status / next step (Phase 2)

Phase 1 (this) is **done**: the rotation step is selectable, `hadamard`
stays the default and is byte-identical to the pre-change behaviour,
`planar` is validated quality-neutral, `iso` is faithful to the published
construction but shipped experimental. Phase 2 — the actual RotorQuant win:
fuse the O(d) block rotations into the Triton kernels (K/V on store, Q on
read, output on the way out) so no D×D GEMM is ever launched, and defer the
K rotation to flush time (K stays unrotated in the pool during prefill) — is
**not started**. The published 5.3× prefill / 28% decode deltas are measured
against TurboQuant, not KVarN; against KVarN the honest expectation is a
~2–5% prefill win at short/mid context, ~0 at 100k, and no decode win
(KVarN's decode path already pays zero rotation cost). Revisit only if a
fused rotation is wanted for another reason; the Phase-1 builders
(`_planar_cs`, `_iso_matrix`) are the reference math for it.
