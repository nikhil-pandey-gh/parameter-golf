# Shared Depth Reuse (ALBERT-style shared cores on the 11L static stack)

## Hypothesis

The best static branch in this repository already has a strong recipe: 11 layers, 3x MLP, SmearGate + BigramHash, late-layer XSA, partial RoPE, LN scaling, EMA/SWA, and GPTQ-lite-style mixed low-bit export. The main unexplored lever is **cross-layer parameter reuse**.

This candidate tests whether a **smaller bank of shared transformer cores** can be executed across **more logical depths** while keeping **per-depth control parameters untied**. The goal is to trade redundant weight bytes for additional effective depth without abandoning the repository's proven architecture or export path.

## Why this is promising for this repository

Repository review showed a few stable trends:

- Sliding-window evaluation is now table stakes.
- Low-bit export quality is a central bottleneck, so artifact bytes matter.
- Depth and MLP width kept helping from 9L to 10L to 11L.
- The strongest static line is the 11-layer branch culminating in `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`.
- The current best overall record adds TTT and systems work, but its pre-TTT stack still suggests that the static model has room left.

Under this repository's state-dict-based artifact accounting, storing 6 shared cores instead of 12 distinct full blocks is a direct byte saving. That makes shared depth reuse one of the few ideas that can plausibly improve the compute/byte tradeoff without introducing broad new infrastructure.

## Prior records and candidates that influenced this candidate

There was **no pre-existing `candidates/` directory** when this candidate was created.

The main local influences were:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Chosen as the implementation base because it is the strongest clean static branch in `records/`.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - Carried forward the partial-RoPE and LN-scale direction.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Motivated swapping `ReLU^2` for `LeakyReLU(0.5)^2` on the static path.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
  - Important negative result: naive layer recurrence was bad under a strict wall-clock budget. This candidate differs by reusing only the large weight-bearing cores while keeping per-depth norms, scales, residual mixing, skip structure, and XSA placement untied.

## External research that informed it

- **ALBERT** (cross-layer parameter sharing): https://arxiv.org/abs/1909.11942
  - Motivates reusing transformer weights to cut parameter count without discarding depth.
- **Universal Transformer** (iterative depth reuse): https://arxiv.org/abs/1807.03819
  - Supports the idea that repeated computation over a shared transition can increase effective depth.
- **DeepNet / DeepNorm**: https://arxiv.org/abs/2203.00555
  - Motivates preserving depth-dependent scaling when increasing logical depth. This candidate keeps per-layer norm/scale controls untied instead of making the entire layer fully recurrent.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

This candidate keeps the base stack's major ingredients:

- GPTQ-lite mixed low-bit export
- EMA/SWA training logic
- SmearGate + BigramHash
- partial RoPE
- late-layer XSA
- LN scaling
- shared value embeddings
- 2048-token training / sliding-window evaluation path

Then it makes the following targeted changes:

1. **Shared-depth architecture**
   - Default logical depth becomes `NUM_LAYERS=12`.
   - New hyperparameter: `NUM_SHARED_BLOCKS=6`.
   - Large attention/MLP weights live in `shared_blocks` and are reused across 12 logical layers.
   - Per-depth control parameters live in `layer_controls` and remain unique.

2. **Mirror reuse schedule**
   - The default mapping is a U-Net-like mirror:
   - `[0, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0]`
   - This keeps the decoder side aligned with skip-connected encoder depths instead of using a flat repeated loop.

3. **Untied per-depth controls**
   - Each logical layer still owns its own:
     - `attn_norm`
     - `mlp_norm`
     - `attn_scale`
     - `mlp_scale`
     - `resid_mix`
     - optional DTG gate
     - XSA enable/disable decision
     - depth-dependent LN scale factor

4. **LeakyReLU(0.5)^2 MLP**
   - Replaces `ReLU^2` with `LeakyReLU(0.5)^2` in the MLP, following the later record evidence.

5. **Attention fallback for validation**
   - If `flash_attn_interface` is unavailable, attention falls back to PyTorch SDPA. This is mainly to make low-cost local smoke checks possible; the intended fast path on H100 still uses FlashAttention when available.

6. **Candidate-local checkpoint schema**
   - The artifact serialized by this candidate uses the candidate's own `shared_blocks` + `layer_controls` structure and is reloaded by the same script during round-trip evaluation.
   - This intentionally keeps the candidate self-contained without modifying repository-wide loader code outside the new candidate directory.

## How to run or evaluate it

From the repository root:

```bash
cd candidates/202603292145_shared-depth-reuse
RUN_ID=shared_depth_reuse \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Important defaults for this candidate:

- `NUM_LAYERS=12`
- `NUM_SHARED_BLOCKS=6`
- `ROPE_DIMS=16`
- `XSA_LAST_N=4`
- `VE_LAYERS=10,11`
- `TRAIN_SEQ_LEN=2048`
- `TRAIN_BATCH_TOKENS=786432`

If you want to ablate the sharing idea cleanly, the easiest knobs are:

```bash
NUM_SHARED_BLOCKS=12   # disables sharing while keeping 12 logical layers
NUM_SHARED_BLOCKS=6    # default shared-depth candidate
NUM_LAYERS=11 NUM_SHARED_BLOCKS=11  # fully untied 11-layer control comparison
```

For the mirrored shared-depth schedule, `NUM_SHARED_BLOCKS` should be either:

- `<= min(num_encoder_layers, num_decoder_layers)` for actual sharing, or
- exactly `NUM_LAYERS` to disable sharing cleanly.

## Expected risks and tradeoffs

- **Step count risk:** more logical depth raises per-step compute, so a 600s run may complete fewer optimizer steps.
- **Specialization risk:** shared cores may under-specialize compared with fully untied 11L/12L stacks.
- **Recurrence history:** a prior 1x5090 recurrence experiment was clearly negative. This candidate attempts to address that by keeping per-depth controls untied, but it is still the main uncertainty.
- **Mapping sensitivity:** the mirror reuse schedule is a reasonable first choice, not a proven optimum. A different sharing map could outperform it.
- **Quantization interactions:** fewer unique large matrices should help the artifact budget, but the best reinvestment of those bytes may need more tuning (depth, bigram capacity, VE placement, or int5/int6 mix).

## Validation

Validation was run after implementation and the exact commands/outcomes are recorded below.

- `python -m compileall candidates/202603292145_shared-depth-reuse/train_gpt.py`
  - Outcome: **passed**
- CPU smoke import / forward pass with a tiny randomly initialized model and FlashAttention fallback disabled by environment
  - Outcome: **not feasible in this runner** because the local Python environment does not have `torch` installed (`ModuleNotFoundError: No module named 'torch'`). The candidate script itself was updated so that importing the module no longer requires `flash_attn_interface`, but an actual forward-pass smoke test still needs PyTorch.
- `python -m compileall train_gpt.py train_gpt_mlx.py data`
  - Outcome: **passed**
