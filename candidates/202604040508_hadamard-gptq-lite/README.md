# Hadamard-search GPTQ-lite on the 03-23 stack

## Hypothesis

The current leaderboard is already squeezing a lot out of the 11-layer XSA/partial-RoPE/EMA/GPTQ-lite family, so the next cheap win is likely in the **post-training int6 export path** rather than in a brand-new architecture. The hypothesis here is that **searching over a small bank of deterministic block-Hadamard rotations before GPTQ-lite quantization** will reduce outlier-driven rowwise quantization error enough to improve the final int6 roundtrip/sliding-window BPB, while adding **zero training-time cost** and almost no code-size overhead.

## Why this is promising here

- The repo's strongest pre-TTT record already comes from better export quality: GPTQ-lite clip search + EMA + longer warmdown reached **1.1233** in `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md`.
- The current top run, `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`, keeps the same broad stack and then wins with small additive improvements (`LeakyReLU^2`, parameter banking, legal TTT), which suggests the architecture is mature enough that **small quantization gains still matter**.
- The repo's non-record 4-hour baseline still suffered a large post-quant gap (`records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/README.md`), reinforcing that compression/export remains a core bottleneck.
- Negative evidence matters too: naive layer recurrence regressed in `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md`, so this candidate deliberately avoids spending a candidate slot on a broad architecture detour.

There were **no prior `candidates/` experiments** in this checkout, so this starts from the best available record stack instead of revisiting an older candidate branch.

## Prior records that influenced this candidate

1. `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` — chosen base implementation.
2. `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` — direct inspiration for targeting the GPTQ-lite export path.
3. `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` — reminder that the stack's late gains have been coming from small, cheap additions (partial RoPE, LN scaling), not heavy rewrites.
4. `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/` — negative signal against recurrence/depth reuse under a tight wall-clock budget.

## External research that informed it

- **QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs** — <https://arxiv.org/abs/2404.00456>  
  Main takeaway used here: orthogonal rotations can make quantization easier by spreading outliers without changing the full-precision function.
- **SpinQuant: LLM quantization with learned rotations** — <https://arxiv.org/abs/2405.16406>  
  Main takeaway used here: some rotations are clearly better than others, so a tiny search over multiple rotations is more plausible than a single fixed transform.
- **KurTail: Kurtosis-based LLM Quantization** — <https://arxiv.org/abs/2503.01483>  
  Main takeaway used here: low-cost, data-free rotation selection is a practical direction when full learned rotations are too heavy.

This candidate does **not** attempt the full learned-rotation setup from SpinQuant. Instead it borrows the central idea and adapts it to this repo's constraints with a tiny deterministic search over sign-flipped block-Hadamard rotations.

## What changed vs the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in `train_gpt.py`:

1. Added `ROTATE_BLOCK_SIZE` and `ROTATE_SEARCH_SEEDS` hyperparameters.
2. Added cached block-Hadamard helpers plus deterministic sign-pattern generation.
3. Changed the int6 GPTQ-lite path so each eligible 2D int6 matrix (specifically, ones whose input width is divisible by a **power-of-two** `ROTATE_BLOCK_SIZE`):
   - first tries the existing identity-basis GPTQ-lite quantization,
   - then tries a small bank of rotated bases,
   - keeps the best candidate by reconstruction MSE after undoing the rotation.
4. Stored the selected rotation in quantization metadata and undid it during dequantization.
5. Logged how many tensors actually chose a rotated basis.

Training, architecture, parameter banking, legal TTT, and the LeakyReLU(0.5)^2 MLP are otherwise unchanged from the 03-23 record.

## How to run

From this candidate directory:

```bash
ROTATE_BLOCK_SIZE=128 ROTATE_SEARCH_SEEDS=0,1,2,3 \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful output to watch for:

- `int6_rotation:block_size:... rotated_tensors:...`
- `final_int6_roundtrip_exact ...`
- `final_int6_sliding_window_exact ...` when `EVAL_STRIDE=64`
- `final_int6_sliding_window_s64_exact ...` when you run a non-64 eval stride and still want the standard stride-64 report
- legal TTT metrics when `TTT_ENABLED=1`

`ROTATE_BLOCK_SIZE` must stay a positive power of two (for example `64`, `128`, or `256`).

## Validation

Commands run in this workflow:

```bash
python -m compileall candidates/202604040508_hadamard-gptq-lite/train_gpt.py
python - <<'PY'
import importlib.util
print('torch_found', importlib.util.find_spec('torch') is not None)
print('sentencepiece_found', importlib.util.find_spec('sentencepiece') is not None)
print('numpy_found', importlib.util.find_spec('numpy') is not None)
PY
```

Outcomes:

- `compileall` succeeded.
- A CPU import/smoke run was **not feasible in this runner** because the available Python environment did not have `torch`, `sentencepiece`, or `numpy` installed, so the validation here is limited to syntax compilation.

## Main risks and tradeoffs

- Better reconstruction MSE may not translate into better validation BPB after sliding-window eval or TTT.
- Rotated weights may compress a bit differently under `lzma`, so a quantization win could be partly offset by artifact-size changes.
- Export time is higher because each eligible matrix now evaluates multiple rotation candidates.
- The current search is intentionally lightweight and data-free; if this direction works, the next experiment should probably test either a smarter rotation bank or a small calibration-aware variant.
