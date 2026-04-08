# Mirror-Shared Depth on the Banked 11L Stack

## Hypothesis

The repo's best 10-minute runs already spend most of their artifact budget on a strong 11-layer transformer plus quantization tricks. This candidate tests a different axis: **keep the full 11 logical layers and training-time depth, but reuse the large attention/MLP matrix banks across mirrored encoder/decoder depths** while leaving per-layer norms, gates, skip weights, XSA flags, and value-embedding scales independent.

The hope is that this behaves like a lightweight ALBERT/Universal-Transformer-style depth reuse scheme: lower large-matrix payload, milder quantization pressure, and some regularization from repeated banks, without paying the wallclock overhead that hurt explicit recurrent-depth experiments elsewhere.

## Why this is promising here

- The record history already squeezed obvious wins out of sliding-window eval, EMA/SWA, SmearGate, BigramHash, partial RoPE, GPTQ-lite, and late-stage quantization tuning.
- No prior `candidates/` directory existed, and none of the reviewed `records/` implemented explicit cross-layer transformer bank sharing.
- The March 23 banked stack already centralizes the heavy matrices, so mirrored sharing can be added surgically instead of rewriting the model.
- The non-record 1x5090 exploration reported that **layer recurrence** was unhelpful, but that experiment added extra recurrence/compute. This candidate keeps the same 11 logical layers and only shares parameter banks, so it is a narrower and cheaper test.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - strongest clean pre-TTT stack: 11L, XSA, partial RoPE, LN scaling, EMA, GPTQ-lite.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - banked large matrices, Parallel Muon, and LeakyReLU(0.5)^2 MLPs.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - partial RoPE + LN scaling mattered even when late QAT turned out to be a no-op.
- `records/track_10min_16mb/2026-03-20_11L_EfficientPartialXSA_FA3_SWA120/`
  - deep-layer XSA remained one of the best architecture-side improvements.

## External research that informed it

- [ALBERT](https://arxiv.org/abs/1909.11942) — showed that cross-layer parameter sharing can preserve depth while reducing parameter count.
- [Universal Transformer](https://arxiv.org/abs/1807.03819) — motivated repeated shared transformations across depth instead of treating every layer as fully independent.
- [Recurrent multiple shared layers in Depth for Neural Machine Translation](https://arxiv.org/abs/2108.10417) — showed that shared-depth transformers can recover much of deeper-model quality at a lower parameter budget.
- [SCORE: Replacing Layer Stacking with Contractive Recurrent Depth](https://arxiv.org/search/?query=%22SCORE%3A+Replacing+Layer+Stacking+with+Contractive+Recurrent+Depth%22&searchtype=all&source=header) — recent evidence that recurrent depth can be a viable alternative to fully distinct stacks.

I also considered SASQ-style activation-only QAT and SpinQuant/QuaRot-style rotation-assisted PTQ, but mirror-shared depth was the best fit for a single self-contained candidate without introducing wider infrastructure.

## What changed versus the chosen base

Base implementation: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. Added `SHARED_DEPTH` as an explicit reconstruction/logging parameter for the mirrored bank layout. In this candidate it is intentionally fixed to the U-Net-aligned default `ceil(NUM_LAYERS/2)`, which gives the 11-layer map `[0, 1, 2, 3, 4, 4, 3, 2, 1, 0, 5]`.
2. Reduced the large bank tensors from per-layer banks to **shared mirrored banks**:
   - `qo_bank`: `2 * SHARED_DEPTH`
   - `kv_bank`: `2 * SHARED_DEPTH`
   - `mlp_up_bank`: `SHARED_DEPTH`
   - `mlp_down_bank`: `SHARED_DEPTH`
3. Kept logical-layer-specific control parameters intact:
   - RMSNorms
   - `attn_scale`, `mlp_scale`, `resid_mix`
   - skip weights
   - XSA placement
   - value-embedding layer scales
4. Routed each logical layer through `self._bank_weights(layer_idx)` so the model still executes 11 distinct logical blocks while reusing only the heavy matrices. The sharing now follows the actual skip-paired execution order: `(4,5), (3,6), (2,7), (1,8), (0,9)`, with layer 10 kept unmatched.
5. Left the rest of the March 23 stack intact, including LeakyReLU(0.5)^2, Parallel Muon, SmearGate, BigramHash, partial RoPE, VE, EMA, and optional legal TTT support.

## How to run

From this candidate directory:

```bash
SHARED_DEPTH=6 NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_LAYERS=9,10 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
LAWA_ENABLED=0 TTT_ENABLED=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script inherits the March 23 optional evaluation-time features, but this candidate is meant to be tested first as a **training-time shared-depth** change with `TTT_ENABLED=0`.

## How to evaluate

- The script still writes `final_model.int6.ptz` and runs its roundtrip + sliding-window evaluation path.
- It now logs `shared_depth` and `layer_to_bank` so runs clearly show which logical layers are sharing banks.

## Validation run for this candidate

Executed here:

```bash
python -m compileall candidates/202604080423_mirror-shared-depth/train_gpt.py
```

Outcome:

- **Passed**.

Runtime smoke test status:

- **Not feasible in this environment.**
- The available Python environment is missing `numpy`, `torch`, `sentencepiece`, and `flash_attn_interface`.
- The script also hard-requires CUDA at runtime, so a true CPU-only launch would not reflect the supported execution path.

## Main risks and tradeoffs

- **Late-layer specialization may weaken.** The best recent records benefited from making the deepest layers special (XSA, VE, partial-RoPE-era tuning); mirrored bank sharing may partially undo that.
- **Quantized export still expands the shared banks into logical-layer views before rebanking for roundtrip compatibility.** The compressed artifact may therefore save fewer bytes than the raw parameter count suggests.
- **Weight sharing can underfit.** If the repo's current gains mostly come from increasingly specialized late layers, this may regularize too hard and lose BPB.
- **The negative recurrence result is adjacent, not identical.** If shared-depth was bad there for fundamental reasons rather than extra wallclock cost, this candidate may repeat that failure mode.
