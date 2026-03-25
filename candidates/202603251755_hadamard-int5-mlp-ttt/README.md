# Hadamard-Rotated Int5 MLP + Legal TTT

## Hypothesis

The current repo history says the best gains now come from two places: keeping the strong 11-layer LeakyReLU² + TTT training stack, and squeezing more quality out of the export path. This candidate tests a lightweight QuaRot/SpinQuant-style idea: apply a deterministic block-Hadamard rotation to MLP weights before post-training quantization so the MLP banks tolerate **int5** export better, then reinvest the saved bytes into a larger **BigramHash(3072)** table.

## Why this is promising for this repository

- The best current record is `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`, which already shows that the strongest training/eval stack is **11L + LeakyReLU² + legal score-first TTT + parameter banking**.
- `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/README.md` showed that **int5 MLP / int6 attention** can buy meaningful artifact headroom.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md` showed that **quantizer-only improvements** still matter near the frontier.
- Recent PTQ papers argue that **orthogonal rotations reduce outliers and improve low-bit quantization** without changing full-precision function, which is a strong fit for this repo's artifact-limited setting.

There were no prior experiments under `candidates/` when this candidate was created, so the comparison set here is the root baseline plus `records/`.

## Prior records that influenced this candidate

- Base implementation: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
- Mixed-bit export precedent: `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/`
- Quantizer refinement precedent: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- Architectural trendline: `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`

## External research that informed it

- **GPTQ** — Frantar et al., 2022: <https://arxiv.org/abs/2210.17323>
  - Strong evidence that post-training weight quantization can preserve model quality at low bitwidths.
- **QuaRot** — Ashkboos et al., 2024: <https://arxiv.org/abs/2404.00456>
  - Shows that orthogonal rotations can remove outliers and make low-bit quantization easier while preserving the underlying model function.
- **SpinQuant** — Liu et al., 2024/2025: <https://arxiv.org/abs/2405.16406>
  - Shows that rotation choice matters materially for low-bit quantization accuracy.
- **SmoothQuant+** — Pan et al., 2023: <https://arxiv.org/abs/2312.03788>
  - Reinforces that weight-only PTQ still has room for better preprocessing/equalization before quantization.

This candidate intentionally uses the simplest repo-friendly version of those ideas: a fixed block-Hadamard rotation in the export path, with no calibration set and no new training infrastructure.

## What changed versus the chosen base implementation

Chosen base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

- Keeps the base model/training stack:
  - 11 layers, 512 hidden size
  - LeakyReLU(0.5)^2 MLP
  - XSA on late layers
  - partial RoPE
  - VE layers
  - EMA + SWA
  - legal score-first TTT
  - parameter banking / Parallel Muon path
- Changes default `BIGRAM_VOCAB_SIZE` from the smaller submitted value to **3072**.
- Changes default TTT settings to match the strongest repo result more closely:
  - `TTT_ENABLED=1`
  - `TTT_FREEZE_BLOCKS=0`
- Replaces the old export-only mixed int6 path with a **rotated mixed-bit path**:
  - MLP tensors default to **int5**
  - attention tensors default to **int6**
  - MLP tensors are block-Hadamard rotated on the last dimension before quantization and inverse-rotated after dequantization
  - export artifact renamed to `final_model.mixedint.ptz`

The training loop itself is intentionally left almost unchanged; this candidate is mostly testing whether the extra artifact headroom from rotation-aided MLP int5 export can be converted into a better frontier point with a larger bigram table.

## How to run or evaluate it

From this directory:

```bash
RUN_ID=hadamard_int5_mlp_ttt \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
SEED=1337 \
BIGRAM_VOCAB_SIZE=3072 \
MLP_QUANT_BITS=5 \
ATTN_QUANT_BITS=6 \
QUANT_ROTATE_MLP=1 \
QUANT_ROTATION_BLOCK=128 \
TTT_ENABLED=1 \
TTT_FREEZE_BLOCKS=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

If you want a faster ablation before the full score-first TTT pass, disable TTT:

```bash
TTT_ENABLED=0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Main expected risks and tradeoffs

- A fixed Hadamard rotation is cheaper than learned rotations from SpinQuant, but it may also be weaker.
- Bigger `BigramHash(3072)` improves token-pair capacity but adds parameters that still need to fit the 16MB budget after export.
- Int5 MLP export may save bytes but still lose too much quality if the rotation is not enough.
- This candidate does **not** implement a corrected late-QAT path, which remains an obvious follow-up experiment.

## Validation

Validation run in this workflow:

```bash
python -m compileall candidates/202603251755_hadamard-int5-mlp-ttt/train_gpt.py
```

Outcome:

- Passed.

Attempted CPU-only smoke validation:

```bash
python - <<'PY'
import importlib.util
print(importlib.util.find_spec("torch"))
print(importlib.util.find_spec("sentencepiece"))
PY
```

Outcome:

- The runner Python did **not** have `torch` or `sentencepiece` installed, so importing the training script for a real CPU-side quantization roundtrip smoke test was **not feasible in this environment**.
- A full startup smoke test is also limited by the script's CUDA requirement in `main()`.

## Suggested next experiments

1. Compare `BIGRAM_VOCAB_SIZE` at `2048`, `3072`, and `4096` with the rotated int5 MLP export unchanged.
2. Compare `QUANT_ROTATION_BLOCK=64` vs `128` vs `256`.
3. Try the same export path on the `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` base without TTT to isolate export quality from eval-time adaptation.
4. Combine this candidate with a real, runtime-safe late-QAT implementation instead of the compile-foldable flag pattern used earlier in the repo history.
