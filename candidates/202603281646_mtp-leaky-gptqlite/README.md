# Candidate: MTP + LeakyReLU² on the GPTQ-lite 11L stack

## Hypothesis

Add a **single training-only multi-token prediction (MTP) head** to the strongest pre-TTT quantization stack, and combine it with the repo's later **LeakyReLU(0.5)^2** activation win.

The bet is that one auxiliary future-token head improves sample efficiency during the 600-second training window, while LeakyReLU^2 preserves negative-gradient flow in the MLP. Because the extra MTP head is excluded from export, the artifact budget stays focused on the main model.

## Why this is promising for this repository

This repo's records suggest two constraints matter most late in the game:

- the strongest submissions are already close to the 16MB artifact cap, so techniques that improve training **without adding exported weights** are unusually valuable;
- the leaderboard is wallclock-limited, so **sample efficiency per training step** matters at least as much as raw parameter count.

MTP is a good fit for both constraints. The trunk stays unchanged at inference/export time, and this codebase already had dormant MTP hooks that were never activated in the recorded runs. That makes it a high-leverage, low-infrastructure experiment.

## Repository evidence that influenced this candidate

### Chosen base implementation

This candidate starts from:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md`

That record is the cleanest strong **pre-TTT** stack in this repo: 11 layers, XSA on the last 4 layers, Partial RoPE, LN scale, VE layers, EMA, tight SWA, GPTQ-lite per-row clip search, and late QAT.

### Other records that shaped the choice

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`
  - showed that **LeakyReLU(0.5)^2** was a real improvement on a strong 11-layer stack.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md`
  - confirmed that Partial RoPE + LN scale are worth preserving.
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/README.md`
  - confirmed XSA + EMA as part of the winning lineage.

There were **no prior experiments under `candidates/`** at the time of this run.

## External research that informed this candidate

I reviewed several compact-model directions before choosing this one, including rotation-aware quantization (QuaRot / SpinQuant), block sharing (MobileLLM-LS), LayerSkip-style intermediate exits, and MTP.

The deciding paper for this candidate was:

- Fabian Gloeckle et al., **"Better & Faster Large Language Models via Multi-Token Prediction"** (arXiv:2404.19737)
  - https://arxiv.org/abs/2404.19737
  - The paper reports that predicting multiple future tokens from a shared trunk improves training sample efficiency and downstream quality while keeping inference-time deployment flexible.

I also considered these higher-risk alternatives and intentionally did **not** implement them here because they would require broader trainer/export surgery:

- QuaRot (rotation-aware quantization): https://arxiv.org/abs/2404.00456
- SpinQuant: https://arxiv.org/abs/2405.16406
- MobileLLM / MobileLLM-LS: https://arxiv.org/abs/2402.14905

Given this repository's existing code and the requirement to keep changes precise, MTP had the best upside-to-complexity ratio.

## What changed versus the chosen base

Compared with the `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` base, this candidate makes four targeted changes:

1. **Enable one auxiliary MTP head by default**
   - `MTP_NUM_HEADS=1`
   - `MTP_LOSS_WEIGHT=0.15`
   - The extra head predicts the next additional token during training.
   - Export still excludes `mtp_heads.*`, so artifact size is unaffected.

2. **Switch the MLP activation from ReLU^2 to LeakyReLU(0.5)^2**
   - This imports the activation change that helped the later legal-TTT record, but applies it on the stronger GPTQ-lite / EMA pre-TTT stack.

3. **Make the candidate runnable from its own directory**
   - Default `DATA_PATH` and `TOKENIZER_PATH` now resolve relative to the repository root instead of `./data` under the candidate directory.

4. **Add a non-FlashAttention fallback for import/smoke testing**
   - If `flash_attn_interface` is unavailable, attention falls back to PyTorch SDPA.
   - This is mainly for lightweight CPU-side sanity checks; H100 runs should still use FlashAttention when available.

## How to run / evaluate

From the candidate directory:

```bash
cd candidates/202603281646_mtp-leaky-gptqlite

# Main run (8 GPU, same overall recipe family as the base record)
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful ablations:

```bash
# Disable MTP to isolate the activation change
MTP_NUM_HEADS=0 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Revert to ReLU^2 while keeping the rest of the candidate
LEAKY_RELU_SLOPE=0.0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script still exports an int6 GPTQ-lite-style artifact and excludes the auxiliary MTP head from export.

## Expected risks / tradeoffs

- **Training throughput risk:** even one auxiliary MTP head adds extra logits and CE work every step. If the step-time regression is too large, the sample-efficiency gain may be canceled out.
- **Objective mismatch risk:** MTP may help trunk representations at larger scales but could be neutral or noisy at this tiny-model / short-run regime.
- **Activation interaction risk:** LeakyReLU^2 helped on a later stack, but interactions with GPTQ-lite + VE + late QAT are still untested.
- **Validation uncertainty:** this candidate is syntax-checked locally, but a full end-to-end train/eval run still requires the repo's normal GPU environment plus dataset/tokenizer assets.

## Validation

Commands attempted during this workflow:

```bash
python -m compileall candidates/202603281646_mtp-leaky-gptqlite/train_gpt.py
```

Outcome:

- Passed.

Additional lightweight smoke validation attempted:

- create a temporary venv under `/tmp/gh-aw/agent/mtp-venv`,
- install `torch`, `sentencepiece`, and `numpy`,
- import the candidate module,
- instantiate a small CPU `GPT`,
- run `forward`, `forward_logits`, and int6 quantization/dequantization helpers on random tensors.

Outcome:

- Passed.
- Smoke result: `smoke_ok 4.79246 (2, 16, 64) (2, 16, 64) 51 51`

Environment limitation:

- A full `main()` training/eval run was **not** possible in this workflow checkout because the shared `data/datasets/*` and `data/tokenizers/*` assets are not present locally.
