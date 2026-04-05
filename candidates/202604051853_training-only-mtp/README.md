# Training-Only MTP on the Banked 11L Stack

## Hypothesis

The latest banked 11-layer recipe already carries multi-token prediction (MTP) code, but the auxiliary heads are not wired into optimization after the parameter-banking refactor. Re-enabling a **single training-only MTP head** should improve sample efficiency during the 10-minute budget while leaving the exported artifact unchanged, because the script already excludes `mtp_heads.*` from export.

## Why this is promising for this repository

- The current frontier is dominated by the 11-layer banked stack plus small cumulative refinements: XSA, partial RoPE, LN scale, EMA, GPTQ-lite, LeakyReLU squared, and legal TTT.
- The repository already explored heavier architectural changes and found some of them too slow for the 10-minute regime, especially depth recurrence and SwiGLU-heavy variants.
- MTP is one of the few **research-backed sample-efficiency ideas** already present in the codebase but not actually exercised in the current best stack.

## Prior repository evidence that influenced this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`
- **Winning stack context:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`
- **The 11L/XSA/EMA/partial-RoPE progression:**  
  `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/README.md`  
  `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md`  
  `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md`
- **Negative evidence to avoid repeating:**  
  `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md` notes depth recurrence and full/late QAT slowdowns;  
  `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md` reports layer recurrence as a clear miss in a fixed wall-clock budget.
- **Dormant MTP support in earlier 11L scripts:** the pre-banked 11L records already routed `mtp_heads` into `matrix_params`, while the banked 2026-03-23 script does not.

At authoring time there were **no prior directories under `candidates/`**, so this candidate is the first candidate-only experiment in this repository.

## External research that informed it

- Fabian Gloeckle et al., **“Better & Faster Large Language Models via Multi-token Prediction”** (2024): <https://arxiv.org/abs/2404.19737>  
  Motivates MTP as a sample-efficiency auxiliary objective during pretraining.
- Anastasios Gerontopoulos et al., **“Multi-Token Prediction Needs Registers”** (2025): <https://arxiv.org/abs/2505.10518>  
  Reinforces that lightweight MTP variants can work without large architectural expansion.
- Somesh Mehra et al., **“On multi-token prediction for efficient LLM inference”** (2025): <https://arxiv.org/abs/2502.09419>  
  Highlights that NTP-trained hidden states are specialized, which argues for a conservative setup here: one auxiliary head and modest loss weight.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

This candidate makes four focused changes:

1. **Enable MTP by default** with `MTP_NUM_HEADS=1` and `MTP_LOSS_WEIGHT=0.1`.
2. **Wire MTP heads back into Muon optimization** so the auxiliary head actually trains on the banked stack.
3. **Initialize the training-only MTP head from the tied embedding weights** instead of leaving it at zero init, so the auxiliary objective starts from the same vocabulary geometry as the main head.
4. **Resolve default paths relative to the repository root** so `train_gpt.py` can be launched directly from this candidate directory.

The export path still drops `mtp_heads.*`, so the extra head is a **training-only parameter cost**, not an artifact cost.

## How to run

From this candidate directory:

```bash
cd candidates/202604051853_training-only-mtp
RUN_ID=training_only_mtp \
MTP_NUM_HEADS=1 \
MTP_LOSS_WEIGHT=0.1 \
TTT_ENABLED=1 \
TTT_LR=0.002 \
TTT_EPOCHS=3 \
TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 \
TTT_MOMENTUM=0.9 \
TTT_BATCH_SEQS=32 \
TTT_GRAD_CLIP=1.0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All other defaults intentionally inherit the copied 2026-03-23 banked 11-layer recipe.

If you only want to isolate the training-side hypothesis first, leave `TTT_ENABLED=0`.

## Validation

From this candidate directory:

- `python -m compileall train_gpt.py`
- `python -c "import importlib.util; spec = importlib.util.spec_from_file_location('candidate_train', 'train_gpt.py'); module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)"`

Outcomes on this workflow runner:

- `compileall`: **passed**
- module import smoke: **not feasible on this runner** because the repository Python dependencies are not installed here; the import failed immediately on missing `numpy` before any candidate logic ran, and `requirements.txt` also expects `torch` and `sentencepiece`.

## Expected risks and tradeoffs

- **Throughput risk:** even one auxiliary vocab head adds real training-time compute, so any sample-efficiency gain has to beat the lost steps.
- **Objective mismatch:** the main metric is still next-token compression; too much MTP weight could hurt the final NTP optimum.
- **Small-model uncertainty:** the strongest published MTP gains are often larger-model results, so the 512d / 11-layer regime may benefit less.
- **TTT interaction:** if MTP changes the pre-TTT model in ways the score-first TTT stage does not exploit well, the post-TTT gain could be smaller than the pre-TTT gain.
