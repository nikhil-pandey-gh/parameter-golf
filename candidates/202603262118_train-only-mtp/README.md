# Train-only MTP on the leaky 11L stack

## Hypothesis

Enable **multi-token prediction (MTP)** as a **training-only auxiliary loss** on top of the strongest current 11-layer stack. The goal is to improve sample efficiency during the 600-second training window without paying any artifact-size cost at export time, because the extra MTP heads are stripped before quantization and evaluation.

## Why this is promising for this repository

The repository's best gains have come from small, local changes that either improve **compression-aware training** or improve **effective context usage**. The current record is already close to the 16 MB cap, so the best next idea should be **training-only** or otherwise byte-neutral.

MTP fits that constraint unusually well here:

- the recent record lineage already contains dormant MTP code paths,
- the auxiliary heads are already excluded from the exported checkpoint,
- the added compute is small compared with the full 11-layer trunk, and
- external work suggests MTP improves sample efficiency and induction-style reasoning under fixed training budgets.

## Prior records and candidates that influenced this candidate

There were **no prior `candidates/` directories** in the repository when this candidate was created.

The main record influences were:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - strongest current stack,
  - provides the leaky-ReLU-squared MLP, parameter banking, parallel Muon, XSA, partial RoPE, EMA/SWA, and legal TTT scaffold.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - first high-performing script in this line that already wired MTP heads and excluded them from export.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - confirms the value of partial RoPE + LN scaling while also showing that one late-QAT path was accidentally a no-op.

## External research that informed the choice

- Fabian Gloeckle et al., **"Better & Faster Large Language Models via Multi-Token Prediction"** (2024)  
  https://arxiv.org/abs/2404.19737
- Guoliang Zhao et al., **"Self-Distillation for Multi-Token Prediction"** (2026)  
  https://arxiv.org/abs/2603.23911
- Lorenzo Noci et al., **"Thinking into the Future: Latent Lookahead Training for Transformers"** (2026)  
  https://arxiv.org/abs/2603.20219

The first paper is the direct justification: MTP adds future-token supervision with independent heads on a shared trunk and improves sample efficiency with little training-time overhead. The later 2026 papers are relevant because they suggest the same general direction still has headroom, but they also introduce more machinery than seems appropriate for a first candidate in this repo.

## What changed versus the chosen base implementation

Base implementation: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. **Turned on training-only MTP by default**
   - `MTP_NUM_HEADS=2`
   - `MTP_LOSS_WEIGHT=0.1`

2. **Fixed optimizer wiring for MTP heads**
   - the copied 2026-03-23 script logged MTP parameters and excluded them from export, but did not actually attach the MTP head weights to an optimizer path.
   - this candidate adds the MTP head weights to the AdamW-managed replicated parameter set so the auxiliary heads train correctly.

3. **Kept export byte-neutral**
   - exported checkpoints still drop all `mtp_heads.*` tensors before quantization and evaluation.

4. **Made the script runnable from the candidate directory**
   - default dataset and tokenizer paths now point to `../../data/...`.

5. **Added a FlashAttention fallback**
   - when `flash_attn_interface` is unavailable, or when the model is running on CPU, the script falls back to `torch.nn.functional.scaled_dot_product_attention`.
   - this is mainly for importability and tiny CPU shape-smoke checks; the intended full run remains the FA3 path on challenge hardware.

## How to run

From this candidate directory:

```bash
cd candidates/202603262118_train-only-mtp
RUN_ID=mtp_aux_heads \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

That uses the candidate defaults, including:

- 11 layers, 512 width, 8 heads / 4 KV heads
- 3x MLP with leaky-ReLU-squared activation
- partial RoPE + LN scale + XSA on the last 4 layers
- EMA + tight SWA + int6/lzma export
- **2 MTP heads with weight 0.1**

To evaluate whether MTP stacks with the current legal-TTT recipe, run the same command with:

```bash
TTT_ENABLED=1
```

If the MTP overhead proves too high, the first ablations to try are:

```bash
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.05
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.10
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.05
```

## Main risks and tradeoffs

- **Training-time overhead:** even though MTP adds no export bytes, extra vocab heads still cost compute and may reduce the number of steps completed in 600 seconds.
- **Objective mismatch:** the auxiliary heads may improve pre-quant training loss without translating into better post-quant BPB.
- **Need for tuning:** one or two heads is probably safe, but larger MTP configurations could destabilize or simply waste budget.
- **Interaction with TTT:** legal TTT already changes the evaluation frontier, so pre-TTT wins may or may not stack linearly.

## Validation

The validation performed for this candidate is intentionally lightweight and local:

- `python -m compileall train_gpt.py`
- a tiny CPU-only import / forward smoke test using the new attention fallback

Results in this workflow:

- `python -m compileall train_gpt.py`  
  **Passed**
- tiny CPU-only import / forward smoke test  
  **Not feasible in this container** because both `python` and `python3` are missing the `torch` package, so the candidate cannot be imported far enough to instantiate the model. The script now includes a CPU-safe FlashAttention fallback specifically so this smoke becomes possible in any environment that has PyTorch, even if FA3 is missing or CUDA is unavailable.
