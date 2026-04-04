# Candidate: one-head MTP auxiliary loss on the current SOTA stack

## Hypothesis

Enable a **single training-only multi-token prediction (MTP) head** on top of the current best repository stack. The auxiliary head should improve representation learning and sample efficiency during the fixed 600-second training budget, while adding **no exported-model bytes** because the MTP head is excluded from export before quantization.

## Why this is promising for this repository

- The repo's biggest wins come from techniques that improve **compressed** model quality, not just pre-quant loss.
- The current record stack already has a dormant MTP path in code, but every reviewed record and log keeps `mtp_num_heads:0`, so this remains effectively untested here.
- MTP fits the challenge constraints unusually well: it adds training signal without forcing a larger exported model.
- External research reports that multi-token prediction improves **sample efficiency** and helps induce stronger algorithmic / induction-head behavior in autoregressive transformers, which is exactly the regime this challenge cares about.

## Prior records that influenced this candidate

1. `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
   - strongest pure train/export base before legal TTT,
   - established the 11L + XSA4 + partial RoPE + LN scale + EMA + GPTQ-lite stack.
2. `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
   - current best full stack,
   - contributed LeakyReLU(0.5)^2, VE on layers 9-10, legal score-first TTT, and the candidate's concrete base script.
3. `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
   - important negative result: naive layer recurrence / reuse regressed badly in this repo's fixed wallclock setting.

That last point matters because external research also suggested paired-layer weight sharing as a promising compression idea. I did **not** choose it here because this repository already has a strong local warning that reuse-style ideas can lose more step throughput than they gain in parameter efficiency.

## External research

- Fabian Gloeckle et al., **"Better & Faster Large Language Models via Multi-token Prediction"** (arXiv:2404.19737)  
  <https://arxiv.org/abs/2404.19737>

Key claims from that paper that matter here:

- predicting multiple future tokens from a shared trunk improves **sample efficiency**,
- the extra heads act as an **auxiliary training task**,
- the approach improves internal next-token modeling behavior even when the architecture is still an autoregressive LM.

## What changed versus the chosen base implementation

Base implementation copied from:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Candidate-specific changes:

1. **Enabled one MTP head by default**
   - `MTP_NUM_HEADS=1`
   - `MTP_LOSS_WEIGHT=0.1`
2. **Fixed the dormant MTP path so the head actually trains**
   - the copied base script had MTP loss code, but the MTP head weights were not wired into any optimizer.
   - this candidate adds the MTP head weights to the replicated Adam parameter set so the auxiliary loss can influence both the head and the trunk.
3. **Kept the head training-only**
   - the existing export path already strips `mtp_heads.*` from the saved artifact before quantization and evaluation.
4. **Made the script runnable from inside this candidate directory**
   - default `DATA_PATH` and `TOKENIZER_PATH` now resolve relative to the repository root instead of assuming the working directory is the repo root.
5. **Promoted the record-tuned evaluation defaults into the candidate file**
   - `BIGRAM_VOCAB_SIZE=1536`
   - `TTT_ENABLED=1`
   - `TTT_FREEZE_BLOCKS=0`

The core idea is still deliberately narrow: keep the current best stack intact and change only the training signal.

## How to run

From this candidate directory:

```bash
cd candidates/202604041214_mtp-aux-head
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful overrides:

```bash
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.1 torchrun --standalone --nproc_per_node=8 train_gpt.py
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.1 torchrun --standalone --nproc_per_node=8 train_gpt.py
TTT_ENABLED=0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Evaluation notes

- Final artifact evaluation still uses the existing int6 roundtrip + sliding-window path.
- Legal score-first TTT remains enabled by default, matching the current strongest stack.
- MTP heads are removed before export, so they do **not** add exported-model bytes even though this candidate still pays for its code bytes.

## Main risks and tradeoffs

- **Wallclock risk:** even one auxiliary head adds training compute, so higher sample efficiency must outweigh reduced steps.
- **Objective mismatch:** MTP can regularize the trunk well, but it can also pull optimization away from the exact next-token objective if the loss weight is too large.
- **Interaction risk:** the repo's current stack already mixes LeakyReLU^2, VE, XSA, EMA/SWA, GPTQ-lite, and TTT; MTP could help, but it could also interfere with the carefully tuned late-training regime.
- **Tuning uncertainty:** the best setting may be `MTP_NUM_HEADS=1`, `2`, or even a smaller loss weight than `0.1`.

## Validation

Commands run for this candidate:

```bash
python -m compileall candidates/202604041214_mtp-aux-head/train_gpt.py
python -m compileall train_gpt.py train_gpt_mlx.py data
```

Outcome summary:

- syntax compilation succeeded for the candidate script,
- baseline low-cost compilation succeeded for the existing root scripts and `data/`,
- a true runtime smoke test was **not feasible** in this environment because the runner did not have `torch` or `flash_attn_interface` installed, and the candidate targets CUDA execution in the real training environment.
