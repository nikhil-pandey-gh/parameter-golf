# Lookahead MTP on the 11L EMA + GPTQ-lite Stack

## Hypothesis

Training the strongest clean 11-layer stack with a small **multi-token prediction (MTP)** auxiliary loss should improve sample efficiency inside the fixed 600-second budget, while keeping the final artifact budget intact because the auxiliary heads are **dropped before export**.

The core bet is that this repo is already close to a good quantized/training equilibrium on the `2026-03-22` stack, so the next high-upside move is to improve the shared trunk during training rather than rewriting the export path again.

## Why this is promising for this repository

- The best non-TTT training stack is already very strong: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`.
- Prior records repeatedly show that **quantization/export is the bottleneck**, not just raw training time. The 4-hour non-record baseline still lands at `1.2074` post-quant (`records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/README.md`).
- Several strong record codepaths already contain dormant MTP plumbing, but the logs show `mtp_num_heads:0`, so this is effectively **untried in practice** for this repo even though the mechanism exists in precursor code.
- Unlike TTT, MTP spends compute during training rather than evaluation, which better matches the repo's recent pattern of improving the base model first and keeping eval simpler unless the gains are clearly worth the budget.

## Prior records that influenced this candidate

- **Base implementation**: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - kept the 11L / MLP3x / XSA4 / partial-RoPE / LN-scale / EMA / GPTQ-lite skeleton
- **Immediate architectural predecessor**: `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - confirms the 11L partial-RoPE + LN-scale stack is a good base
- **Top overall result**: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - reinforces that the current frontier favors small, targeted changes layered onto an already-good 11L stack
- **Non-record 4-hour baseline**: `records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/`
  - motivates looking for better **training efficiency per step**, not just “train longer”

## External research that informed it

- **Multi-Token Prediction** — Gloeckle et al., *Better & Faster Large Language Models via Multi-token Prediction*, arXiv:2404.19737
  - argues that predicting multiple future tokens with auxiliary heads improves sample efficiency while leaving the shared trunk useful for standard next-token inference
- **SmoothQuant** — Xiao et al., arXiv:2211.10438
- **AWQ** — Lin et al., arXiv:2306.00978
- **QuaRot** — Croci et al., arXiv:2404.00456

The quantization papers were not copied directly here; they mainly strengthened the decision to keep the already-competitive weight-only export path from the base record and spend the experiment budget on a training-only objective instead.

## What changed versus the base implementation

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`:

1. **MTP is on by default**
   - `MTP_NUM_HEADS=2`
   - `MTP_LOSS_WEIGHT=0.15`

2. **Farther-future heads are down-weighted**
   - new `MTP_LOSS_DECAY` hyperparameter
   - the first lookahead target gets the largest weight, later heads decay geometrically

3. **Auxiliary heads warm-start from the main output projection**
   - new `MTP_INIT_FROM_MAIN=1`
   - this is intended to make the auxiliary task useful earlier in a short 10-minute run

4. **MTP heads are still excluded from export**
   - the artifact remains the same deployed model family as the base run

5. **Candidate-local usability improvements**
   - default dataset/tokenizer paths resolve from the repo root even when the script is run from this candidate directory
   - explicit FlashAttention -> GQA-aware SDPA fallback
   - `COMPILE=0/1` switch so lightweight local smoke checks are possible without changing the main H100 path

## How to run

From the repository root:

```bash
cd candidates/202603271946_lookahead-mtp
RUN_ID=lookahead_mtp \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

This candidate keeps the strong base defaults for:

- 11 layers, 512 dim, 8 heads / 4 KV heads
- MLP 3x
- XSA on the last 4 layers
- partial RoPE (`16/64`)
- LN scale
- EMA + GPTQ-lite + warmdown 3500
- BigramHash + shared value embeddings

Useful knobs to sweep first:

```bash
MTP_NUM_HEADS=1|2|3
MTP_LOSS_WEIGHT=0.10|0.15|0.20
MTP_LOSS_DECAY=0.3|0.5|0.7
MTP_INIT_FROM_MAIN=0|1
```

## How to evaluate

The script keeps the base record flow:

- train under the wallclock cap
- apply EMA
- export without `mtp_heads`
- quantize with the existing mixed int6 / GPTQ-lite path
- re-evaluate the roundtrip model
- optionally run stride-64 sliding-window eval

## Main risks / tradeoffs

- **Training-time overhead**: extra output heads cost compute and could reduce total steps enough to erase the sample-efficiency gain.
- **Auxiliary-loss mismatch**: the model is exported and evaluated as a standard next-token model, so too much MTP weight could over-regularize the late training phase.
- **Warm-start may be too sticky**: initializing MTP heads from the main projection may help short runs, but it could also reduce useful specialization.
- **CPU fallback is for smoke only**: the SDPA path is for validation convenience, not for matching H100 throughput.

## Validation

Commands run for this candidate:

```bash
python -m compileall candidates/202603271946_lookahead-mtp/train_gpt.py
```

Outcome:

- `compileall` **passed**

Attempted extra validation:

- I attempted a tiny CPU smoke test with a temporary SentencePiece model plus synthetic shards under `/tmp/gh-aw/agent/`.
- That smoke run was **not feasible in this container** because the runtime Python environment lacks `numpy`, `torch`, and `sentencepiece`, and this job environment is externally managed/offline, so I could not install the repo's declared dependencies during the workflow.

## Expected next experiments

1. Sweep `MTP_NUM_HEADS` and `MTP_LOSS_WEIGHT` on a 1xH100 or 8xH100 short run.
2. If MTP helps pre-quant but not post-quant, try tapering the auxiliary loss later in warmdown.
3. If MTP is neutral on this clean stack, retry it on the faster parameter-banking / parallel-Muon stack from `2026-03-23`.
