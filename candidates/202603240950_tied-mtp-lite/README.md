# 202603240950_tied-mtp-lite

## Hypothesis

The strongest unexplored lever in this repo is a **training-only auxiliary objective** rather than another permanent architecture change. Recent multi-token-prediction style training ideas suggest that asking hidden states to predict a slightly farther-future token can improve sample efficiency and representation quality. In this repository, that is especially attractive because the challenge is bottlenecked by **artifact size** and **10-minute wallclock**, not just raw parameter count.

This candidate tests a lightweight version of that idea: keep the current best 11-layer EMA/GPTQ-lite/XSA/VE architecture, but add a **single 2-step-ahead auxiliary loss** during training using **tied output embeddings**, **tiny low-rank adapters**, and **strided positions** so the extra supervision is cheap. The auxiliary weights are excluded from export, so the shipped artifact stays focused on the main next-token model.

## Why this is promising for this repository

The records already show that:

- recurrence / depth reuse was a bad wallclock trade under tight training budgets,
- permanent architectural additions must justify every byte under the 16MB cap,
- the best line of progress comes from stacking small, low-risk improvements on the strongest 11-layer base,
- the current top record already contains dormant MTP plumbing but leaves it disabled.

That makes an **export-free, sample-efficiency-oriented** change a good next bet. Compared with trying a heavier recurrent block, longer context, or broader quantization rewrite, this candidate aims for a cleaner trade: modest training overhead in exchange for better hidden-state supervision, while leaving inference and submission size essentially unchanged.

## Prior records that influenced this candidate

Primary base:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`

Most relevant influences:

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - showed the strong 11-layer XSA/EMA family and still carried disabled MTP support.
- `records/track_10min_16mb/2026-03-20_11L_EfficientPartialXSA_FA3_SWA120/`
  - showed that the repo rewards efficient versions of ideas, not just more expressive ones.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
  - documented that recurrent depth reuse lost badly because step-count loss outweighed parameter savings.

There was **no existing `candidates/` directory** at the time this candidate was created, so this is the first candidate iteration in that tree.

## External research that informed it

This candidate is informed by two broad lines of recent compact-model / efficient-LLM research:

- **Multi-token prediction / future-token auxiliary objectives**, which aim to improve sample efficiency by forcing hidden states to model a slightly longer horizon than plain next-token loss alone.
- **Low-rank adapter / parameter-sharing approaches**, which make auxiliary branches cheap enough to add during training without turning them into permanent inference-time baggage.

Important note: live outbound research fetches were blocked by the workflow firewall in this run, so I could not retrieve fresh paper pages during execution. The design here is therefore based on prior literature knowledge and repository evidence, not on newly fetched web content from this workflow run.

## What changed versus the chosen base implementation

Base file:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

1. **Enable MTP by default**
   - `MTP_NUM_HEADS` default changed from `0` to `1`.
   - This adds one auxiliary head that predicts the token after the standard next-token target.

2. **Replace full auxiliary vocab heads with tied low-rank future adapters**
   - The base code had dormant full `mtp_heads` projections.
   - This candidate swaps them for `TiedFutureAdapter`, a tiny low-rank residual adapter that reuses the main tied embedding matrix for logits.
   - This sharply reduces training-only auxiliary parameters.

3. **Strided auxiliary supervision**
   - New `MTP_STRIDE` default is `4`.
   - The future-token loss is computed on every fourth eligible position instead of every token, which keeps extra compute bounded.

4. **New tunables**
   - `MTP_RANK=32`
   - `MTP_LOSS_WEIGHT=0.12`
   - `MTP_STRIDE=4`

5. **Export remains clean**
   - Training-only MTP adapter weights are excluded from export, just like the old dormant MTP weights would have been.

6. **CPU smoke-test path**
   - Added `SMOKE_TEST=1` mode so the candidate can instantiate, run forward/backward, quantize, dequantize, and reload on CPU without dataset shards or GPUs.

7. **Fallback attention path for smoke mode**
   - If `flash_attn_interface` is unavailable, the script falls back to PyTorch SDPA.
   - The intended fast path on benchmark hardware is still FlashAttention when present.

8. **Candidate-directory-friendly defaults**
   - Default `DATA_PATH` and `TOKENIZER_PATH` now resolve relative to the repository root inferred from `__file__`, so `train_gpt.py` can be run from inside this candidate directory.

## How to run or evaluate it

From this directory:

```bash
cd candidates/202603240950_tied-mtp-lite
python train_gpt.py
```

Useful overrides:

```bash
cd candidates/202603240950_tied-mtp-lite
SEED=1337 MTP_NUM_HEADS=1 MTP_RANK=32 MTP_STRIDE=4 MTP_LOSS_WEIGHT=0.12 python train_gpt.py
```

CPU-only smoke check:

```bash
cd candidates/202603240950_tied-mtp-lite
SMOKE_TEST=1 python train_gpt.py
```

## Validation run for this candidate

Using a temporary venv at `/tmp/gh-aw/agent/pg-venv`:

```bash
/tmp/gh-aw/agent/pg-venv/bin/python -m compileall \
  /home/runner/work/parameter-golf/parameter-golf/train_gpt.py \
  /home/runner/work/parameter-golf/parameter-golf/train_gpt_mlx.py \
  /home/runner/work/parameter-golf/parameter-golf/data \
  /home/runner/work/parameter-golf/parameter-golf/candidates/202603240950_tied-mtp-lite/train_gpt.py
```

Outcome: **passed**

```bash
cd /home/runner/work/parameter-golf/parameter-golf/candidates/202603240950_tied-mtp-lite
SMOKE_TEST=1 /tmp/gh-aw/agent/pg-venv/bin/python train_gpt.py
```

Observed output:

```text
smoke_test:ok train_loss:7.7748 eval_loss:6.9564 export_params:26993756
```

## Main expected risks and tradeoffs

- **Training-time overhead:** even strided 2-step supervision still adds compute, so a real 8xH100 run may lose some steps.
- **Objective interference:** future-token supervision can help early representation learning, but it can also fight the final next-token objective if weighted too strongly.
- **Quantization interaction:** the auxiliary branch is not exported, so any gains must transfer into the shared trunk rather than survive in the disposable adapter.
- **Need for real benchmark validation:** this candidate smoke-tests correctly, but the actual BPB/value tradeoff still needs a genuine GPU run.
- **Fallback attention is for validation convenience only:** the SDPA fallback is there so the candidate can self-check on CPU; the target training path remains FlashAttention-backed on proper hardware.
