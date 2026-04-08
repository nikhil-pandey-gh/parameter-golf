# Training-Only MTP Auxiliary Heads on the 03-23 LeakyReLU² + TTT Stack

## Hypothesis

A small training-only multi-token-prediction (MTP) loss should improve sample efficiency in the repository's fixed 10-minute training budget without increasing the exported artifact size. The auxiliary heads are excluded from export, so any gain should come from a better-trained shared trunk rather than from spending more bytes.

## Why this is promising for this repository

Repository review shows a consistent pattern: the best improvements here come from ideas that preserve the strong 11-layer stack while improving either evaluation efficiency or training efficiency. The current checked-in frontier moved through sliding-window evaluation, quantization/export refinement, XSA on late layers, partial RoPE, LN scale, EMA/SWA, and finally LeakyReLU(0.5)^2 plus legal score-first TTT.

The strongest checked-in base is:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`

That record already carries dormant `MTP_NUM_HEADS` / `MTP_LOSS_WEIGHT` plumbing in code, but the published run configuration keeps `MTP_NUM_HEADS=0`, so the idea is still absent from the checked-in experiment history. This candidate turns that path into a real training signal while keeping the rest of the winning stack intact.

I explicitly did **not** choose recursive/shared-depth ideas for this run. External research made them tempting, but the repository's own non-record sweep already reports depth/layer recurrence as a negative result under the 600-second budget.

## Prior records and candidate history that informed this

- `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`: best current checked-in score and direct code base for this candidate.
- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`: strongest cleaner non-TTT base; reinforced that post-training/export refinements still matter.
- `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`: showed partial RoPE + LN scale helped, while the late-QAT flag in that version was effectively a no-op.
- `2026-03-19_SlidingWindowEval`: established that evaluation-aware changes can be large enough to dominate small architectural tweaks.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090`: useful mainly as a warning that layer/depth recurrence under a fixed wall-clock cap can backfire badly.

There was **no checked-in `candidates/` directory** in this checkout, so there were no prior in-tree candidate folders to reuse or avoid.

## External research

Most relevant primary source:

- Fabian Gloeckle et al., **Better & Faster Large Language Models via Multi-token Prediction** — <https://arxiv.org/abs/2404.19737>

Why it matters here:

- It argues that predicting multiple future tokens from a shared trunk improves sample efficiency.
- It reports that the method can help small algorithmic tasks and induction-head development, which is attractive for a tiny model trained under a strict time cap.
- It does so without requiring the auxiliary heads to survive into the final inference/export path.

I also compared this direction against other externally motivated ideas before choosing it:

- ALBERT / Universal Transformer style sharing: attractive on bytes, but mismatched with the repo's negative recurrence history.
- QuaRot / SpinQuant style rotation-aware PTQ: promising, but a larger and riskier code change for this run.

## What changed vs the chosen base implementation

Base:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. **Enable MTP in the candidate run configuration** with `MTP_NUM_HEADS=1` and `MTP_LOSS_WEIGHT=0.1` while keeping the script default at `MTP_NUM_HEADS=0` so exported checkpoints stay reloadable without auxiliary heads.
2. **Make the auxiliary heads real trainable parameters** by wiring `mtp_heads` into their own AdamW optimizer group.
3. **Seed the auxiliary heads from the model's main vocab geometry** instead of leaving them effectively cold-started.
4. **Keep export size unchanged in principle** by continuing to drop `mtp_heads.*` from the serialized/quantized checkpoint.
5. **Add a safe attention fallback**: if `flash_attn_interface` is unavailable, the model falls back to PyTorch `scaled_dot_product_attention`, which makes lightweight local smoke checks practical.

The rest of the stack stays intentionally close to the 03-23 record: LeakyReLU(0.5)^2, parameter banking + Parallel Muon, XSA on late layers, partial RoPE, LN scale, VE128, GPTQ-lite int6+lzma export, and legal score-first TTT.

## How to run

From this directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.1 \
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

EMA remains baked into this copied 03-23 stack with decay `0.997`; the exposed late-QAT knob in this candidate is `LATE_QAT_THRESHOLD`.
The candidate script resolves its default dataset and tokenizer paths relative to the repository root, so running it from inside this candidate directory works without extra path overrides.

## Validation

The exact commands and outcomes recorded for this workflow are below.

## Validation outcomes

The runner did not have Python ML dependencies preinstalled, so the runtime smoke check used a temporary virtual environment under `/tmp/gh-aw/agent/pg-venv`.

1. `python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604080741_mtp-aux-heads/train_gpt.py`  
   **Passed.**
2. `. /tmp/gh-aw/agent/pg-venv/bin/activate && python - <<'PY' ... PY` with a tiny CPU-only `GPT(...)` instantiation, forward/backward pass, and an `AdamW` step on the auxiliary MTP head  
   **Passed** with:
   - `smoke_loss:4.604989`
   - `qo_grad:0.275553`
   - `mtp_grad:0.144082`
   - `mtp_delta:0.063995`

That smoke test confirms the SDPA fallback imports cleanly without `flash_attn_interface`, the candidate model executes a full forward/backward pass on CPU, gradients flow through both the shared banked trunk and the auxiliary MTP head, and the auxiliary head actually updates under optimizer step.

## Main risks and tradeoffs

- Even one auxiliary head adds training-time compute, so step throughput could drop enough to erase the sample-efficiency gain.
- The best loss weight is unknown; too much auxiliary pressure could hurt the main next-token objective.
- Any pre-quant improvement still has to survive the int6 export path and then the legal TTT evaluation.
- Because this candidate keeps the strong 03-23 stack intact, measured gains may be small and require careful comparison against both pre-TTT and post-TTT metrics.
