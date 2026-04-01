# Shared-Depth Adapters

## Hypothesis

The repo has squeezed a lot out of evaluation, quantization, and a handful of cheap architectural biases, but it has not yet found a good **parameter-sharing** recipe for the 16MB regime. A prior non-record attempt at plain layer recurrence got worse because it effectively doubled compute and cut the number of optimizer steps in the fixed wall-clock budget. This candidate tests a more surgical version: **keep the same virtual depth, share only the heavy block weights, and give every virtual layer its own lightweight adapter tensors** so repeated blocks do not collapse into identical behavior.

In short: reuse the expensive matrices, not the whole layer identity.

## Why this looks promising here

Three patterns stand out in the repository review:

- The leaderboard keeps winning by getting **more useful model capacity per byte**, first through mixed-precision export, then by using those saved bytes for 10-11 layers, 3x MLPs, BigramHash, XSA, and similar cheap biases.
- The current best stacks are already strong on eval and post-training quantization, so a fresh gain is more likely to come from a **new architectural compression lever** than yet another tiny quantization tweak.
- The one recurrence failure in `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md` was naive full-layer reuse with higher compute cost. This candidate explicitly addresses that failure mode with **front-loaded sharing plus per-layer adapters**, while keeping total virtual depth fixed.

## Prior repo work that influenced this candidate

- Root baseline: `/train_gpt.py`
  - Supplies the clean U-Net GPT baseline, tokenizer-agnostic BPB evaluation, and int8+zlib roundtrip export.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md`
  - Motivated the inclusion of **Partial RoPE (16/64 dims)** and **layer scaling**, both zero-parameter or near-zero-cost ideas that stacked cleanly there.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`
  - Motivated switching the MLP nonlinearity from `relu^2` to **LeakyReLU(0.5)^2**, which showed a strong, cheap gain on the current top stack.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md`
  - Supplies the key dead-end to avoid: naive layer recurrence that spends too much wall-clock on extra compute and loses on steps.

## External research informing the idea

- **ALBERT: A Lite BERT for Self-supervised Learning of Language Representations** (`arXiv:1909.11942`)
  - Shows that cross-layer parameter sharing can preserve strong performance while cutting parameter count materially.
- **Universal Transformers** (`arXiv:1807.03819`)
  - Provides the core intuition that repeated depth can be useful when the model keeps enough per-step identity.
- **Intra-Layer Recurrence in Transformers for Language Modeling** (`arXiv:2505.01855`)
  - Argues that recurrence budget is often most useful in **earlier layers**, which motivated the default **front-loaded** sharing schedule here.
- **Basis Sharing: Cross-Layer Parameter Sharing for Large Language Model Compression** (`arXiv:2410.03765`)
  - Reinforces that cross-layer sharing remains effective under aggressive compression, even though that paper focuses on SVD-style post-training compression.

## What changed versus the chosen base implementation

Chosen base: the repository root `train_gpt.py`.

This candidate keeps the baseline training/export pipeline but changes the model body:

- Adds **shared heavy blocks** (`SHARED_BLOCKS`, default `8`) for `NUM_LAYERS` virtual layers (default `11`).
- Adds **per-layer adapters** with:
  - `depth_bias`
  - `resid_mix`
  - `attn_scale`
  - `mlp_scale`
- Uses a **front-loaded layer map** by default, so earlier virtual layers are the ones repeated first.
- Adds **Partial RoPE** through `ROPE_DIMS=16`.
- Adds optional **layer scaling** (`1/sqrt(layer_idx+1)`) outside the shared blocks.
- Switches the MLP default to **LeakyReLU(0.5)^2**.
- Adds a **CPU smoke-test mode** with `SMOKE_TEST=1` so the candidate can be sanity-checked without a GPU run.

## How to run

From this candidate directory:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script resolves its default `DATA_PATH` and `TOKENIZER_PATH` from the repository root, so this command works from inside the candidate directory without extra path overrides.

Recommended knobs to keep the candidate in its intended regime:

```bash
NUM_LAYERS=11 SHARED_BLOCKS=8 LAYER_SHARE_SCHEME=frontloaded \
MLP_MULT=3 ROPE_DIMS=16 LAYER_SCALE=1 \
MLP_ACTIVATION=leaky_relu2 LEAKY_RELU_SLOPE=0.5 \
TRAIN_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=786432 VAL_BATCH_SIZE=786432 \
WARMDOWN_ITERS=3000 ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

For a fast local CPU sanity check:

```bash
SMOKE_TEST=1 SMOKE_BATCH_SEQS=2 SMOKE_SEQ_LEN=32 python train_gpt.py
```

## Validation run for this candidate

Commands executed while preparing this candidate:

```bash
python -m compileall candidates/202604010548_shared-depth-adapters/train_gpt.py
SMOKE_TEST=1 SMOKE_BATCH_SEQS=2 SMOKE_SEQ_LEN=32 python candidates/202604010548_shared-depth-adapters/train_gpt.py
```

Observed outcomes:

- `python -m compileall ...` completed successfully.
- Smoke test output (now includes one real CPU optimizer step using the same shared-block / layer-adapter parameter grouping as the training path):

```text
smoke_test:ok loss:6.9420 post_step_loss:11.5246 roundtrip_loss:11.1147 shared_blocks:8/11 layer_map:[0, 0, 1, 1, 2, 2, 3, 4, 5, 6, 7] int8_payload_bytes:19581184
```

- Additional size sanity check at random initialization:

```text
params 19429440
payload_bytes 19581184
torch_raw_bytes 19627985
zlib_bytes 5440171
code_bytes 53968
total_bytes 5494139
```

That last check is **not** a trained-model score, but it does confirm that the shared-depth layout has comfortable structural headroom under the 16MB artifact cap before any serious tuning.

The smoke path still does **not** cover CUDA-only pieces such as `torch.compile`, DDP, or the full data loader, so the next real check should still be a minimal GPU launch on actual shards.

## Main risks and tradeoffs

- **Wall-clock risk:** even with shared parameters, virtual depth still costs compute. This candidate avoids the worst dead end by not doubling depth, but step time remains the first thing to watch.
- **Under-specialization risk:** sharing too aggressively can flatten layer roles. The per-layer adapters are meant to recover enough individuality, but the best `SHARED_BLOCKS` value is still an open question.
- **Quantization interaction risk:** the new small control tensors are kept out of the heavy quantization path, but the final post-quant behavior still needs a real GPU run to judge.
- **Stacking question:** if this idea works, the next step is probably porting it onto the current stronger 11L/XSA/EMA/QAT stack rather than keeping it on the cleaner baseline forever.
