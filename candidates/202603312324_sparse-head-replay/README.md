# Sparse Head Replay

## Hypothesis

Static layer recurrence has already looked too expensive for this repository's 10-minute budget, but recent work on Sparse Growing Transformers suggests that **very selective depth reuse** can recover some of the benefit of extra depth without paying full block-level recurrence cost. This candidate tests a minimal adaptation of that idea on top of a strong local 11-layer recipe: replay only a **small subset of deep attention heads** inside the deepest layers, while keeping the rest of the model unchanged.

## Why this is promising here

The records in this repository already pushed most of the obvious artifact-side levers: int6/int5 quantization, GPTQ-lite, EMA/SWA, partial RoPE, XSA, BigramHash, SmearGate, and evaluation-side TTT. The review of prior runs also showed that **full recurrence / looped depth was a dead end under fixed wallclock** on earlier experiments, which makes a sparse, head-only replay path especially attractive: it keeps the parameter count almost unchanged and only adds compute where later records have already shown deep-layer specialization matters most.

## Main local influences

This candidate is primarily based on `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, which is the strongest pre-TTT / pre-parameter-banking stack in the repository and already includes the stable 11-layer EMA + GPTQ-lite + XSA + partial-RoPE recipe.

The specific design also borrows from:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`, which showed `LeakyReLU(0.5)^2` was a strong incremental gain on top of the mature 11-layer stack.
- `records/track_10min_16mb/2026-03-20_11L_EfficientPartialXSA_FA3_SWA120/README.md`, which showed that **deep-only targeted changes** can work better than touching every layer.
- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md` and `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md`, both of which called out layer recurrence as too slow or unhelpful in earlier forms.

There were no prior experiments under `candidates/` at the time this candidate was created.

## External research that informed it

- **Sparse Growing Transformer (Chen et al., 2026)** — https://arxiv.org/abs/2603.23998
  - Motivated the core hypothesis: increase effective depth by looping only a sparse subset of informative attention heads instead of replaying whole blocks.
- **Universal Transformer (Dehghani et al., 2019)** — https://arxiv.org/abs/1807.03819
  - Reinforced the broader idea that iterative refinement / recurrent depth can improve sequence modeling when the extra computation is placed carefully.
- **ALBERT (Lan et al., 2020)** — https://arxiv.org/abs/1909.11942
  - Helped frame the artifact-budget side of the problem: reusing computation is attractive precisely because it can raise effective depth without adding much model state.

## What changed versus the chosen base implementation

Relative to the 2026-03-22 11-layer EMA + GPTQ-lite record, this candidate makes four deliberate changes:

1. **SGT-inspired sparse head replay** in the deepest layers.
   - New defaults: `SPARSE_REPLAY_LAST_N=2`, `SPARSE_REPLAY_HEADS=2`, `SPARSE_REPLAY_SCALE=0.35`.
   - Only the deepest layers receive replay, and only a small query-head subset is replayed.
   - The replay path reuses the existing attention K/V state, projects the first-pass attended representation back into head-local query space, reapplies the same positional treatment, and then runs a second causal attention pass with learned per-head replay gains.
   - This keeps the change parameter-light and compute-local.

2. **LeakyReLU(0.5)^2** replaces plain `relu^2` in the MLP.
   - This is copied from the current best packaged record because it is already locally validated and orthogonal to the replay idea.

3. **Repo-root-relative dataset/tokenizer defaults**.
   - The script now resolves default `DATA_PATH` and `TOKENIZER_PATH` relative to the repository root, so it can be run directly from this candidate directory without extra path wiring.

4. **Optional CPU smoke path / attention fallback for validation**.
   - `SMOKE_TEST_ONLY=1` instantiates a tiny dummy model and runs a forward pass without requiring datasets or CUDA.
   - The attention path falls back to PyTorch SDPA if FlashAttention is unavailable, while still using FlashAttention when present on CUDA.

## How to run or evaluate it

From this candidate directory:

```bash
cd candidates/202603312324_sparse-head-replay
RUN_ID=sparse_head_replay \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful ablations:

- Disable replay entirely: `SPARSE_REPLAY_LAST_N=0`
- Keep replay but shrink it further: `SPARSE_REPLAY_HEADS=2 SPARSE_REPLAY_LAST_N=1`
- Test whether the gain is mostly from activation: `MLP_NEGATIVE_SLOPE=0.0 SPARSE_REPLAY_LAST_N=2`

Local smoke check (no dataset needed):

```bash
SMOKE_TEST_ONLY=1 \
VOCAB_SIZE=64 NUM_LAYERS=2 MODEL_DIM=64 NUM_HEADS=4 NUM_KV_HEADS=2 \
MLP_MULT=2 BIGRAM_VOCAB_SIZE=0 XSA_LAST_N=0 VE_ENABLED=0 \
TRAIN_SEQ_LEN=16 EVAL_SEQ_LEN=16 \
SPARSE_REPLAY_LAST_N=1 SPARSE_REPLAY_HEADS=2 SPARSE_REPLAY_SCALE=0.2 \
python train_gpt.py
```

## Validation

Commands run during candidate creation:

```bash
python3 -m venv --system-site-packages /tmp/gh-aw/agent/venv
/tmp/gh-aw/agent/venv/bin/pip install --quiet numpy sentencepiece torch
/tmp/gh-aw/agent/venv/bin/python -m compileall candidates/202603312324_sparse-head-replay/train_gpt.py
SMOKE_TEST_ONLY=1 VOCAB_SIZE=64 NUM_LAYERS=2 MODEL_DIM=64 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=2 \
BIGRAM_VOCAB_SIZE=0 XSA_LAST_N=0 VE_ENABLED=0 TRAIN_SEQ_LEN=16 EVAL_SEQ_LEN=16 \
SPARSE_REPLAY_LAST_N=1 SPARSE_REPLAY_HEADS=2 SPARSE_REPLAY_SCALE=0.2 \
/tmp/gh-aw/agent/venv/bin/python candidates/202603312324_sparse-head-replay/train_gpt.py
```

Outcomes:

- `compileall`: passed.
- CPU smoke test: passed in a temporary venv after installing `numpy`, `sentencepiece`, and `torch`; output was `smoke_ok loss:4.1668 logits_shape:(2, 8, 64)`.

## Main expected risks / tradeoffs

- Earlier repo experiments suggest **full recurrence can easily lose too many training steps**. Even this sparse replay path could still trade away enough throughput to negate its quality benefit.
- This implementation is intentionally lighter than full SGT: it uses fixed deep-layer placement plus learned per-head replay gains, not the paper's full entropy-driven progressive growth schedule.
- Replaying only attention heads may help semantic refinement, but it may also produce redundant context mixing if the selected heads collapse toward already-solved patterns.
- The CPU smoke path is only a correctness aid; the real question is whether the replay overhead stays small enough on H100s to preserve the strong 11-layer training recipe.
