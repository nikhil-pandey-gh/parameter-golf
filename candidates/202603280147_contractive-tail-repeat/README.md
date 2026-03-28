# Contractive Tail Repeat

## Hypothesis

A **selective recurrent tail** can buy a little more effective depth on the strongest non-TTT 11-layer stack **without** paying the full throughput penalty of naive full-model recurrence.

The idea is to keep the proven `2026-03-22` architecture intact, then reuse only the deepest two blocks for one extra refinement pass. Each extra tail update is blended back into the current hidden state with a learned per-channel sigmoid step (`tail_step_logits`), so the repeated pass behaves like a **contractive residual refinement** rather than a second full-depth traversal.

I also carry over the latest repo-proven cheap activation win, **LeakyReLU(0.5)^2**, from the current best record.

## Why this is promising for this repository

The repository evidence points in two directions at once:

- the best non-TTT stack is already very strong (`2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`), so a new candidate should start there rather than from the root baseline,
- but both `2026-03-18_FP16Embed_WD3600` and `2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090` report that **naive depth recurrence** is too expensive under a 10-minute wallclock cap.

This candidate tries to split that difference:

- only the **deepest tail** is repeated,
- the repeated update is **contracted** with learned per-channel step sizes,
- the base stack still keeps the repo's proven wins: 11 layers, 3x MLP, BigramHash, SmearGate, XSA, partial RoPE, LN scaling, shared value embeddings, EMA, GPTQ-lite-style int6 export, and stride-64 sliding eval.

## Prior experiments that influenced this candidate

### Chosen base

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`
  - strongest non-TTT stack in the repo,
  - already combines the repo's most reusable train-time and export-time improvements.

### Direct carry-overs

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`
  - contributed the `LeakyReLU(0.5)^2` MLP activation,
  - but this candidate does **not** bring over legal TTT or Parameter Banking / Parallel Muon.

### Negative evidence this candidate is explicitly responding to

- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600`
  - notes that depth recurrence looked promising but needed more steps than the 10-minute budget allows.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090`
  - reports that doubling depth via full layer reuse was clearly negative.

The twist here is that I am **not** repeating the full network. I am only repeating the last two blocks once, and I contract that update with learned gates.

## External research that informed this candidate

This design was motivated by recent work on making recurrence useful by **restricting where it happens** and **stabilizing the repeated update**, instead of simply looping an entire transformer stack:

- **SCORE: Replacing Layer Stacking with Contractive Recurrent Depth** (2026)
  - motivated the contractive update view: `x <- x + alpha * (f(x) - x)`.
- **Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA** (2025)
  - reinforced that shared-depth models usually need some extra flexibility beyond naive reuse.
- **Intra-Layer Recurrence in Transformers for Language Modeling** (2025)
  - supports the idea that recurrence should be localized instead of applied indiscriminately to every layer.
- **Thinking Deeper, Not Longer: Depth-Recurrent Transformers for Compositional Generalization** (2026)
  - another reminder that parameter-efficient extra depth can work when recurrence is treated as iterative refinement.
- **ALBERT: A Lite BERT for Self-supervised Learning of Language Representations** (2020)
  - older precedent for cross-layer sharing as a parameter-efficiency tool.

## What changed versus the chosen base implementation

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate makes five deliberate changes:

1. **Selective recurrent tail**
   - new hyperparameters:
     - `TAIL_REPEAT_BLOCKS` (default `2`)
     - `TAIL_REPEAT_ITERS` (default `1`)
     - `TAIL_STEP_INIT` (default `0.35`)
   - after the normal 11-layer forward pass, the script reuses only the deepest `TAIL_REPEAT_BLOCKS` blocks for `TAIL_REPEAT_ITERS` extra refinement passes.

2. **Contractive tail blending**
   - each repeated tail block has a learned per-channel gate stored in `tail_step_logits`.
   - the repeated block output is mixed back as:
     - `x <- x + sigmoid(step) * (block(x) - x)`
   - this is meant to preserve throughput better than a naive full-strength repeat.

3. **Carry over LeakyReLU(0.5)^2**
   - the MLP now uses `F.leaky_relu(..., negative_slope=0.5).square()`.
   - this was the clearest zero-artifact-cost win in the latest repo record.

4. **Disable late-QAT toggling by default**
   - `LATE_QAT_THRESHOLD` now defaults to `0.0`.
   - the earlier late-QAT pattern was compile-sensitive, so this candidate keeps QAT honest and explicit: use `QAT_ENABLED=1` only if you want full-run QAT from step 0.

5. **Candidate-local default paths**
   - unlike the inherited record script, the default `DATA_PATH` and `TOKENIZER_PATH` now resolve from `__file__` back to the repo root,
   - so the script can be launched directly from this candidate directory without extra path overrides.

## How to run

From this directory:

```bash
RUN_ID=contractive_tail_repeat \
VOCAB_SIZE=1024 \
NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 \
MLP_MULT=3 BIGRAM_VOCAB_SIZE=2048 BIGRAM_DIM=128 \
XSA_LAST_N=4 ROPE_DIMS=16 LN_SCALE=1 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TAIL_REPEAT_BLOCKS=2 TAIL_REPEAT_ITERS=1 TAIL_STEP_INIT=0.35 \
QAT_ENABLED=0 LATE_QAT_THRESHOLD=0 \
EVAL_STRIDE=64 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

If you want to ablate the new idea away, set `TAIL_REPEAT_ITERS=0`.

## How to evaluate / what to look for

The script keeps the base stack's export path:

- EMA weights are applied before export.
- The model is exported with mixed int6/int8 compression.
- The script prints both roundtrip eval and stride-64 sliding-window eval.

The most important comparison is:

- **this candidate with** `TAIL_REPEAT_ITERS=1`
- versus the same script with `TAIL_REPEAT_ITERS=0`

That isolates whether the contractive recurrent tail is helping or just burning steps.

## Main expected risks and tradeoffs

- **Throughput risk**: even a small recurrent tail reduces steps inside the 600s cap.
- **No guarantee the gates stay contractive enough**: if the learned tail steps saturate high, this can still drift toward the naive recurrence failure mode.
- **Tail-only sharing may be too weak**: the idea might be directionally right but need more than one extra pass or a different repeated slice.
- **Late-QAT is intentionally de-emphasized**: this candidate prioritizes a clean recurrence test over a potentially brittle compile-time feature toggle.

## Validation

### Commands run

1. Syntax validation on the repo's existing Python entrypoints plus the new candidate:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603280147_contractive-tail-repeat/train_gpt.py
```

Outcome: **passed**.

2. Checked whether a local CPU smoke test was feasible in this runner:

```bash
python - <<'PY'
import importlib.util
print({'torch_installed': importlib.util.find_spec('torch') is not None})
PY
```

Outcome: `{'torch_installed': False}`.

### Validation notes

A true CPU-only model instantiation smoke test was **not feasible in this workflow runner** because `torch` is not installed here. I therefore limited validation to syntax checks and documented the missing dependency explicitly.
