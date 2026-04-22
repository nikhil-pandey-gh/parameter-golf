# Shared Tail Reinvest

## Hypothesis

The strongest clean 11-layer stack in this repo already treats the deepest layers as special: XSA only turns on late, partial RoPE is late-friendly, and the best non-TTT record already shares a value-embedding table across late layers. This candidate pushes that idea one step further by **sharing only the late decoder/XSA tail at the weight level**, then **reinvesting the saved bytes into more precision exactly where those reused weights matter most**.

The concrete hypothesis is:

1. the last 4 logical layers are redundant enough to share as 2 physical blocks,
2. the main failure mode of naive recurrence in this repo was extra compute / fewer optimizer steps, not reuse itself,
3. tiny **timestep-conditioned gains + step embeddings** are enough to preserve layer specialization across the reused tail, and
4. the saved artifact budget is best spent on **keeping the shared tail in int8** plus a slightly larger lexical side-channel.

## Why this is promising here

- The best non-TTT record (`2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`) already uses:
  - late-only XSA,
  - shared value embeddings on layers 9-10,
  - GPTQ-lite mixed quantization,
  - EMA/SWA,
  - a 15.55 MB artifact, leaving some headroom but still paying a real post-quantization gap.
- The best overall record (`2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`) shows that even very small improvements still matter near the top.
- The repo's negative recurrence evidence is specifically about **extra repeated compute under a fixed wallclock**. This candidate keeps the **same logical depth and same forward-pass count**; it only reuses parameters in the deepest tail.

## Prior records that influenced this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- **Late-layer lexical/value reuse precedent:** same base record, especially shared value embeddings on layers 9-10.
- **Bigram-capacity evidence:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md` ablates a bigger BigramHash positively.
- **Recurrence cautionary evidence:** `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md` reports naive layer recurrence as a clear negative because it reduced usable training steps.

## External research used

- **ALBERT** (Lan et al., 2019, arXiv:1909.11942): cross-layer parameter sharing can improve parameter efficiency without destroying model quality.
- **Universal Transformer** (Dehghani et al., 2018, arXiv:1807.03819): repeated Transformer computation can add expressivity without adding distinct parameters.
- **Looped Transformers for Length Generalization** (Fan et al., 2024/2025, arXiv:2409.15647): looped/shared blocks can work well when the task benefits from iterative computation.
- **On Expressive Power of Looped Transformers** (Xu & Sato, 2024/2025, arXiv:2410.01405): timestep-conditioned scaling helps recover expressivity lost by weight sharing / looping.
- **Loop, Think, & Generalize** (Kohli et al., 2026, arXiv:2604.07822): recurrent-depth Transformers can help compositional generalization, but too much recurrence can also hurt ("overthinking"), which supports a **partial** rather than global reuse design here.

## What changed vs the chosen base

Base: `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

1. **Shared late tail:** logical layers `7,8,9,10` are executed through only **2 physical blocks** (`SHARED_TAIL_PERIOD=2`).
2. **Timestep-conditioned specialization:** each logical shared-tail layer gets:
   - a learned `shared_tail_steps[layer]` input offset,
   - a learned `shared_tail_gains[layer]` residual gain.
3. **Precision reinvestment:** `shared_tail_blocks.*` are exported with **int8** quantization instead of the normal int6 path, so the reused weights keep more fidelity.
4. **Late lexical reinvestment:** `BIGRAM_VOCAB_SIZE` default increases from `2048` to `3072`.
5. **Broader late VE use:** `VE_LAYERS` default expands from `9,10` to `7,8,9,10`, still using the same shared VE table plus per-layer scales.
6. **Candidate-folder portability:** default dataset/tokenizer paths resolve relative to the repository root, so the script can be run directly from this candidate directory.

Everything else intentionally stays close to the base record: 11L/512d, XSA on the last 4 logical layers, EMA + tight SWA, GPTQ-lite mixed export, partial RoPE, SmearGate, and the same training/eval structure.

## How to run

From the repository root:

```bash
cd candidates/202604222148_shared-tail-reinvest
RUN_ID=shared_tail_reinvest \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful overrides:

```bash
# Disable the shared-tail idea for ablation
SHARED_TAIL_LAYERS= SHARED_TAIL_PERIOD=0

# Change the precision reinvestment rule
PRECISION_REINVEST_INT8_NAME_PATTERNS=

# By default the shared-tail and VE layer lists track NUM_LAYERS automatically.
# Override them only if you want a custom mapping.
NUM_LAYERS=9
```

## Evaluation / validation

Recorded in this workflow:

```bash
python -m compileall candidates/202604222148_shared-tail-reinvest/train_gpt.py
python -m py_compile candidates/202604222148_shared-tail-reinvest/train_gpt.py
```

Outcome:

- `compileall`: **passed**
- `py_compile`: **passed**

Attempted but not feasible in this workflow container:

```bash
python - <<'PY'
# import candidate module and run a tiny CPU forward pass with a FlashAttention fallback
PY
```

That smoke test could not run here because the container's Python environment does not have `torch` installed (`ModuleNotFoundError: No module named 'torch'`). The script also depends on CUDA/FlashAttention for any real training run.

## Main expected risks / tradeoffs

- **Underfitting risk:** even late-layer sharing may remove too much capacity if the deepest layers are not redundant enough.
- **Quantization tradeoff:** keeping the reused tail in int8 spends bytes that could instead go to a larger lexical module or more fp16 passthrough.
- **Mapping sensitivity:** the default shared-tail and VE mappings scale with `NUM_LAYERS`, but custom `SHARED_TAIL_LAYERS` choices still need empirical tuning.
- **Default-shape coupling:** the defaults are tuned for the 11-layer stack, so changing `NUM_LAYERS` is now runnable but still may not be optimal.
