# Mirror-Shared Depth + LeakyReLU^2

## Hypothesis

The strongest no-TTT stack in this repository already looks quantization-limited rather than obviously optimization-limited. My hypothesis is that **mirrored cross-depth weight sharing** can improve parameter efficiency and compressibility without throwing away the strong local/contextual tricks that already work here, as long as the shared core still gets **layer-specific adapters**.

Concretely, this candidate reuses only **6 shared transformer cores across 11 layers** with a mirrored assignment `0,1,2,3,4,5,4,3,2,1,0`, while keeping per-layer RMSNorms, residual mixes, layer scales, XSA placement, skip wiring, VE scaling, a learned depth token, and an identity-biased shared-delta gate. It also folds in the repo-proven **LeakyReLU(0.5)^2** MLP activation because that was the cleanest low-cost gain in the current best record.

## Why this is promising for this repository

This repo's record history shows a clear pattern:

- most wins came from better quantization/export plus modest architectural nudges,
- the best no-TTT stack is already very mature,
- and a naive recurrence attempt was previously negative on slower hardware.

That makes this a good place to try a **more careful recurrence / sharing variant**, not a fully new architecture. The key twist versus the failed recurrence result is that this candidate does **not** fully loop one anonymous block. Instead, it keeps the strong 2026-03-22 stack almost intact and only shares the expensive matrices, while preserving layer identity with lightweight per-layer controls.

## Prior records and experiments that influenced this candidate

### Chosen base implementation

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`

This was the strongest pre-TTT stack available in the repo: XSA4, EMA, GPTQ-lite clip search, partial RoPE, LN scaling, shared VE, BigramHash, SmearGate, tight warmdown, and mixed int6/int8 export.

### Additional repo influences

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - showed that `LeakyReLU(0.5)^2` was a real low-cost gain on top of a closely related architecture.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - reinforced that small zero-parameter architectural nudges can stack meaningfully in this regime.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
  - documented that **naive layer recurrence x2** was a negative result, which directly motivated keeping layer-specific adapters instead of doing full anonymous reuse.

## External research that informed the design

- **ALBERT** — Zhenzhong Lan et al., 2019. <https://arxiv.org/abs/1909.11942>
  - Strong evidence that cross-layer parameter sharing can preserve much of the benefit of deeper Transformers while sharply reducing parameter count.
- **Universal Transformer** — Mostafa Dehghani et al., 2018. <https://arxiv.org/abs/1807.03819>
  - Motivated treating depth as iterative refinement and keeping explicit step identity instead of repeating an indistinguishable block.
- **Thinking Deeper, Not Longer: Depth-Recurrent Transformers for Compositional Generalization** — Hung-Hsuan Chen, 2026. <https://arxiv.org/abs/2603.21676>
  - Motivated identity-biased recurrent updates and stable deep reuse.
- **SCORE: Replacing Layer Stacking with Contractive Recurrent Depth** — Guillaume Godin, 2026. <https://arxiv.org/abs/2603.10544>
  - Motivated the explicit contractive-style `shared_delta` gate that scales the shared-core update instead of forcing every reused block application to have full residual strength.

## What changed versus the chosen base implementation

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate:

1. Replaces per-layer block matrices with **mirrored shared cores**.
   - 11 logical layers now reuse 6 shared transformer cores.
   - Each logical layer still has its own norms, `attn_scale`, `mlp_scale`, `resid_mix`, XSA flag, and VE routing.

2. Adds two lightweight layer-identity mechanisms.
   - `depth_token`: a learned per-layer additive token before the shared core.
   - `shared_delta`: a learned per-layer gate on the shared core's residual delta.

3. Switches the MLP activation to **LeakyReLU(0.5)^2**.

4. Adds a **FlashAttention -> SDPA fallback**.
   - If `flash_attn_interface` is unavailable, the module can still import and use PyTorch SDPA.
   - This makes local smoke-imports and CPU-side structural checks much easier.

5. Fixes the default data/tokenizer paths so the script can be run **from inside this candidate directory**.
   - Defaults now resolve relative to the repository root via `Path(__file__)`.

Everything else is intentionally kept close to the strong 2026-03-22 stack: BigramHash, SmearGate, VE, XSA on the deepest layers, partial RoPE, LN scaling, EMA, tight SWA, late QAT, and GPTQ-lite mixed export.

## How to run or evaluate

From this candidate directory:

```bash
cd candidates/202603262248_mirror-shared-depth

NUM_LAYERS=11 \
BIGRAM_VOCAB_SIZE=2048 \
XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
SHARED_DEPTH=1 SHARED_DELTA_INIT=0.75 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

If your environment does not have `flash_attn_interface`, the script falls back to PyTorch SDPA automatically.

## Main expected risks and tradeoffs

- **Too much tying:** even with per-layer adapters, mirrored sharing may still overconstrain the model versus the untied 11-layer base.
- **Compute is unchanged:** this saves artifact/parameter budget, not training FLOPs. If the shared inductive bias is wrong, there is no free speed rescue.
- **Quantization interaction is uncertain:** the shared cores may compress better due to repeated structure, but they may also become more outlier-heavy if all depth roles fight through one matrix set.
- **XSA/VE coupling may shift:** those tricks were tuned for untied layers, so their best settings under shared depth may differ.

## Validation

### Commands run

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603262248_mirror-shared-depth/train_gpt.py
python -m compileall candidates/202603262248_mirror-shared-depth/train_gpt.py
python - <<'PY'
# attempted import + CPU forward smoke
# blocked in this runner because torch is not installed
PY
```

### Outcomes

- `compileall` **passed** for the repository baseline scripts and this candidate script.
- A lightweight import/forward smoke test was **attempted but not feasible in this runner** because the available Python environment does not have `torch` installed (`ModuleNotFoundError: No module named 'torch'`).
- No dataset-backed training smoke run was attempted for the same reason.

Once the normal challenge dependencies are installed, the next check I would run is a single-process structural smoke import plus a tiny one-step CUDA launch.
