# Candidate: Stage-Shared Depth

## Hypothesis

The current record trajectory strongly rewards deeper 10-11 layer stacks, better compression-aware training, and a richer lexical front-end, but the artifact budget still pays heavily for fully unique blocks. This candidate tests whether **stage-shared transformer cores** can preserve the benefits of an 11-layer logical network while cutting repeated block bytes, then use **small per-layer contractive adapters** to keep each logical layer distinct and stable.

Concretely, the model keeps the same 11 logical layer applications as the strong 2026-03-22 backbone, but replaces 11 unique heavy blocks with **3 shared encoder cores + 3 shared decoder cores**. Each logical layer still has its own:

- residual mixing vector,
- attention scale vector,
- MLP scale vector,
- contractive depth-step scalar.

The contractive update is inspired by recent recurrent-depth work:

```python
x_next = x_in + sigmoid(depth_step_logit) * (proposal - x_in)
```

This is intended to make shared-weight reuse less brittle than naive "repeat the same block twice" recurrence.

## Why this is promising for this repository

Repository evidence points in a clear direction:

- `records/track_10min_16mb/2026-03-20_11L_EfficientPartialXSA_FA3_SWA120/README.md` showed that moving from the earlier 9-10 layer family to an 11-layer stack was a major win.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md` showed that low-cost architectural control tricks like Partial RoPE and LN scaling still had headroom on top of that 11-layer backbone.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md` is the strongest clean train/export backbone before eval-time TTT.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md` showed that LeakyReLU(0.5)^2 is a cheap, high-value activation swap.

There is also an important cautionary negative result:

- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md` reported that a naive "layer recurrence x2" experiment was much worse because it doubled compute and reduced optimizer steps under a fixed wallclock budget.

This candidate explicitly avoids that failure mode. It does **not** increase logical depth beyond 11 layer applications. Instead, it keeps compute in the same rough regime as the successful 11-layer family and only changes how many unique heavy weights are stored.

## Prior runs that influenced this candidate

Primary base:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`

Direct influences carried forward:

- 11 logical layers, 512d, 8H/4KV, 3x MLP
- SmearGate + BigramHash front-end
- Partial RoPE (`ROPE_DIMS=16`)
- LN scale factor `1/sqrt(layer_idx+1)`
- value embedding reinjection (`VE_DIM=128`, layers `9,10`)
- GPTQ-lite mixed int6/int8 export
- EMA + tight SWA export path

Additional carryover:

- LeakyReLU(0.5)^2 activation from `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
- Slightly larger `BigramHash` table (`3072`) motivated by the same 2026-03-23 run's positive bigram-size ablation

No prior `candidates/` directory existed in this repository when this candidate was created.

## External research that informed it

Primary sources:

- **ALBERT** (`arXiv:1909.11942`): cross-layer parameter sharing can reduce model size substantially while retaining depth benefits.
- **Universal Transformer** (`arXiv:1807.03819`): iterative/shared-depth processing can improve sequence modeling by reusing a recurrent reasoning core.
- **Thinking Deeper, Not Longer: Depth-Recurrent Transformers for Compositional Generalization** (`arXiv:2603.21676`): recent evidence that shared-weight depth can trade parameter count for reasoning depth when stabilized with identity-biased updates.
- **SCORE: Replacing Layer Stacking with Contractive Recurrent Depth** (`arXiv:2603.10544`): motivates the contractive residual form used here, where a learned step size controls how aggressively each logical layer moves away from its input state.

I also reviewed recent quantization work such as **QuaRot** (`arXiv:2404.00456`), **SpinQuant** (`arXiv:2405.16406`), **QuIP#** (`arXiv:2402.04396`), **AWQ** (`arXiv:2306.00978`), and optimizer/quantization interaction results (`arXiv:2509.23500`). Those are promising, but they would require a broader quantization-specific rewrite than felt justified for a minimal next candidate in this repo. The present candidate stays closer to the repository's proven train/export pattern and changes the architecture instead.

## What changed versus the chosen base implementation

Compared with `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`:

1. **Stage-shared depth**
   - Replaced 11 unique heavy blocks with:
     - 3 shared encoder cores
     - 3 shared decoder cores
   - Default logical-to-shared mapping is stage-wise:
     - encoder: `[0, 0, 1, 1, 2]`
     - decoder: `[0, 0, 1, 1, 2, 2]`

2. **Per-layer contractive adapters**
   - Added per-logical-layer `depth_step_logit` and update rule
   - Kept per-logical-layer `resid_mix`, `attn_scale`, and `mlp_scale`
   - This preserves layer identity even though the heavy attention/MLP weights are shared within a stage

3. **LeakyReLU(0.5)^2 MLP**
   - Swapped ReLU^2 for LeakyReLU(0.5)^2, following the strongest current record

4. **Slightly larger lexical table**
   - Default `BIGRAM_VOCAB_SIZE` increased from `2048` to `3072`

5. **CPU-safe attention fallback for validation**
   - If `flash_attn_interface` is unavailable or tensors are not on CUDA, attention falls back to `torch.nn.functional.scaled_dot_product_attention`
   - This does not change the intended CUDA path, but it makes lightweight local smoke validation more realistic in non-H100 environments

## How to run or evaluate it

From the candidate directory:

```bash
RUN_ID=stage_shared_depth \
NUM_LAYERS=11 SHARED_ENCODER_BLOCKS=3 SHARED_DECODER_BLOCKS=3 \
DEPTH_STEP_INIT=0.65 LEAKY_RELU_SLOPE=0.5 \
BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=128 \
XSA_LAST_N=4 ROPE_DIMS=16 LN_SCALE=1 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script inherits the 2026-03-22 style export/eval path:

- train under the 600s wallclock cap,
- apply EMA before export,
- quantize with GPTQ-lite-style int6 block weights + int8 fallback,
- run standard roundtrip validation,
- run sliding-window eval (`stride=64`).

## Validation run locally

Commands executed in this workflow runner:

```bash
python -m compileall candidates/202603292245_stage-shared-depth/train_gpt.py
```

Outcome:

- `compileall` **succeeded**.

Attempted additional smoke validation:

```bash
python - <<'PY'
# import candidate module, instantiate a tiny GPT on CPU,
# run a forward pass, then exercise quantization roundtrip helpers
PY
```

Outcome:

- Could **not** complete in this runner because the local Python environment does not have `torch` installed (`ModuleNotFoundError: No module named 'torch'`).
- I did **not** install a heavyweight PyTorch runtime into the shared workflow environment just for this candidate check.
- The candidate script therefore has syntax validation here, but not an executed forward-pass smoke test in this environment.

## Main expected risks / tradeoffs

- **Capacity risk:** sharing block weights across stage-local layers may hurt specialization enough to offset the byte savings.
- **Optimization risk:** even with contractive depth steps, shared cores may train less robustly than the fully unique 11-layer baseline.
- **Quantization interaction risk:** the shared-depth architecture may change weight statistics enough that the inherited GPTQ-lite settings are no longer optimal.
- **Compute-neutral, not compute-free:** this candidate saves artifact bytes, but it does not reduce per-token compute because it still executes 11 logical layer applications.
- **Prior-art caution:** the repo already has a negative naive recurrence result; if this candidate underperforms, that would be evidence that parameter sharing itself is not worth the lost specialization at this scale.

## Suggested next experiments if this idea is directionally good

- Sweep `SHARED_ENCODER_BLOCKS` / `SHARED_DECODER_BLOCKS` between `3/3`, `4/4`, and `5/5`
- Try a stronger identity bias such as `DEPTH_STEP_INIT=0.5`
- Increase `VE_LAYERS` to `8,9,10` if shared decoder stages appear to benefit from more token-identity reinjection
- Re-run the same candidate with the 2026-03-23 parameter-banking / legal-TTT stack if the non-TTT backbone is promising
