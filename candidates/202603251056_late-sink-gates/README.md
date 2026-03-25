# Late Sink Gates on the 11L XSA/EMA/TTT Stack

## Hypothesis

The current record family is already strong on depth, compression, and evaluation tricks, but it does not yet make sink control a first-class design goal. This candidate adds **late-layer explicit sink-control gates** so the model can absorb attention and residual sink behavior into a few cheap learnable parameters instead of relying entirely on emergent outliers.

Concretely, the candidate enables three small mechanisms only in the deepest layers:

- detached **per-head attention gates**,
- detached **delta/residual gates** on block updates,
- **late value anchoring** via value residual mixing.

The expectation is better stability in the same 10-minute budget and a smaller post-quantization gap, especially on top of the already-strong 11-layer int6 export path.

## Why this is promising for this repository

Repository evidence points to a very stable winning recipe:

- 11 layers beat the earlier 9L/10L families once compression got good enough.
- XSA on the deepest layers, EMA, partial RoPE, LN scaling, and GPTQ-lite all produced repeatable gains.
- Long-context/sliding evaluation and legal TTT help, but the largest durable gains still came from **training-side architecture and quantization robustness**, not from recurrence or other broad rewrites.

Recent research also supports this direction:

- **Qiu et al., “A Unified View of Attention and Residual Sinks”** (`arXiv:2601.22966`) argue that explicit gated rescaling can absorb sink behavior and improve quantization robustness.
- **Bae et al., “Affine-Scaled Attention”** (`arXiv:2602.23057`) show that modest input-dependent attention rescaling improves stability and optimization.
- **Huang et al., “Threshold Differential Attention”** (`arXiv:2601.12145`) reinforce the broader point that sink suppression and inhibitory attention views matter for language modeling.

I intentionally did **not** implement a full new attention kernel here. The candidate keeps the current repository’s winning training/export infrastructure intact and adds the smallest practical approximation to those ideas.

## Prior records and experiments that influenced this candidate

Primary positive influences:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - strongest non-TTT training stack,
  - established EMA + GPTQ-lite + warmdown3500 as a durable base.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - strongest overall stack,
  - showed LeakyReLU^2 and legal score-first TTT can stack on top of the mature 11L recipe.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - showed partial RoPE and LN scale are real gains, while the specific late-QAT path there was effectively a no-op.

Negative or cautionary influences:

- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
  - layer recurrence was clearly negative under a fixed wallclock budget.
- `records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/`
  - longer training alone did not solve the post-quantization bottleneck.

There were no prior `candidates/` directories in the repository when this candidate was created.

## Chosen base implementation

This candidate is a focused fork of:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

I chose the 2026-03-23 stack rather than the simpler 2026-03-22 non-TTT base because it is the strongest current end-to-end implementation and already contains the repository’s mature parameter-banked 11-layer training path. The new work here is intentionally narrow and training-side, so preserving the strongest known scaffold felt like the right trade.

One detail worth calling out: the 2026-03-23 record README used `BIGRAM_VOCAB_SIZE=1536` in its leaderboard-tuned run command, while the underlying code default stayed at `2048`. This candidate keeps the code default at `2048` so the sink-gating change is isolated at the script level, and treats `1536` as an explicit run-time tuning flag when matching the artifact-tuned record stack.

## What changed versus the chosen base

The new `train_gpt.py` keeps the winning 11L/XSA/EMA/TTT/export stack, but adds:

1. **Late-layer-only sink gates**
   - `SINK_GATE_LAST_N` controls how many deepest blocks get sink-control gates.
   - `GATED_ATTENTION=1` and `DTG_ENABLED=1` are now on by default for the candidate.

2. **Detached gate inputs**
   - attention gates can use detached block inputs (`ATTN_GATE_DETACH=1`),
   - delta gates can use detached residual inputs (`DTG_GATE_DETACH=1`).

   This makes the gates behave more like explicit rescaling factors than another large feature-learning path.

3. **Late-layer value anchoring**
   - `VALUE_RESIDUAL=1` is enabled by default,
   - `VALUE_RESIDUAL_LAST_N` limits that residual value mixing to deep layers instead of all layers.

4. **FlashAttention fallback for local bring-up**
   - if `flash_attn_interface` is unavailable, the script falls back to PyTorch `scaled_dot_product_attention`.
   - this is mainly for smoke validation and local experimentation; the intended leaderboard path is still CUDA + the existing fast stack.

5. **CPU smoke mode**
   - `CPU_SMOKE_TEST=1` instantiates the candidate model on CPU and runs a tiny synthetic forward pass without needing dataset shards or CUDA.

6. **Path resolution fixed for candidate-directory execution**
   - the default dataset and tokenizer paths now resolve relative to the repository root via the script location, so running `python train_gpt.py` from inside the candidate directory works without extra path flags.

## How to run or evaluate it

From the candidate directory:

```bash
cd candidates/202603251056_late-sink-gates
```

Minimal local smoke:

```bash
CPU_SMOKE_TEST=1 python train_gpt.py
```

Suggested 8xH100-style run, keeping the current leaderboard-style stack but with this candidate’s sink-control defaults:

```bash
BIGRAM_VOCAB_SIZE=1536 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful candidate-specific knobs:

```bash
SINK_GATE_LAST_N=4
VALUE_RESIDUAL_LAST_N=4
ATTN_GATE_DETACH=1
DTG_GATE_DETACH=1
ATTN_GATE_INIT_BIAS=3.0
DTG_GATE_INIT_BIAS=2.0
```

## Validation run for this candidate

I ran the following lightweight checks:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r ../../requirements.txt
python -m compileall candidates/202603251056_late-sink-gates/train_gpt.py

cd candidates/202603251056_late-sink-gates
CPU_SMOKE_TEST=1 python train_gpt.py
```

Observed outcome:

```text
cpu_smoke:ok loss:6.9474 logits_shape:(1, 8, 1024)
```

Notes:

- The repository runner initially lacked `numpy`, `sentencepiece`, and `torch`, so I installed the repo’s declared Python requirements in a temporary virtualenv before running the smoke check.
- This validation confirms the candidate script imports, constructs the model, and executes a forward path locally.
- I did **not** run a real GPU training job from this environment.

## Main expected risks and tradeoffs

- The gates may **over-dampen** the same late layers that XSA is trying to sharpen.
- Value residual mixing is plausible but still more speculative than the gating pieces.
- This is a **pragmatic approximation** to sink-aware/affine attention ideas, not a full kernel-level implementation.
- The candidate is syntactically and structurally validated, but it still needs real GPU ablations to prove whether the extra sink control helps pre-quant BPB, post-quant BPB, or both.
