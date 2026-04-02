# AWQ-lite folded PTQ on the current LeakyReLU² record stack

## Hypothesis

The current best stack is already strong on the training side, so the next clean gain is more likely to come from the float-to-artifact boundary than from another large architecture rewrite. This candidate keeps the March 23 record's 11-layer LeakyReLU(0.5)² + legal score-first TTT + Parallel Muon recipe, but replaces plain GPTQ-lite export with an **activation-aware folded PTQ** pass that rescales internal channels before int6 packing.

The key repo-specific twist is that two large subpaths admit **exact, no-runtime-overhead equivalence transforms**:

1. the LeakyReLU² MLP path, because LeakyReLU is positively homogeneous and the square makes the down-projection compensation deterministic; and
2. the value-to-output attention path on layers without XSA or value embeddings, because it stays linear from `v` to `proj`.

That lets the script re-balance the channels that matter most for quantization, then reuse the existing GPTQ-lite int6 path and lzma packaging.

## Why this is promising here

Recent repo history says the frontier is quantization-aware more than architecture-wide churn:

- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` showed that a better int6 export path alone was worth another `-0.0013` BPB over the previous 11-layer stack.
- `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` pushed further with a mostly model-side LeakyReLU² change, but it still exports through the same GPTQ-lite family.
- Older records repeatedly found that fp16 embeddings, int6/int5 packing, mixed precision, and sliding evaluation mattered more than many riskier architecture ideas.

So the bet here is simple: **preserve the best training recipe, but make the exported int6 model easier to quantize without changing the float model's function**.

## Prior records that influenced this candidate

- **Primary base:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
- **Direct export inspiration:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- **Context for where gains have come from:** the March 19-21 int6/QAT/XSA/EMA/Partial-RoPE record chain and the sliding-window evaluation record.

## External research that informed it

- **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration** — activation-driven channel protection for weight-only PTQ, using equivalent scaling instead of mixed-precision exceptions. <https://arxiv.org/abs/2306.00978>
- **SmoothQuant+** — channel smoothing for weight-only PTQ, motivated by activation outliers amplifying quantization error. <https://arxiv.org/abs/2312.03788>
- **QuaRot** — outlier removal via exact equivalent transforms, showing that mathematically identical reparameterizations can materially improve low-bit quantization. <https://arxiv.org/abs/2404.00456>
- **SpinQuant** — learned rotations outperform random ones in hard low-bit settings, reinforcing the idea that better pre-quantization geometry is valuable even without changing the core model. <https://arxiv.org/abs/2405.16406>

This candidate does **not** implement full AWQ/QuaRot/SpinQuant infrastructure. Instead it ports their most relevant idea to this repo's constraints: a tiny, exact, post-training channel reparameterization that can be expressed inside the existing trainer/export script.

## What changed versus the chosen base

Compared with `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`, this candidate adds:

1. **Candidate-directory-safe defaults** for `DATA_PATH` and `TOKENIZER_PATH`, resolved from the script's location so `train_gpt.py` can be launched from this directory directly.
2. **Activation-aware PTQ calibration** on a small slice of the training split, not the validation split.
3. **Exact folded MLP hidden-channel scaling** before quantization:
   - scale `mlp.fc` rows up by a learned activation-derived factor,
   - scale `mlp.proj` columns down by the square of that factor.
4. **Selective exact folded value/output scaling** before quantization on layers where the path stays linear:
   - scale `attn.c_v` rows,
   - inversely scale `attn.proj` input columns.
5. **Master-only export/calibration work**, since only rank 0 needs to write the packaged int6 artifact before the other ranks reload it for evaluation.

The training loop, model topology, EMA/LAWA handling, lzma packaging, sliding eval, and legal TTT path stay otherwise aligned with the strongest known base.

## How to run

From this candidate directory, in the same CUDA/FlashAttention environment used by the recent 11-layer record scripts:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
ACT_AWARE_PTQ=1 PTQ_CALIBRATION_TOKENS=262144 PTQ_MLP_ALPHA=0.5 PTQ_VO_ALPHA=0.5 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The new PTQ knobs are:

- `ACT_AWARE_PTQ=1` to enable the folded export transform
- `PTQ_CALIBRATION_TOKENS` to control how much train-split data is used for calibration
- `PTQ_MLP_ALPHA` and `PTQ_VO_ALPHA` to control how aggressively activation stats shape the channel scales
- `PTQ_SCALE_LIMIT` to clamp the geometric rebalancing range

Other inherited behavior from the base record, such as the EMA export path, remains hard-coded in the script exactly as in the chosen starting point.

Like the recent 11-layer record files, this candidate still expects the `flash_attn_interface` Python module to be available at runtime.

## Evaluation notes

- The candidate keeps the record stack's standard int6 roundtrip eval, stride-64 sliding eval, and optional legal score-first TTT.
- Calibration is intentionally taken from the **training split** so the export heuristic does not peek at the validation set.
- Value/output scaling is skipped on layers where exact folding would be violated by repo-specific features like XSA or value embeddings.

## Main risks and tradeoffs

- The calibration heuristic is intentionally lightweight; it may underfit the best possible rebalancing schedule.
- Because exactness matters, the V->O transform is applied only on layers where XSA/value-embedding/value-residual logic does not break the linear equivalence.
- Export time increases slightly because the script now runs a short calibration pass before packing the int6 artifact.
- The gain could be modest if GPTQ-lite already removed most of the easy row-scale error on this stack.

## Validation run in this workflow

Commands executed here:

```bash
python -m compileall candidates/202604021555_awq-lite-ptq/train_gpt.py
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
PY
```

Outcome:

- `compileall` succeeded.
- A stronger smoke test was **not feasible in this runner** because the provided Python environment does not currently have the repo's declared `torch` dependency installed (`ModuleNotFoundError: No module named 'torch'`), and this script hard-requires CUDA for actual startup.
