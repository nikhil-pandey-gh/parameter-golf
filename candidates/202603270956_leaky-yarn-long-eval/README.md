# LeakyReLU^2 + YaRN Long-Context Eval

## Hypothesis

On top of the strongest non-TTT stack in this repository, a **LeakyReLU(0.5)^2 MLP** plus **YaRN-style RoPE extrapolation for a longer evaluation window** should improve tokenizer-agnostic validation bpb without changing the training-time sequence length or paying the full complexity cost of legal TTT.

The bet is that this challenge has repeatedly rewarded cheap context improvements at evaluation time, and that the current non-TTT stack is already strong enough that a small activation gain plus better long-context extrapolation is more attractive than a larger architectural rewrite.

## Why this is promising for this repository

Three patterns showed up clearly in the existing records:

- `SlidingWindowEval` showed that evaluation context alone can move bpb materially even when training is unchanged.
- The earlier `LongContextSeq2048` and `TrainingOptSeq4096` records showed that longer usable context helps this benchmark family.
- The current top record, `LeakyReLU_LegalTTT_ParallelMuon`, reports a measurable gain from swapping `relu^2` to `LeakyReLU(0.5)^2`.

This candidate keeps the strong 11-layer EMA/GPTQ-lite/partial-RoPE/XSA backbone from the best non-TTT record, then adds only two low-cost changes:

1. `LeakyReLU(0.5)^2` in the MLP.
2. YaRN-style rotary scaling with a longer default `EVAL_SEQ_LEN=3072`.

That makes it a good fit for the repository's current trajectory: small, composable changes that preserve the proven training recipe.

## Record lineage that influenced this candidate

There were **no existing `candidates/` directories** in the repository at the time this folder was created, so the comparison set was entirely the prior `records/` history.

The main local influences were:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - chosen as the base implementation because it is the strongest non-TTT stack in-repo.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - contributed the `LeakyReLU(0.5)^2` activation idea.
- `records/track_10min_16mb/2026-03-19_SlidingWindowEval/`
  - reinforced that evaluation-only context improvements are worth pursuing here.
- `records/track_10min_16mb/2026-03-18_LongContextSeq2048/` and `records/track_10min_16mb/2026-03-19_TrainingOptSeq4096/`
  - showed that this benchmark benefits from longer usable context.
- `records/track_10min_16mb/2026-03-19_WarmdownQuantization/`
  - reported that `eval_seq_len` itself can move results, which makes a longer-context eval candidate especially relevant.

## External research that informed it

The external research thread was about **long-context degradation without replacing the entire attention kernel**:

- **YaRN** (`arXiv:2309.00071`) showed that RoPE-based models can extend usable context more efficiently with better scaling than plain interpolation.
- **Softmax-1** (`arXiv:2410.17174`) and **SSMax / Scalable-Softmax** (`arXiv:2501.19399`) both argue that attention quality degrades as context grows because vanilla softmax flattens. Those papers motivated targeting long-context behavior explicitly.

In this repository, directly replacing softmax inside FlashAttention would be a much larger infrastructure change. So this candidate takes the lowest-risk path that still addresses the same pressure point: keep FlashAttention compatibility, but improve long-context behavior on the RoPE side with YaRN-style scaling.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

- Swapped the MLP activation from `relu^2` to `LeakyReLU(0.5)^2`.
- Added a **YaRN-style RoPE scaling path** for sequence lengths above the original training context.
- Changed the default evaluation context from `2048` to `3072` via `EVAL_SEQ_LEN=3072`.
- Added a small **FlashAttention fallback** to `scaled_dot_product_attention` so the module can still run on CUDA setups that have PyTorch SDPA available even if `flash_attn_interface` is absent, and so it can be imported more easily in non-FlashAttention environments.
- Threaded the new knobs through both the training model and the export/reload evaluation model so quantized evaluation uses the same activation and RoPE behavior.

Everything else intentionally stays close to the 2026-03-22 base: 11 layers, partial RoPE, XSA on the last 4 layers, LN scaling, shared value embedding, EMA, GPTQ-lite export, and sliding-window evaluation.

## How to run or evaluate it

From the repository root:

```bash
RUN_ID=leaky_yarn_long_eval \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=11 \
TRAIN_SEQ_LEN=2048 \
EVAL_SEQ_LEN=3072 \
MLP_LEAKY_SLOPE=0.5 \
ROPE_SCALING=yarn \
ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS=2048 \
ROPE_YARN_BETA_FAST=32 \
ROPE_YARN_BETA_SLOW=1 \
EVAL_STRIDE=64 \
MAX_WALLCLOCK_SECONDS=600 \
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  candidates/202603270956_leaky-yarn-long-eval/train_gpt.py
```

Useful ablations to try next:

- `EVAL_SEQ_LEN=2048` with `ROPE_SCALING=yarn` to isolate the scaling change.
- `EVAL_SEQ_LEN=3072` with `ROPE_SCALING=dynamic` to isolate the longer window alone.
- `MLP_LEAKY_SLOPE=0.0` to recover `relu^2` behavior.
- `EVAL_SEQ_LEN=4096` if evaluation time budget still looks safe on 8xH100.

## Main expected risks and tradeoffs

- **Evaluation time risk**: `EVAL_SEQ_LEN=3072` will cost more than the 2048-token base during final eval.
- **Long-context mismatch risk**: the model is still trained at `TRAIN_SEQ_LEN=2048`, so some of the extra eval context may be poorly exploited.
- **RoPE scaling risk**: YaRN-style scaling may help less than hoped on a stack that already uses partial RoPE and strong sliding eval.
- **Non-TTT ceiling**: even if this candidate improves the non-TTT frontier, it may still trail the current legal-TTT record.

## Validation

### Commands run

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603270956_leaky-yarn-long-eval/train_gpt.py
```

Outcome: **passed**.

Attempted runtime smoke test:

```bash
python - <<'PY'
import importlib.util
from pathlib import Path
import torch
path = Path('candidates/202603270956_leaky-yarn-long-eval/train_gpt.py')
spec = importlib.util.spec_from_file_location('candidate_train_gpt', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
model = module.GPT(
    vocab_size=64,
    num_layers=2,
    model_dim=32,
    num_heads=4,
    num_kv_heads=2,
    mlp_mult=2,
    tie_embeddings=True,
    tied_embed_init_std=0.01,
    logit_softcap=30.0,
    rope_base=10000.0,
    qk_gain_init=1.5,
    mtp_num_heads=0,
    mtp_loss_weight=0.0,
    bigram_vocab_size=32,
    bigram_dim=16,
    xsa_last_n=1,
    rope_dims=8,
    ln_scale=True,
    dtg=False,
    ve_enabled=True,
    ve_dim=8,
    ve_layers='1',
    mlp_leaky_slope=0.5,
    rope_scaling='yarn',
    rope_original_max_position_embeddings=32,
    rope_yarn_beta_fast=32.0,
    rope_yarn_beta_slow=1.0,
)
PY
```

Outcome: **not feasible in this workflow container** because `torch` is not installed here (`ModuleNotFoundError: No module named 'torch'`). The candidate script itself is syntax-valid, but an execution smoke test has to run in a Python environment that actually has the training runtime dependencies installed.
