# Primer-style QKV Conv on the 11L EMA/GPTQ-lite Stack

## Hypothesis

The repository has already converged on a strong compact-model recipe built around 11 layers, XSA on deep layers, partial RoPE, VE, EMA, GPTQ-lite, and a 3x MLP. It has also already discovered that squared-style activations remain strong in this regime, with the current top record improving further by switching from `relu^2` to `LeakyReLU(0.5)^2`.

This candidate adds the other main idea from Primer-style transformers that the repo has not yet tried: a lightweight causal depthwise convolution after the Q/K/V projections. The bet is that tiny models under a strict artifact cap still benefit from a little explicit local mixing, especially on byte-compression-heavy validation where short-range structure matters a lot.

To keep the runtime risk small, the depthwise Q/K/V convolutions are only applied on the deepest 4 layers, matching the existing "spend extra compute on the last layers" pattern already used for XSA.

## Why this looks promising here

- The repo has **no prior convolutional attention experiments** in `records/`, so this is a genuinely new architectural direction rather than another quantization-only remix.
- Existing winning additions such as `SmearGate`, `BigramHash`, VE, and XSA all suggest that adding cheap local/context bias helps compact models.
- Primer-style depthwise Q/K/V mixing adds only a tiny number of parameters relative to the 16MB budget, so most of the artifact budget still goes to the core transformer.
- The candidate keeps the strong `2026-03-22` pre-TTT base intact, which makes it easier to tell whether the model change itself is useful.

## Prior repository runs that influenced this candidate

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Chosen as the implementation base because it is the strongest clean pre-TTT stack in the repo.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Motivated carrying over `LeakyReLU(0.5)^2` instead of plain `relu^2`.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - Reinforced the "small zero/near-zero-parameter architectural tweaks can still move BPB" pattern.
- `records/track_10min_16mb/2026-03-19_SlidingWindowEval/`
  - Reminder that evaluation context matters a lot, so this candidate preserves the repo's sliding-window path instead of changing eval methodology.

There were no prior `candidates/` directories at the time this candidate was created.

## External research that informed it

- **Primer: Searching for Efficient Transformers for Language Modeling** (`arXiv:2109.08668`)
  - Primer attributes much of its gain to two simple changes: squared ReLU activations and a depthwise convolution after Q, K, and V.
  - This repo already explored the activation family aggressively; this candidate adds the missing Q/K/V convolution piece.
- **Systems and Algorithms for Convolutional Multi-Hybrid Language Models at Scale** (`arXiv:2503.01868`)
  - A more recent result arguing that convolutional/hybrid language models retain complementary benefits for token manipulation and compression tasks.
- **Mechanistic evaluation of Transformers and state space models** (`arXiv:2505.15105`)
  - Reports that short convolutions can support induction-like behavior, which is relevant when trying to improve compact autoregressive models with minimal added parameters.

## What changed versus the chosen base

Base implementation:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

1. Added **Primer-style causal depthwise convolutions** after the `c_q`, `c_k`, and `c_v` projections.
2. Applied those convolutions only on the **last 4 layers** via:
   - `PRIMER_CONV_LAST_N=4`
   - `PRIMER_KERNEL_SIZE=3`
3. Switched the MLP activation from `relu^2` to **`LeakyReLU(0.5)^2`**.
4. Added a **FlashAttention fallback** to PyTorch SDP, including non-flash SDP backend enablement when `flash_attn_interface` is unavailable.
5. Changed default dataset/tokenizer paths to be **script-relative**, so `train_gpt.py` works when run from inside this candidate directory.

## Files added

- `train_gpt.py` — self-contained candidate training/eval script
- `README.md` — this note

## How to run

From the candidate directory:

```bash
cd candidates/202603251630_primer-qkvconv

RUN_ID=primer_qkvconv \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
PRIMER_CONV_LAST_N=4 PRIMER_KERNEL_SIZE=3 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script defaults `DATA_PATH` and `TOKENIZER_PATH` relative to the repository root, so if the standard challenge dataset/tokenizer layout exists under `data/`, you do not need to override them when launching from this folder.

## Expected tradeoffs / risks

- The extra Q/K/V local mixing may help BPB, but it can also reduce throughput enough to cost training steps inside the fixed 10-minute budget.
- Applying the conv only to deep layers is a compromise; if the effect is too weak, the feature may simply not move the metric.
- The new weights are tiny, but they still alter the optimizer/quantization landscape and could slightly worsen post-quantization behavior.
- `LeakyReLU(0.5)^2` and Primer-style Q/K/V mixing may interact nonlinearly; this is a "best-next-idea" candidate, not a clean single-factor ablation.

## Validation

Executed on this runner:

```bash
python -m compileall candidates/202603251630_primer-qkvconv/train_gpt.py
```

Outcome:

- **Passed**.

Attempted CPU smoke validation with a synthetic import-and-forward path:

```bash
python - <<'PY'
import importlib.util
from pathlib import Path
import torch

path = Path('candidates/202603251630_primer-qkvconv/train_gpt.py').resolve()
spec = importlib.util.spec_from_file_location('candidate_train_gpt', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
...
PY
```

Outcome:

- **Could not run on this runner** because both `torch` and `sentencepiece` are missing from `/usr/bin/python` and `/usr/bin/python3`.
- The candidate code now includes a PyTorch SDP fallback for attention, and it re-enables non-flash CUDA SDP backends when `flash_attn_interface` is unavailable.
