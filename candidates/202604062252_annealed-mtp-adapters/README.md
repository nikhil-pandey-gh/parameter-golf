# Annealed tied-output MTP adapters

## Hypothesis

Multi-token prediction should improve **sample efficiency** on this challenge's fixed 10-minute budget, but it needs to be cheap enough that the auxiliary objective does not erase its own gain by slowing training. This candidate turns the dormant MTP path from the strong 2026-03-22 non-TTT stack into a lighter variant: predict **t+2 and t+3** with small horizon-specific adapters, reuse the main output projection, and **anneal the auxiliary loss away during warmdown** so the end of training matches the next-token evaluation objective.

## Why this is promising here

- The repository's best pure-training stack already looks compute-efficient; the next obvious lever is **better bits learned per step**, not a larger artifact.
- The challenge uses a **1024-token SentencePiece vocab**, so extra horizon logits are unusually cheap compared with typical LLM settings.
- The candidate keeps MTP **training-only** and excludes the auxiliary adapters from export, so the final artifact still targets the same post-training int6 path as the 2026-03-22 base.

## Prior repository evidence that shaped this choice

- **Base fork:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` is the strongest non-TTT stack in the repo and already combines the patterns that keep recurring in good runs: 11 layers, 3x MLP, BigramHash, SmearGate, XSA, partial RoPE, EMA, GPTQ-lite int6 export.
- **Why not recurrence / sharing first:** prior records and non-record runs explicitly call out depth recurrence as a dead end under the 10-minute wall-clock limit.
- **Why a training-objective tweak is still attractive:** the 2026-03-23 record shows that small training-side changes can still move pre-TTT quality by a few thousandths of BPB.

## External research that informed it

- Gloeckle et al., [*Better & Faster Large Language Models via Multi-token Prediction*](https://arxiv.org/abs/2404.19737)
- Gerontopoulos et al., [*MuToR: Multi-Token Training with Rectification*](https://arxiv.org/abs/2505.10518)

These papers both argue that predicting multiple future tokens can improve optimization efficiency. The implementation here follows that spirit, but trims the repo's existing dormant MTP path down to something that should fit this challenge's tight wall-clock budget more gracefully.

## What changed vs. the chosen base

1. **Enabled MTP by default** with `MTP_NUM_HEADS=2` and `MTP_LOSS_WEIGHT=0.15`, so the trunk learns to predict `t+2` and `t+3` in addition to the standard next-token target.
2. Replaced the base script's unused full-vocab auxiliary heads with **small low-rank horizon adapters** (`MTP_ADAPTER_DIM=64`) and reused the main tied output projection for auxiliary logits.
3. Added **warmdown annealing** for the auxiliary objective (`MTP_WARMDOWN_ANNEAL=1`), so the MTP loss smoothly decays with the same scale factor already used for LR warmdown.
4. Added a **PyTorch SDPA fallback** when `flash_attn_interface` is unavailable, keeping the candidate more portable for local inspection and smoke checks.

## Files added

- `train_gpt.py` - self-contained candidate training/eval/export script
- `README.md` - rationale, provenance, and validation notes

## How to run

From this candidate directory:

```bash
RUN_ID=annealed_mtp_adapters \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful knobs:

```bash
MTP_NUM_HEADS=2
MTP_LOSS_WEIGHT=0.15
MTP_ADAPTER_DIM=64
MTP_WARMDOWN_ANNEAL=1
```

## Validation run in this workflow

- `python -m compileall candidates/202604062252_annealed-mtp-adapters/train_gpt.py` - **passed**
- Lightweight CPU/import smoke beyond syntax was **not feasible in this runner** because the Python environment here does not have the repository's core ML dependencies installed (`torch`, `numpy`, `sentencepiece` were all missing).

## Main risks / tradeoffs

- Even lightweight MTP still adds training-time compute; if the extra horizons cost too many steps, the net result can go negative.
- Because the auxiliary adapters are excluded from export, the win has to come from a better-trained trunk rather than from extra inference capacity.
- The best loss weight and adapter rank are likely narrow; `0.15` / `64` are reasonable starting points, not tuned optima.
