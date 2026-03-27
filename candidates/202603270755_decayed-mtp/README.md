# Decayed MTP on the 11L LeakyReLU2 + TTT stack

## Hypothesis

A lightweight multi-token prediction (MTP) auxiliary loss should improve sample efficiency for this repository's 10-minute training regime, because it asks the model to predict not just the next token but a short near-future horizon from the same hidden state. If the extra heads are excluded from export, the candidate can potentially improve trained backbone quality without paying an artifact-size penalty.

This candidate uses **two** MTP heads with a **distance-decayed weighting** (`1.0, 0.5`) so the closest future token dominates the auxiliary objective. The decay is meant to preserve the sample-efficiency benefit of MTP while reducing noise from farther-ahead targets.

## Why this is promising for this repository

The local experiment history shows that this challenge is now bottlenecked less by basic architecture and more by how much useful learning can be packed into a fixed 600-second budget.

The strongest local record at `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` reached **1.1194 BPB** by stacking a mature 11-layer recipe with better activation choice and legal score-first TTT. Earlier 11-layer records already carried dormant MTP code paths, but the logged runs kept `MTP_NUM_HEADS=0`, so the idea exists in-repo without ever being actually tested as a candidate.

That makes MTP a strong next step here:

- it targets **training efficiency**, which is the right bottleneck for the 10-minute track,
- it adds **no exported parameters**, because MTP heads are removed before serialization,
- it reuses existing infrastructure instead of introducing a new model family,
- and it is clearly distinct from the local winning trends so far, which focused on quantization, XSA, EMA/SWA, Partial RoPE, and TTT.

## Prior local experiments that influenced this candidate

Primary base:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`

Supporting local evidence:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
- `records/track_10min_16mb/2026-03-20_11L_EfficientPartialXSA_FA3_SWA120/`

Those runs established the strong 11-layer backbone, while their code/logs also show that MTP plumbing existed but was left disabled (`MTP_NUM_HEADS=0`).

## External research that informed it

- **DeepSeek-V3 Technical Report** ([arXiv:2412.19437](https://arxiv.org/abs/2412.19437)) explicitly states that DeepSeek-V3 "sets a multi-token prediction training objective for stronger performance." That is the main motivation for trying MTP here as a training-efficiency improvement rather than an inference trick.
- **Medusa** ([arXiv:2401.10774](https://arxiv.org/abs/2401.10774)) shows that extra future-token heads can share a backbone and still be useful enough to justify the added supervision. While Medusa is aimed at decoding speed, it reinforces the practicality of lightweight multi-head future prediction.
- **EAGLE** ([arXiv:2401.15077](https://arxiv.org/abs/2401.15077)) emphasizes that farther-ahead prediction becomes less certain. That observation motivated the candidate's **decayed per-head weighting**, so the auxiliary objective emphasizes near-future targets instead of treating all horizons equally.

## What changed versus the chosen base implementation

Base file:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Candidate changes:

- turned MTP on by default with `MTP_NUM_HEADS=2`
- reduced default MTP loss weight to `0.15`
- added `MTP_HEAD_DECAY` (default `0.5`) and changed the MTP loss to use a weighted average across heads instead of a flat mean
- added a safe `scaled_dot_product_attention` fallback when `flash_attn_interface` is unavailable, so importing the script and CPU-side smoke tests are possible in more environments

Everything else stays intentionally close to the best local stack: 11 layers, parameter banks + Parallel Muon, LeakyReLU^2 MLP, XSA, SmearGate, BigramHash, VE layers, EMA export path, mixed int6/lzma serialization, and optional legal TTT.

## How to run or evaluate it

From the candidate directory:

```bash
cd candidates/202603270755_decayed-mtp
SEED=1337 TTT_ENABLED=1 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script resolves its default dataset and tokenizer paths relative to the repository root, so the command above works without manually overriding `DATA_PATH` or `TOKENIZER_PATH`.

Useful ablations:

```bash
# Disable the new idea entirely
MTP_NUM_HEADS=0 TTT_ENABLED=1 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Keep MTP but make farther heads matter less/more
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.15 MTP_HEAD_DECAY=0.25 TTT_ENABLED=1 torchrun --standalone --nproc_per_node=8 train_gpt.py
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.15 MTP_HEAD_DECAY=0.75 TTT_ENABLED=1 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Main expected risks and tradeoffs

- MTP increases training compute, so if the extra supervision does not pay for itself quickly enough, total steps in 600 seconds may drop too much.
- The repository's strongest published gains lately come from eval policy and TTT; MTP may improve the backbone while still producing only a modest end-to-end BPB shift.
- The best local code path already contains many interacting tricks. If MTP helps, the gain may depend strongly on `MTP_LOSS_WEIGHT` and the decay schedule.
- The CPU fallback is for robustness and smoke testing, not for competitive training speed.

## Validation

Commands run in this workflow:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data
python -m compileall candidates/202603270755_decayed-mtp/train_gpt.py
```

Outcomes:

- `python -m compileall train_gpt.py train_gpt_mlx.py data` -> passed
- `python -m compileall candidates/202603270755_decayed-mtp/train_gpt.py` -> passed
- attempted a tiny CPU import/forward smoke test for the candidate, but the workflow container does not have `torch` installed, so a runtime smoke test was **not feasible in this environment**
