## Candidate: two-head multi-token prediction on top of the current 11L GPTQ-lite record

### Hypothesis

Enable a small amount of **training-only multi-token prediction (MTP)** on top of the current best 11-layer stack. MTP should improve sample efficiency during the fixed 10-minute training window, while keeping the final submission size unchanged because the auxiliary heads are already excluded from export.

### Why this is promising for this repository

The repo history shows a clear trend: the best records have mostly converged on the same 11-layer, seq-2048, int6+zstd, EMA, XSA, Partial-RoPE stack, and the remaining gains have come from changes that improve either **sample efficiency** or **quantization robustness** without adding meaningful artifact bytes.

MTP is attractive here because it attacks the sample-efficiency side of that equation:

- it adds richer supervision during training,
- it costs **zero artifact bytes at submission time** because `mtp_heads` are dropped from the exported state dict,
- and the strongest existing record already contains dormant MTP plumbing, so the implementation risk is low.

### Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`
  - chosen base implementation and current strongest published stack.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`
  - established Partial RoPE + LN scale as strong zero-byte architectural improvements.
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271`
  - established the 11L + XSA4 + EMA family that the current best record extends.

There were no prior `candidates/` directories in the repository when this candidate was created.

### External research that informed it

- **Better & Faster Large Language Models via Multi-token Prediction** (Gloeckle et al., 2024, arXiv:2404.19737)
  - motivates adding auxiliary future-token heads to improve training efficiency and representation quality.
- **DeepSeek-V2** (arXiv:2405.04434)
  - reinforces the broader theme that training-only auxiliary objectives and architectural tricks can improve efficiency without directly increasing deployment cost.
- In contrast, more invasive ideas considered during research (BitNet-style native low-bit training, Mamba/RetNet swaps, full MLA ports) looked much riskier for a surgical candidate in this repo.

### What changed versus the chosen base implementation

Base:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

- default `DATA_PATH` / `TOKENIZER_PATH` now resolve relative to the repository root, so the script can be run directly from this candidate directory
- `MTP_NUM_HEADS` default changed from `0` to `2`
- `MTP_LOSS_WEIGHT` default changed from `0.2` to `0.1`

Everything else is intentionally kept identical to the current best record:

- 11 layers, 512 dim, 8 heads / 4 KV heads
- MLP 3x, relu-squared
- SmearGate + BigramHash
- XSA on the last 4 layers
- Partial RoPE (`16/64`) + LN scaling
- EMA + GPTQ-lite clip search
- int6/int8 mixed export + zstd compression

### Why two heads and 0.1 weight

This is a conservative default for the 10-minute budget:

- `2` heads keeps the auxiliary objective meaningful without committing to the highest compute overhead.
- `0.1` is intentionally milder than the previous dormant default (`0.2`) to reduce the risk that auxiliary losses interfere with the final next-token objective or late quantization-sensitive refinement.

### How to run or evaluate it

From the candidate directory:

```bash
cd candidates/202603242104_twohead-mtp
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The defaults in this file already represent the candidate. You can still override the MTP knobs explicitly if you want to sweep around it:

```bash
cd candidates/202603242104_twohead-mtp
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.1 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Expected tradeoffs and risks

- **Wallclock overhead:** extra vocab projections for the auxiliary heads may reduce steps completed in 600 seconds.
- **Objective mismatch:** a stronger auxiliary loss can help early learning but hurt final next-token optimization if over-weighted.
- **Quantization interaction:** because the exported model does not include the MTP heads, the hope is better trunk features transfer cleanly, but this is still an empirical question.
- **Compute-to-gain uncertainty:** MTP is well motivated by literature, but this repository has not yet measured its net effect inside the current quantization-heavy 11L stack.

### Validation run for this candidate

Commands run:

```bash
python -m compileall candidates/202603242104_twohead-mtp/train_gpt.py
python - <<'PY'
from pathlib import Path
candidate = Path('candidates/202603242104_twohead-mtp/train_gpt.py').resolve()
repo_root = candidate.parent.parent.parent
print(repo_root / 'data' / 'datasets' / 'fineweb10B_sp1024')
print(repo_root / 'data' / 'tokenizers' / 'fineweb_1024_bpe.model')
PY
```

Outcome:

- `compileall` passed successfully.
- the candidate's default dataset/tokenizer paths now resolve to repository-root `data/...` locations instead of incorrectly resolving inside the candidate directory.

Attempted additional smoke validation:

- I attempted a tiny CPU-only import/forward smoke test, but this workflow environment does not have `torch` installed in `/usr/bin/python`, so a runtime smoke test was not feasible here without adding new infrastructure.
- This checkout also does not include the actual training dataset/tokenizer blobs under `data/datasets/...`, so no data-dependent launch smoke test was possible locally.
