# Candidate: MTP Auxiliary Heads on the 11L EMA/GPTQ-lite Stack

## Hypothesis

Training-only multi-token prediction (MTP) heads should improve sample efficiency for this repo's strongest clean 11-layer stack, while costing **zero exported model bytes** because the auxiliary heads are dropped before quantization/export.

## Why this is promising here

The current records are already crowded around the same 10-11 layer, 512d, seq2048, BigramHash/SmearGate, XSA, Partial-RoPE, EMA, and GPTQ-lite/PTQ recipe. The easiest remaining win is likely a **training-only** improvement that leaves the 16MB artifact budget intact.

This repo already carries dormant MTP support in several top scripts, but every record log I found still reports `mtp_num_heads:0`. That makes MTP unusually attractive here: the plumbing largely exists, but it has not been turned into a focused candidate yet.

## Prior repository evidence that informed this candidate

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md` is the clean pre-TTT base I chose because it already combines EMA, GPTQ-lite, XSA, Partial RoPE, LN scaling, VE, and the current export stack.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md` showed that LeakyReLU(0.5)^2 is now the strongest cheap activation tweak in-repo, so I carried that forward here instead of keeping plain relu^2.
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/train.log`,
  `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/train.log`,
  `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train.log`,
  and `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_seed1337.log` all show the MTP code path existed but was left disabled with `mtp_num_heads:0`.
- No prior `candidates/` directory existed in this checkout, so there were no earlier candidates to avoid or build upon directly.

## External research that informed it

- **Multi-Token Prediction** (Gloeckle et al., 2024): <https://arxiv.org/abs/2404.19737>  
  The paper reports better sample efficiency from auxiliary future-token heads with a shared trunk, which is exactly the kind of training-only gain that fits this challenge.
- **MobileLLM** (Liu et al., 2024): <https://arxiv.org/abs/2402.14905>  
  This reinforced the repo's existing deep-and-thin, GQA-heavy design bias for sub-billion models, even though I did not import its block-sharing idea into this particular candidate.
- **ALBERT** (Lan et al., 2019): <https://arxiv.org/abs/1909.11942> and **Universal Transformer** (Dehghani et al., 2018): <https://arxiv.org/abs/1807.03819>  
  I considered selective sharing/recurrent depth reuse after reading these, but the repo's own negative recurrence results made MTP the lower-risk first bet.

## What changed versus the chosen base

Base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

1. **Enable MTP by default** with `MTP_NUM_HEADS=1` and `MTP_LOSS_WEIGHT=0.15`.
2. **Carry forward LeakyReLU(0.5)^2** in the MLP from the current best record instead of plain relu^2.
3. **Fix default paths** so the script can be run directly from `candidates/202604012119_mtp-aux-heads/` without needing to override `DATA_PATH` or `TOKENIZER_PATH`.

Everything else stays deliberately close to the 2026-03-22 stack: 11L/512d, XSA on late layers, Partial RoPE, LN scaling, VE, BigramHash/SmearGate, EMA + tight SWA, and GPTQ-lite export.

## How to run

From the repository root:

```bash
cd candidates/202604012119_mtp-aux-heads
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful sweeps to try first:

```bash
cd candidates/202604012119_mtp-aux-heads
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.10 torchrun --standalone --nproc_per_node=8 train_gpt.py
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.15 torchrun --standalone --nproc_per_node=8 train_gpt.py
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.10 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Evaluation / export note

The training-time MTP heads are already excluded from the exported state dict before quantization, so the final artifact is still scored using the plain next-token model trunk plus the normal quantized weights.

## Validation

1. `python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604012119_mtp-aux-heads/train_gpt.py`  
   **Outcome:** succeeded.
2. CPU smoke import + tiny forward pass against the candidate `GPT` class in a temporary venv with `torch`, `numpy`, and `sentencepiece`, using a small local `flash_attn_interface` stub so the Hopper-only dependency was not required on this CPU host.  
   **Outcome:** succeeded with `cpu_smoke_loss=4.853342`.

## Main risks / tradeoffs

- MTP improves sample efficiency only if the extra head compute does not cost too many training steps inside the 600s wallclock.
- The best `MTP_LOSS_WEIGHT` is probably narrow; too much weight can turn the auxiliary loss into regularization noise.
- This candidate keeps the existing GPTQ-lite export path unchanged, so any remaining score gap from roundtrip quantization still has to be paid.
- If MTP works, the next obvious follow-up is combining it with the stronger 2026-03-23 Parameter Banking / Parallel Muon stack rather than treating this script as the final endpoint.
