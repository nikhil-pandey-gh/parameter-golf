# Candidate: LeakyReLU2 + Legal TTT + Training-Only MTP Auxiliary Heads

## Hypothesis

Add **multi-token prediction (MTP) auxiliary heads** to the current best stack so the shared trunk learns longer-horizon structure more sample-efficiently during the 600s train budget, while keeping the exported artifact unchanged by dropping the auxiliary heads before serialization.

## Why this is promising for this repository

The repo's history shows that the current frontier is already dominated by mature architectural and compression choices: 11 layers, 3x MLP, EMA/SWA, Partial RoPE, XSA, BigramHash, value embeddings, aggressive low-bit export, and legal score-first TTT. The remaining room looks more like **training-efficiency** headroom than another obvious artifact-side win.

This candidate fits that gap well:

- prior records already include dormant MTP support in the stronger 11L scripts, but every reviewed run kept `MTP_NUM_HEADS=0`;
- MTP adds **training-only** parameters, so it does not spend the 16MB artifact budget;
- the local best stack already exports without MTP heads, making the implementation low-risk and minimal;
- the main subtlety is TTT interaction, so this candidate explicitly disables MTP during validation-time adaptation to keep TTT aligned with the scored next-token objective.

## Prior repository work that influenced this candidate

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`
  - current best local score: **1.1194 mean val_bpb**
  - contributes the base stack: LeakyReLU(0.5)^2, legal score-first TTT, Parallel Muon, Parameter Banking, XSA, Partial RoPE, VE128, BigramHash, EMA/SWA, GPTQ-lite-style low-bit export
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md`
  - strongest non-TTT training-centric stack and evidence that late-stage gains are now small and require precise, low-overhead additions
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md`
  - confirms Partial RoPE + LN scale are strong zero/near-zero-cost improvements and documents a prior `torch.compile` pitfall; that informed keeping this candidate's MTP change simple and explicit

There were **no existing `candidates/` entries** in the repository when this candidate was created.

## External research that informed this candidate

1. Fabian Gloeckle et al., **"Better & Faster Large Language Models via Multi-Token Prediction"** ([arXiv:2404.19737](https://arxiv.org/abs/2404.19737))
   - proposes predicting multiple future tokens with independent heads on top of a shared trunk
   - reports higher sample efficiency with no meaningful training-time overhead in their setup
   - explicitly frames MTP as an auxiliary objective, which maps well to this repository because auxiliary heads can be excluded from the final artifact
2. DeepSeek-AI et al., **"DeepSeek-V3 Technical Report"** ([arXiv:2412.19437](https://arxiv.org/abs/2412.19437))
   - uses a multi-token prediction training objective as one of the ingredients behind a strong, efficient modern LM stack
   - reinforces that MTP is a serious contemporary optimization target, not just a niche decoding trick

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

This candidate keeps the base architecture and export path intact, then makes only the MTP-related changes needed for a real experiment:

1. **Enable MTP by default**
   - `MTP_NUM_HEADS=2`
   - `MTP_LOSS_WEIGHT=0.15`
2. **Default to the best known local eval stack**
   - `BIGRAM_VOCAB_SIZE=1536`
   - `TTT_ENABLED=1`
   - `TTT_FREEZE_BLOCKS=0`
3. **Keep MTP strictly training-only**
   - raw `final_model.pt` keeps the full training state, including MTP heads, so it can be reloaded with the script's training config
   - the shipped low-bit artifact still strips `mtp_heads.*`, so artifact size and exported eval behavior stay aligned with the challenge budget
   - if TTT is ever run on a model instance that still includes MTP heads, the TTT path now excludes those heads from adaptation and temporarily zeroes MTP so updates stay aligned with the scored next-token objective

## How to run

From this candidate directory:

```bash
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful overrides if you want to sweep the new idea:

```bash
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.10 torchrun --standalone --nproc_per_node=8 train_gpt.py
MTP_NUM_HEADS=4 MTP_LOSS_WEIGHT=0.15 torchrun --standalone --nproc_per_node=8 train_gpt.py
TTT_ENABLED=0 MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.15 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## How to evaluate

The script keeps the base record behavior:

- trains under the 600s wallclock cap,
- exports `final_model.int6.ptz`,
- runs roundtrip eval,
- runs sliding-window eval,
- and, with defaults, runs legal score-first TTT.

## Expected risks and tradeoffs

- **Step throughput risk**: even small MTP heads add extra logits and loss work every step, so the gain must beat any reduction in completed steps.
- **Small-model uncertainty**: the strongest MTP paper reports larger gains at larger scales; this tiny-model regime may benefit less.
- **TTT interaction risk**: if MTP were left enabled during TTT, adaptation could drift toward the auxiliary objective; this candidate explicitly disables that path during TTT to avoid objective mismatch.
- **No artifact help by itself**: unlike the repo's strongest past wins, MTP does not directly improve compression, so it must win through representation quality / sample efficiency.

## Validation

Recorded on this candidate directory:

1. `python -m compileall candidates/202604021227_mtp-aux-heads/train_gpt.py`
2. `python - <<'PY' ... PY` import smoke with a stubbed `flash_attn_interface`

Outcome summary:

- `compileall`: passed
- import smoke: blocked in this environment because `torch` is not installed, so even a stubbed-FlashAttention import check cannot execute here without adding heavyweight dependencies
- CPU start-to-train smoke: not attempted because the script hard-requires CUDA and the Hopper/FlashAttention path; in a fuller local environment the safest next smoke would be a stubbed-FlashAttention import/constructor check followed by a tiny CPU forward/backward pass
