# Bank-Aware Late QAT

## Hypothesis

The current best 11-layer stack is already strong on the modeling side, but its dominant attention/MLP weights live in the large parameter banks and still train in float before being quantized at export. This candidate tests whether **late QAT that actually touches those banked weights**, plus **learned per-bank clip multipliers**, can shrink the final int6 round-trip gap more effectively than the existing small-layer-only late-QAT path.

## Why this is promising here

Repository history points to quantization as the recurring bottleneck:

- `2026-03-18_FP16Embed_WD3600` showed that quantization quality can outweigh many architecture tweaks.
- `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` explicitly documented that its late-QAT branch was dead-code-eliminated by `torch.compile`.
- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` improved further with better export-time clipping, implying there is still headroom in the quantization path.
- `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` became the best overall stack, but it still exports the banked weights post hoc.

That makes bank-aware late QAT a better next bet than another large architectural change, especially since prior README notes repeatedly flag recurrence/depth-reuse as net losers under the 10-minute budget.

## Chosen base implementation

This candidate starts from:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

I kept that base because it already combines the strongest current model-side ingredients in this repo:

- 11 layers at 512d
- BigramHash + SmearGate
- XSA on the deepest layers
- Partial RoPE + LN scale
- EMA/SWA
- LeakyReLU^2
- Parameter banking + Parallel Muon
- legal score-first TTT

## Prior work that influenced this candidate

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`: overall base stack and current best leaderboard recipe.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`: GPTQ-lite export clipping and the idea that better quantization alone can still move BPB.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`: important caution that toggling late QAT after `torch.compile` can silently do nothing.
- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600`: strong evidence that quantization-sensitive tensors deserve explicit handling.
- No prior `candidates/` directory existed when this candidate was created.

## External research that informed it

- **EfficientQAT** (Chen et al., 2024/2025, arXiv:2407.11062): motivates block-wise QAT plus end-to-end training of quantization parameters instead of only relying on PTQ.
- **How to Parameterize Asymmetric Quantization Ranges for Quantization-Aware Training** (You et al., 2024, arXiv:2404.16898): motivates learning the quantization range parameterization instead of treating clip ranges as fixed constants.
- **SiLQ: Simple Large Language Model Quantization-Aware Training** (Esser et al., 2025, arXiv:2507.16933): supports the claim that a simple late QAT phase can help without a large extra compute budget.
- **NeUQI** (Lin et al., 2025, arXiv:2505.17595): reinforces that scale initialization and clip choice matter materially in low-bit quantization.

## What changed versus the base implementation

1. **Bank-aware late QAT**
   - Added learnable clip-logit parameters for the four large parameter banks:
     - `qo_bank`
     - `kv_bank`
     - `mlp_up_bank`
     - `mlp_down_bank`
   - During the late QAT phase, the active bank slices are fake-quantized with STE int6 quantization before the forward pass.

2. **Learned clip multipliers**
   - Each active bank slice gets a learnable clip multiplier derived from a logit parameter.
   - The same learned multipliers are reused at export time for the corresponding unbanked weight matrices, so the training-time fake quantization matches the final int6 path more closely.

3. **Compile-safe late-QAT activation**
   - Instead of relying on a class flag that can be constant-folded into the compiled graph, this script **enables QAT and recompiles once** when entering the late-QAT phase.
   - That directly addresses the failure mode documented in the 2026-03-21 record README.

4. **Small-layer QAT consistency**
   - `CastedLinear` modules now also carry learned clip logits and use the same STE int6 fake-quantizer when QAT is enabled.

5. **Candidate-directory usability**
   - The default dataset and tokenizer paths resolve relative to the repository root, so running `train_gpt.py` from this candidate directory does not break on `./data/...` paths.

## How to run

From the candidate directory:

```bash
cd candidates/202604020733_bank-aware-late-qat

NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.18 \
QAT_CLIP_INIT=4.0 QAT_CLIP_FLOOR=0.5 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Notes:

- EMA remains on by default in the script, following the base implementation.
- `LATE_QAT_THRESHOLD=0.18` gives the bank-aware QAT path a slightly longer late window than the earlier 0.15 setting.
- The script still writes the exported model artifact into the current working directory, just like the record scripts.

## How to evaluate

The script keeps the same evaluation structure as the base record:

- full validation eval
- int6 round-trip eval
- sliding-window eval
- legal score-first TTT eval when `TTT_ENABLED=1`

So the most relevant comparison is the final logged:

- `final_int6_sliding_window_s64_exact ...`
- and, if enabled, `legal_ttt_exact ...`

## Main expected risks and tradeoffs

- **Throughput risk**: bank-aware fake quantization adds work late in training, so it may reduce the number of completed steps in the 600s budget.
- **Overfitting the export path**: learned clip multipliers may improve the int6 round-trip but slightly hurt pre-quant quality if the late phase is too aggressive.
- **Interaction risk with TTT**: the candidate inherits legal TTT from the base record, so any gain from better quantization could be partially masked by eval-time adaptation variance.
- **Short QAT window**: if the late-QAT phase is still too short, the learned clip parameters may not move enough to beat pure GPTQ-lite export search.

## Validation

Commands run for this candidate (from the repository root):

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data
python -m compileall candidates/202604020733_bank-aware-late-qat/train_gpt.py
```

Outcomes:

- Both compile checks passed.
- I did **not** run a CPU smoke launch because this script, like the surrounding record scripts, hard-requires CUDA and FlashAttention 3; there is no safe CPU-only execution path in the current repo without adding extra infrastructure.
