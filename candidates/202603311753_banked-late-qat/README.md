# Bank-Aware Late QAT on the 2026-03-23 Stack

## Hypothesis

The current best banked stack already exports well with GPTQ-lite int6 + lzma, but its late-QAT path still under-trains the dominant export-critical weights: the large banked attention and MLP matrices. If we extend STE int6 fake quantization to those banked weights during the late warmdown window, the model should train closer to its final low-bit export regime and reduce the remaining train/export mismatch at almost zero parameter cost.

## Why this is promising for this repository

The record history suggests the biggest reliable gains now come from cheap, stackable changes on top of the 11-layer XSA / EMA / quantization backbone rather than broad architecture resets.

The strongest pure training/export stack in `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` improved largely through better export behavior: GPTQ-lite clip search, EMA, and slightly earlier late-QAT.

The current overall best run in `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` keeps that export-focused backbone, but moves the main block weights into 3D parameter banks for Parallel Muon. In that banked setup, the previous `CastedLinear._qat_enabled` path only fake-quantizes the small `CastedLinear` modules, not the banked attention/MLP matrices that dominate the artifact.

That makes bank-aware late QAT a direct fit for the repo’s current bottleneck: improving the final 16 MB export without changing the high-level model family.

## Prior experiments that influenced this candidate

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Best overall stack.
  - Contributes the base 11-layer banked / Parallel Muon / XSA / partial-RoPE / VE / legal-TTT implementation.

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Strongest evidence that export-aware changes still matter at the frontier.
  - Motivates preserving GPTQ-lite int6 export while improving train-time quantization robustness.

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - Important cautionary note: its README documents that the earlier late-QAT path was effectively dead because `torch.compile` constant-folded the class flag.
  - This candidate explicitly addresses that failure mode by recompiling when late QAT is enabled.

- `records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/`
  - Shows that extended training can get pre-quant quality much lower than final post-quant quality, reinforcing that quantization/export is still a primary bottleneck.

- Prior candidates:
  - None existed in the repository at implementation time.

## External research that informed the choice

- **Squat: Quant Small Language Models on the Edge** (arXiv:2402.10787)
  - Emphasizes that QAT remains especially relevant for small language models where full-parameter training is feasible and deployable low-bit inference matters.
  - That maps well to this challenge’s tiny-model, hard artifact-budget setting.

- **Rope to Nope and Back Again: A New Hybrid Attention Strategy** (arXiv:2501.18795)
  - I reviewed this because the repo’s recent wins already lean on partial RoPE and hybrid positional behavior.
  - It looks promising, but it overlaps with directions already explored here and would require a broader attention redesign than this candidate.

- **GLU Variants Improve Transformer** (arXiv:2002.05202)
  - I reviewed this because the repo has both a strong LeakyReLU² record and a non-record SwiGLU exploration.
  - It is still a good future direction, but bank-aware quantization looked like the cleaner next move for the current best banked stack.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

This candidate makes one focused change:

1. Add `ste_fake_quant_int6_per_row()` and `maybe_fake_quantize_int6()` helpers.
2. Apply that fake quantization during training to:
   - banked attention Q / K / V / output weights
   - banked MLP up / down weights
   - existing `CastedLinear` modules (retaining the prior behavior there)
3. Recompile the model exactly when late QAT turns on so the fake-quant branch is not optimized away by `torch.compile`.

The export path is intentionally left alone:

- GPTQ-lite int6 per-row export remains the same.
- lzma compression remains the same.
- the architecture, optimizer, EMA, XSA, partial RoPE, VE layers, and legal TTT code all remain inherited from the current best stack.

## Expected artifact impact

There are no new persistent model tensors. The only byte increase should come from the added Python code, so this should stay much closer to the base artifact size than a capacity-increasing architecture change.

## How to run

From the candidate directory:

```bash
cd candidates/202603311753_banked-late-qat
```

Training/export-only comparison against the current training stack:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

To compare against the current best full stack including legal score-first TTT:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
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

## Validation

I ran the lightweight validation that fits this environment:

```bash
python -m compileall candidates/202603311753_banked-late-qat/train_gpt.py
```

Outcome: success.

I also ran the repository’s broader low-cost syntax check before editing:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data
```

Outcome: success.

### CPU-only smoke test

I did **not** run a full runtime smoke test in this environment. This script imports `flash_attn_interface` and hard-requires CUDA (`RuntimeError("CUDA is required")`), so a meaningful CPU-only launch here would fail for environmental reasons rather than candidate logic.

## Main risks and tradeoffs

- **Late-stage throughput hit**: bank-aware fake quantization adds extra per-row reduction and rounding work in the late warmdown region.
- **Recompile overhead**: enabling late QAT now triggers a second `torch.compile` pass by design.
- **Proxy mismatch**: training-time fake quantization uses a cheap row-max int6 proxy, while export still uses GPTQ-lite clip search. That is directionally aligned, but not an exact match.
- **Possible marginal gain**: if the quantization gap on the banked stack is already very small, the improvement may be noise-level and require threshold/scope tuning.
- **Evaluation interaction**: legal TTT can partially mask or amplify training/export improvements, so the first clean comparison should be against the non-TTT export metric as well as the final TTT metric.
