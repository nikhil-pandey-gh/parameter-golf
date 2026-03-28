# Candidate: Mirror-Shared U-Net

## Hypothesis

The strongest unexplored parameter-budget idea in this repository is **fixed-compute mirrored layer sharing**: keep the proven 11-layer U-Net-style logical depth and 2048-token training regime, but replace per-layer block weights with a **mirror-shared block bank** so encoder/decoder counterparts reuse the same core weights. The saved artifact bytes are then paired with **private per-layer control tensors** (`q_gains`, residual mixes, attention/MLP scales) plus the already-proven **LeakyReLU(0.5)^2** activation, so the model keeps layer-specific behavior without paying for 11 fully distinct blocks.

In short: preserve depth and wall-clock behavior, cut duplicated weights, and keep expressivity where it is cheapest.

## Why this is promising for this repository

The record history shows a clear theme: the best runs keep spending saved bytes on more useful capacity, better eval, and gentler compression. Mixed low-bit quantization, EMA/SWA, XSA, Partial RoPE, bigger BigramHash tables, and embedding exceptions all follow that pattern.

What has **not** been seriously explored is parameter sharing that does **not** add extra forward passes. The notable negative recurrence result in `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md` was a different experiment: it doubled effective depth by reusing layers, which also destroyed step throughput in a fixed wall-clock budget. This candidate avoids that trap by keeping the same 11 logical layers and merely reducing the number of *unique* block tensors.

## Prior records and experiments that influenced this candidate

Primary local influences:

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - strong clean 11-layer non-TTT stack with XSA, EMA, Partial RoPE, LN scale, SmearGate, and BigramHash.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - confirmed that the later 11-layer stack wanted `WARMDOWN_ITERS=3500`, EMA, and tight artifact budgeting.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - showed that `LeakyReLU(0.5)^2` was a real gain on a strong stack, not a toy activation tweak.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
  - important negative result: naive extra recurrence hurt because it consumed training-time budget. This candidate is designed specifically to *not* make that mistake.

## External research that informed it

- **ALBERT** (`arXiv:1909.11942`) showed that cross-layer parameter sharing can greatly reduce parameter count while preserving much of the benefit of deeper transformers.
- **Universal Transformer** (`arXiv:1807.03819`) argued that recurrent depth and shared transformer computation can retain transformer parallelism while improving generalization.
- **Intra-Layer Recurrence in Transformers for Language Modeling** (`arXiv:2505.01855`) reported that selectively reusing transformer computation is a plausible parameter-efficiency direction for LMs, especially when not applied indiscriminately.
- **SCORE: Replacing Layer Stacking with Contractive Recurrent Depth** (`arXiv:2603.10544`) is a recent 2026 result reinforcing the broader thesis that shared-depth refinement can reduce parameter count while staying competitive.

The implementation here is intentionally simpler than those papers: it adopts the core **shared-depth / shared-weight** idea, but keeps the repo's training and evaluation machinery familiar.

## What changed versus the chosen base implementation

This candidate starts from the repo's strongest clean pre-TTT lineage, specifically the 11-layer XSA/EMA/Partial-RoPE stack in `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`, while nudging defaults toward the later warmdown regime used by `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`.

Key changes in `train_gpt.py`:

1. **Mirror-shared block bank**
   - replaces 11 distinct transformer blocks with a mirrored bank of `ceil(num_layers / 2)` unique block cores.
   - default 11-layer mapping is:
     - `0,1,2,3,4,5,4,3,2,1,0`
   - this keeps the logical encoder/decoder depth unchanged while making the serialized model genuinely smaller.

2. **Private per-layer control tensors**
   - each logical layer keeps its own:
     - `q_gains`
     - residual mix
     - attention scale
     - MLP scale
   - this is the main hedge against the expressivity loss from sharing the heavy block weights.

3. **LeakyReLU(0.5)^2 MLP**
   - imported from the best current record family to make the shared block cores more expressive at no parameter cost.

4. **Stronger defaults for this candidate**
   - `NUM_LAYERS=11`
   - `WARMDOWN_ITERS=3500`
   - `MATRIX_LR=0.025`
   - `SCALAR_LR=0.025`
   - `TIED_EMBED_LR=0.035`
   - `MUON_WD=0.04`, `ADAM_WD=0.04`
   - `MUON_MOMENTUM=0.99` with `0.92 -> 0.99` warmup over 1500 steps
   - `XSA_LAST_N=4`
   - `ROPE_DIMS=16`
   - `LN_SCALE=1`
   - `EMA_ENABLED=1`
   - `BIGRAM_VOCAB_SIZE=3072`
   - `MIRROR_SHARE=1`

5. **No extra files or infrastructure**
   - the candidate stays self-contained in a single `train_gpt.py`, matching repository norms.

## How to run / evaluate

From this candidate directory:

```bash
cd candidates/202603280852_mirror-shared-unet
RUN_ID=mirror_shared_unet \
DATA_PATH=../../data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful overrides:

```bash
MIRROR_SHARE=0          # disable sharing for an apples-to-apples ablation
BIGRAM_VOCAB_SIZE=2048  # revert to a smaller side-channel budget
SEED=1337               # fixed-seed comparison
```

Evaluation behavior remains the same as the chosen base stack: standard validation during training plus post-roundtrip sliding-window evaluation.

## Main expected risks / tradeoffs

- **Layer diversity risk**: even with per-layer controls, shared block weights may still collapse useful specialization across depth.
- **Too-conservative reinvestment**: the saved bytes are only partly reinvested (larger BigramHash + layer-private controls). A better follow-up may widen a more targeted submodule instead.
- **Interaction with XSA placement**: sharing mirrored blocks means the deepest XSA-enabled logical layers may reuse cores that also appear earlier in the stack.
- **Compile/QAT caution**: this candidate does not depend on late-QAT because prior repo work documented a `torch.compile` constant-folding pitfall in that path.

## Validation

Commands run in this repository:

```bash
python -m compileall candidates/202603280852_mirror-shared-unet/train_gpt.py
python - <<'PY'
import importlib.util
print(importlib.util.find_spec('torch'))
PY
```

Outcomes:

- `python -m compileall ...` **passed**.
- The environment's default Python reported `torch` as **unavailable** (`None`), so a meaningful CPU runtime smoke test was **not feasible here** without installing heavyweight dependencies. Because this candidate inherits the repo's PyTorch + FlashAttention style runtime, I did not fabricate a fake success beyond syntax validation.

## Suggested next experiments

1. Compare `MIRROR_SHARE=1` vs `MIRROR_SHARE=0` on the same seed and wall-clock budget.
2. Try sharing only the encoder/early decoder layers while keeping the deepest 2-3 blocks unique.
3. Reinvest more of the saved bytes into `BIGRAM_VOCAB_SIZE`, VE-style side channels, or a slightly wider MLP.
4. If this direction looks promising, port it onto the later GPTQ-lite/VE128 stack rather than keeping it on the simpler pre-TTT code lineage.
