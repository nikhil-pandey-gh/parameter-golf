# Candidate: Mirror-Shared MLP U-Net

## Hypothesis

The current winning line in this repository has already harvested most of the obvious no-cost gains: sliding-window eval, deeper 11-layer stacks, 3x MLPs, SmearGate + BigramHash, XSA, EMA/SWA, Partial RoPE, and better post-training quantization. A remaining underexplored axis is **cross-layer parameter sharing without extra compute**.

This candidate shares the expensive MLP core weights across **mirrored U-Net layers** (`0↔10`, `1↔9`, `2↔8`, `3↔7`, `4↔6`, with layer `5` left unique), while keeping each layer's own norms, residual mixing, attention stack, output scales, and a small per-layer hidden-channel scale vector. The goal is to improve parameter efficiency and artifact compressibility **without** paying the wall-clock penalty that hurt naive depth recurrence in earlier experiments.

## Why this is promising for this repository

Repository review suggests three things at once:

1. **Naive recurrence is a bad fit for a 10-minute budget.** The non-record `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md` reports `Layer recurrence ×2` as the worst ablation, and `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md` also notes that depth recurrence likely needs more steps than the track allows.
2. **The current strong models already have U-Net symmetry.** The best pre-TTT records consistently keep U-Net-style encoder/decoder skip structure, so mirrored layers are a natural place to try partial tying.
3. **Artifact-aware training keeps winning.** Strong runs repeatedly improve quality by making the model easier to quantize and compress rather than only adding raw parameters.

That makes mirror-sharing a better fit than adding more recurrent depth: it keeps the same 11-layer compute path, but asks whether some of the MLP capacity can be reused more efficiently and then compressed more cleanly.

## Base implementation and prior record influence

The code starts from the strongest clean pre-TTT stack I could adapt surgically:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

It also explicitly borrows ideas validated elsewhere in the repo:

- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/README.md`
  for the stable 11-layer + XSA + EMA backbone.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md`
  for Partial RoPE + LN scale as a zero-parameter improvement.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`
  for the LeakyReLU(0.5)^2 MLP activation gain.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md`
  for the negative evidence against extra-compute recurrence.

There were **no prior folders under `candidates/`** when this candidate was created.

## External research that informed this candidate

This design is motivated by a conservative intersection of older and newer parameter-sharing work:

- **ALBERT: A Lite BERT for Self-supervised Learning of Language Representations** (Lan et al., arXiv:1909.11942)
  established cross-layer sharing as a practical path to better parameter efficiency.
- **Basis Sharing: Cross-Layer Parameter Sharing for Large Language Model Compression** (Wang et al., arXiv:2410.03765)
  argues that cross-layer sharing can improve compression quality, especially at high compression ratios.
- **Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA** (Bae et al., arXiv:2410.20672)
  shows that strict tying works better when paired with small depth-specific relaxations rather than forcing every layer to be identical.
- **Intra-Layer Recurrence in Transformers for Language Modeling** (Nguyen and Lin, arXiv:2505.01855)
  reinforces the idea that recurrence/sharing should be applied selectively rather than naively everywhere.

This candidate does **not** copy those papers literally. Instead it takes the minimum change that matches this repo's constraints: share only the largest repeated submodule class (the MLPs), keep attention unique, and add tiny per-layer hidden scales as a cheap relaxation mechanism.

## What changed vs the chosen base

Compared with `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`, this candidate makes four deliberate changes:

1. **Mirror-shared MLP cores**
   - Decoder-side mirrored layers reuse the encoder-side MLP module.
   - Effective sharing pattern for 11 layers: `[0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]`.
   - This keeps the compute graph at 11 layers, unlike looped/depth-recurrent variants.

2. **Per-layer hidden-channel relaxation**
   - Each block keeps a learned `shared_mlp_hidden_scale` vector.
   - This is a tiny depth-specific adapter inspired by relaxed recursive / layer-wise adaptation work.

3. **LeakyReLU(0.5)^2 MLP activation**
   - Adopted from the strongest later record because it is a very cheap change with positive evidence in this repo.

4. **Larger default BigramHash budget**
   - Default `BIGRAM_VOCAB_SIZE` is bumped from `2048` to `3072` to spend part of the expected artifact savings on a repo-proven local-context feature.

One implementation detail matters for the artifact path: because shared weights appear under multiple state-dict keys, the candidate's export path records an **alias map** and stores the shared MLP tensors only once before quantization/compression. This is what turns training-time sharing into actual artifact savings.

## How to run / evaluate

From the candidate directory:

```bash
cd candidates/202603312003_mirror-shared-mlp

RUN_ID=mirror_shared_mlp \
MIRROR_SHARE_MLP=1 \
MLP_NEGATIVE_SLOPE=0.5 \
BIGRAM_VOCAB_SIZE=3072 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script resolves default dataset/tokenizer paths relative to the repository root, so it can be launched directly from this directory without rewriting `DATA_PATH` or `TOKENIZER_PATH`.

Important inherited defaults from the base stack:

- `NUM_LAYERS=11`
- `TRAIN_SEQ_LEN=2048`
- `XSA_LAST_N=4`
- `ROPE_DIMS=16`
- `LN_SCALE=1`
- `VE_ENABLED=1`, `VE_DIM=128`, `VE_LAYERS=9,10`
- EMA export path + GPTQ-lite-style int6 roundtrip + stride-64 sliding eval

## Validation

Commands run in this workflow:

```bash
python -m compileall candidates/202603312003_mirror-shared-mlp/train_gpt.py
```

Outcome:

- `python -m compileall` **passed** for `candidates/202603312003_mirror-shared-mlp/train_gpt.py`.
- A dependency probe in this workflow container reported `torch=False` and `flash_attn_interface=False`.
- A minimal runtime smoke test was therefore **not feasible in this workflow container**, and the script's real execution path also requires CUDA + FlashAttention.

## Main expected risks / tradeoffs

- **Sharing could over-regularize the MLP path.** If mirrored encoder/decoder layers really need different nonlinear subspaces, this may hurt despite the per-layer scales.
- **Artifact savings may be offset by worse optimization.** Better compression does not help if the shared model underfits within 600 seconds.
- **The larger bigram table may or may not be the right place to spend saved bytes.** If the sharing win is small, reverting to `2048` buckets or moving the budget to VE may be better.
- **Late QAT remains inherited from the base stack.** The main hypothesis here is the sharing scheme, not a fresh quantization path.

## Suggested next experiments if this starts to work

1. Share only the **MLP projection** (`proj`) while leaving `fc` unique.
2. Compare mirrored sharing against a **shallower cycle** pattern like `[0,1,2,3,4,5,0,1,2,3,4]`.
3. Replace the diagonal relaxation vector with a tiny low-rank adapter on the shared MLP output.
4. Spend the recovered bytes on `VE_DIM` instead of `BIGRAM_VOCAB_SIZE`.
