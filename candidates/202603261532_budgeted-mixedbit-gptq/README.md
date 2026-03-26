# Budgeted Mixed-Bit GPTQ-lite

## Hypothesis

The current frontier stack is already close to saturating obvious training-side tweaks, but its export path still spends nearly the same bit-width on every large attention/MLP matrix. A fixed 16 MB artifact budget should work better if it is treated like a budget allocator: downgrade the most compression-tolerant matrices to 5-bit, use 6-bit as the default anchor, and only spend 7-bit on the tensors whose reconstruction error drops the most per added byte.

In other words: keep the strongest known 2026-03-23 training/eval stack intact, but replace the fixed int6 GPTQ-lite export with a byte-budgeted 5/6/7-bit GPTQ-lite variant that chooses precision by reconstruction benefit per byte.

## Why this is promising for this repository

The records show two consistent patterns:

- small export improvements still matter at the frontier (`2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` gained from a better clip search alone), and
- older mixed-precision records already proved that hard-coded int5/int6 splits can buy useful capacity or artifact headroom (`2026-03-20_10L_Int5MLP_MuonWD04_SWA50`).

That makes mixed-bit export the most direct next lever that is both underexplored on the latest stack and tightly aligned with the challenge's actual bottleneck: post-training quality under a strict artifact cap.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - chosen base implementation and latest strongest stack
  - provides LeakyReLU(0.5)^2, parameter banking + parallel Muon, XSA, partial RoPE, VE, legal TTT, and GPTQ-lite int6 export

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - showed that quantizer refinement alone is still worth measurable BPB on a near-frontier model
  - specifically motivated reusing the per-row percentile clip search idea instead of swapping to a totally different PTQ recipe

- `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/`
  - demonstrated that mixed int5/int6 allocation can be worthwhile under this artifact budget
  - this candidate changes the rule from a hand-written architectural split to a data-driven byte-budgeted split

## External research that informed it

- **SpinQuant** — Liu et al., arXiv:2405.16406
  - argues that quantization difficulty is highly non-uniform and that some parameterizations are much easier to quantize than others
  - I did not implement learned rotations here, but it reinforced the idea that a uniform bit-width is leaving quality on the table

- **Channel-Wise Mixed-Precision Quantization for Large Language Models (CMPQ)** — Chen et al., arXiv:2410.13056
  - motivates allocating precision adaptively instead of uniformly, with gains even under modest extra memory

- **ROSAQ** — Yoon et al., arXiv:2506.13472
  - saliency-aware mixed-precision quantization is effective because some channels/tensors matter much more than others
  - this candidate uses a simpler proxy: reconstruction benefit per byte instead of PCA saliency

- **SFMP** — Nie et al., arXiv:2602.01027
  - mixed precision under tight memory budgets should be treated as an allocation problem, not just a fixed-format export
  - this candidate adopts that lesson in the lightest repo-compatible way: greedy byte-neutral 5/6/7-bit assignment over existing GPTQ-lite candidates

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

This candidate keeps the training stack and evaluation logic essentially unchanged, but changes the export path in four ways:

1. **Budgeted mixed-bit GPTQ-lite**
   - eligible 2D attention/MLP matrices now build 5-bit, 6-bit, and 7-bit GPTQ-lite candidates
   - each candidate still uses the same per-row percentile clip search family as the 2026-03-22/2026-03-23 stack

2. **Greedy byte-budget allocator**
   - starts from a fixed 6-bit baseline
   - upgrades tensors to 7-bit when their MSE improvement per added byte is strong enough
   - funds those upgrades by downgrading low-sensitivity tensors to 5-bit when their MSE penalty per saved byte is cheaper
   - default budget is byte-neutral relative to packed 6-bit payloads (`MIXBITS_EXTRA_BYTES=0`)

3. **True low-bit packing before compression**
   - selected 5/6/7-bit weights are packed into `uint8` payloads before `lzma`
   - this makes the mixed-bit budget real instead of relying on `int8` tensors with implicit unused headroom

4. **Low-cost local validation support**
   - FlashAttention import is now optional for smoke testing
   - CPU fallback uses PyTorch SDPA when FlashAttention is unavailable
   - `SMOKE_TEST_MODE=1` runs a quantization pack/unpack self-test plus a tiny CPU model forward/backward pass

## How to run

Run from this directory so the artifact only includes the candidate code:

```bash
cd candidates/202603261532_budgeted-mixedbit-gptq

NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
MIXBITS_BITS=5,6,7 MIXBITS_BASE_BITS=6 MIXBITS_EXTRA_BYTES=0 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful toggles:

- `MIXBITS_BITS=6` approximates the fixed-bit export path again.
- `MIXBITS_EXTRA_BYTES=<N>` lets you spend additional packed-payload bytes above the neutral 6-bit baseline and must be non-negative.
- `SMOKE_TEST_MODE=1 python train_gpt.py` runs the built-in CPU smoke path instead of full training.

## Validation run in this workflow

Commands run here:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603261532_budgeted-mixedbit-gptq/train_gpt.py
```

Outcome:

- passed

Command run for the candidate smoke test:

```bash
. /tmp/gh-aw/agent/pgolf-venv/bin/activate && SMOKE_TEST_MODE=1 python candidates/202603261532_budgeted-mixedbit-gptq/train_gpt.py
```

Outcome:

```text
smoke_ok lowbit=4 payload_delta=0 worst_mse=9.057676e-04 model_loss=6.921875
  blocks.0.attn.c_q.weight: 6b
  blocks.0.attn.proj.weight: 6b
  blocks.0.mlp.fc.weight: 6b
  blocks.0.mlp.proj.weight: 6b
```

Notes:

- the smoke test validated pack/unpack for 5/6/7-bit probes, exercised the mixed-bit exporter on large synthetic matrices, and ran a tiny CPU forward/backward pass through the actual model graph
- a full GPU training run was not feasible in this workflow environment

## Main expected risks and tradeoffs

- **Weight-MSE is only a proxy.** The allocator uses reconstruction error per byte, not activation-aware saliency or Hessian information. That is simpler and robust, but it may miss the true most-sensitive tensors.
- **Byte-neutral packed payload is conservative.** The default allocator does not automatically spend all possible artifact headroom; it is designed to be safe first. The best run may want a small positive `MIXBITS_EXTRA_BYTES`.
- **Compressor interaction is nonlinear.** Better packed payload bytes do not perfectly predict final `lzma` bytes, so the best allocation under compressed size may differ slightly from the greedy raw-byte solution.
- **The latest stack is already strong.** Gains here are likely incremental, not architectural step changes. This is intentionally a surgical candidate rather than a broad redesign.
