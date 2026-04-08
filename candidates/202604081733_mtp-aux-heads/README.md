# Export-free MTP auxiliary head on the current SOTA stack

## Hypothesis

A small multi-token prediction (MTP) auxiliary loss should improve trunk sample efficiency in this repository's 600-second training regime by providing denser future-token supervision per forward pass. Because the auxiliary head is trained only and then excluded from export, the candidate can potentially improve the final model **without paying extra artifact bytes at evaluation time**.

## Why this is promising here

- The record lineage has already mined most of the obvious gains from stronger quantization, sliding-window evaluation, XSA, partial RoPE, EMA/SWA, and LeakyReLU^2.
- The current top-record script already contains dormant MTP support **and already strips `mtp_heads.*` from the exported checkpoint**, which makes MTP unusually attractive for a strict 16MB artifact budget.
- The copied base is the strongest current stack in this repo: 11 layers, 512d, 3x MLP with LeakyReLU(0.5)^2, bigram features, XSA on the last 4 layers, partial RoPE, VE, EMA/SWA, GPTQ-lite int6 + lzma, and legal score-first TTT.

## Prior repository runs that informed this candidate

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`: current best overall stack and the source of the dormant MTP path.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`: strongest non-TTT 11L quantization/EMA recipe.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`: useful reminder that training-time efficiency changes can matter more than another export tweak.

No prior `candidates/` directory existed when this candidate was created.

## External research

- **Gloeckle et al., "Better & Faster Large Language Models via Multi-token Prediction" (arXiv:2404.19737)**: predicts multiple future tokens with independent heads on a shared trunk and reports higher sample efficiency plus stronger induction-style behavior.
- **Gerontopoulos et al., "Multi-Token Prediction Needs Registers" (arXiv:2505.10518)**: argues that MTP can be effective with only negligible extra parameters when implemented carefully, reinforcing the appeal of low-cost auxiliary future-token supervision.
- **He, Welleck, Fried, "Reasoning with Latent Tokens in Diffusion Language Models" (arXiv:2602.03769)**: links auxiliary multi-token prediction to better lookahead/global-coherence behavior even outside diffusion models.

## What changed versus the chosen base implementation

Base implementation: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

This candidate makes three surgical changes:

1. **Turns on one auxiliary future-token head by default** with `MTP_NUM_HEADS=1` and `MTP_LOSS_WEIGHT=0.15`. Since the main loss already predicts the next token, this is effectively 2-token prediction overall.
2. **Fixes the dormant MTP path** by actually adding `mtp_heads` weights to the AdamW optimizer. In the copied base script, the README comments said MTP heads should be optimized, but the parameters were never assigned to any optimizer.
3. **Warm-starts the auxiliary head from the tied token embedding matrix** instead of leaving it zero-initialized. Zero-init lets the head learn, but it gives the shared trunk essentially no auxiliary gradient at step 0; copying the tied embedding makes the auxiliary loss useful immediately.

The export path still excludes `mtp_heads.*`, so the serialized model shape used for roundtrip evaluation is unchanged.

## How to run

From this candidate directory:

```bash
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The baked defaults match the intended candidate recipe: `ITERATIONS=9000`, `BIGRAM_VOCAB_SIZE=1536`, `TTT_ENABLED=1`, `TTT_FREEZE_BLOCKS=0`, and the new `MTP_NUM_HEADS=1`.

Useful overrides:

```bash
MTP_NUM_HEADS=0      # disable the candidate idea and recover the copied base behavior
MTP_NUM_HEADS=2      # try a slightly more aggressive horizon
MTP_LOSS_WEIGHT=0.10 # weaker auxiliary loss
MTP_LOSS_WEIGHT=0.20 # stronger auxiliary loss
TTT_ENABLED=0        # isolate the training-time gain before legal TTT
```

## Main risks and tradeoffs

- The extra vocab projection adds training-time compute and may reduce total optimizer steps in the fixed 600-second budget.
- A single future-token head may be too weak to move the needle; multiple heads may help more, but at higher training cost.
- The candidate is built on the current TTT stack, so a pre-TTT win could be diluted or amplified by evaluation-time adaptation.
- Warm-starting from tied embeddings should improve early auxiliary gradients, but it may also slow horizon-specific specialization if the copy stays too close to the main head.

## Validation

- `python -m compileall candidates/202604081733_mtp-aux-heads/train_gpt.py`
- `python -m compileall candidates/202604081733_mtp-aux-heads`

Outcome:

- Python bytecode compilation succeeded.
- A real CPU smoke run is **not** feasible in this environment because this script imports Hopper-specific FlashAttention bindings and hard-requires CUDA during execution.
