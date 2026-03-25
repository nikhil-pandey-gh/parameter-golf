# Training-Only MTP on the Mar-23 Stack

## Hypothesis

A small **training-only** multi-token prediction (MTP) objective can improve sample efficiency on this repository's strongest 11-layer stack **without adding export bytes**. The key idea is to predict one extra future token from the shared hidden state during training, then drop the auxiliary head before artifact export so the final submission size stays dominated by the base model.

## Why this is promising for this repository

The records show that the challenge is already operating near the 16MB artifact ceiling, so the best late gains have come from better use of the existing bytes: stronger quantization, better weight averaging, better evaluation, and careful structural refinements. A training-only auxiliary head fits that pattern well because it adds optimization signal during the 10-minute run but does not need to survive into the compressed artifact.

The local code review made this especially attractive here: the current best record already contains dormant MTP plumbing and already strips `mtp_heads` out of the exported state dict, but its submitted configuration never enabled those heads and the Mar-23 script does not wire them to an optimizer by default. That makes MTP a high-leverage, low-infrastructure next experiment rather than a broad rewrite.

## Prior records that influenced this candidate

The main base is the current best record:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`

That run established the strongest current trunk: 11 layers, 512 width, 3x MLP, partial RoPE, LN scale, XSA on the deepest layers, VE on late layers, GPTQ-lite int6 + `lzma`, Parameter Banking + Parallel Muon, and LeakyReLU^2. I copied that script and only changed the candidate-local file.

Two nearby records also mattered:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`

These reinforced the pattern that the strongest non-TTT gains come from improving the same 11-layer core stack rather than changing the whole architecture. The repo-wide review also showed repeated wins from seq2048, EMA/SWA-style averaging, mixed low-bit compression, XSA, and bigram-side features, while recurrence and slower activations were generally poor fits for a fixed 10-minute wall clock.

## External research that informed the choice

I considered several literature-backed directions, especially:

- **Multi-token prediction**: Fabian Gloeckle et al., *Better & Faster Large Language Models via Multi-token Prediction* (`arXiv:2404.19737`). The paper reports better sample efficiency by predicting multiple future tokens from a shared trunk using extra output heads.
- **DeepSeek-V3**: *DeepSeek-V3 Technical Report* (`arXiv:2412.19437`), which explicitly notes a multi-token prediction training objective as part of its stronger pretraining recipe.
- **Primer**: David So et al., *Primer: Searching for Efficient Transformers for Language Modeling* (`arXiv:2109.08668`), which motivated a plausible alternative candidate based on depthwise Q/K/V convolutions.
- **AWQ**: Ji Lin et al., *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration* (`arXiv:2306.00978`), as a competing compression-first direction.

Primer-style Q/K/V depthwise convolutions were a serious alternative, but I chose MTP here because it is both research-backed and unusually easy to test in this repo: the base script already had dormant hooks and export stripping, so the candidate can focus on turning a promising idea on rather than adding a large new mechanism.

## What changed versus the chosen base implementation

Relative to `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`, this candidate makes four focused changes:

1. **Enable MTP by default**
   - `MTP_NUM_HEADS=1`
   - `MTP_LOSS_WEIGHT=0.15`

   This is intentionally conservative: one extra future-token head should keep the training-time overhead modest while still testing whether auxiliary future-token supervision helps this tiny-model regime.

2. **Wire the MTP heads to an optimizer**
   - The base Mar-23 script instantiates `mtp_heads` and removes them at export time, but does not give them their own optimizer in the main training path.
   - This candidate adds a dedicated Adam optimizer for the auxiliary head weights and includes those parameters in the replicated-grad all-reduce path.

3. **Initialize the MTP head from the main token head / tied embedding**
   - `MTP_INIT_FROM_MAIN_HEAD=1` by default.
   - With tied embeddings, the MTP head starts from `tok_emb.weight` instead of zeros.

   The motivation is challenge-specific: in a short 10-minute run, giving the auxiliary head a useful classifier prior should make it provide cleaner gradients sooner than a pure zero-init head.

4. **Make the default data/tokenizer paths repo-root aware**
   - The copied record script used `./data/...` defaults that are inconvenient when running directly from the candidate directory.
   - This candidate resolves defaults relative to the repository root so it can be launched from inside `candidates/202603251926_mtp-aux-heads/` without extra path overrides.

The rest of the Mar-23 stack is left intact, including optional legal TTT support. Candidate defaults keep `TTT_ENABLED=0` so the effect of training-only MTP can be measured cleanly first.

## How to run

From this candidate directory:

```bash
cd candidates/202603251926_mtp-aux-heads

RUN_ID=mtp_aux_heads \
SEED=1337 \
MTP_NUM_HEADS=1 \
MTP_LOSS_WEIGHT=0.15 \
MTP_HEAD_LR=0.008 \
MTP_INIT_FROM_MAIN_HEAD=1 \
TTT_ENABLED=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful ablations:

- Disable the candidate idea entirely: `MTP_NUM_HEADS=0`
- Keep MTP but use zero-init heads: `MTP_INIT_FROM_MAIN_HEAD=0`
- Try a slightly stronger auxiliary target: `MTP_NUM_HEADS=2`
- If pre-TTT quality improves, try re-enabling the inherited evaluation recipe: `TTT_ENABLED=1`

## Validation

I ran the lightweight validation that fits this environment **from the repository root**:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603251926_mtp-aux-heads/train_gpt.py
```

Outcome: **passed**. The baseline scripts, `data/` utilities, and the candidate script all compiled successfully to bytecode.

If you are already inside `candidates/202603251926_mtp-aux-heads/`, the equivalent local check is:

```bash
python -m compileall train_gpt.py
```

A minimal CPU runtime smoke test was **not feasible in this environment**. The current runner does not have runtime dependencies like `torch`, `numpy`, `sentencepiece`, or `flash_attn_interface` installed, and this script inherits the CUDA/FlashAttention runtime path from the record stack. Because of that, I limited validation here to syntax-level checks only.

## Main risks and tradeoffs

- **Throughput risk**: even one extra vocab head adds training-time compute, so the model may complete fewer optimizer steps within 600 seconds.
- **Objective mismatch risk**: better future-token supervision may not translate into lower next-token BPB on this benchmark.
- **Head-init risk**: copying the main head into the MTP head may help early learning, but it could also reduce useful diversity between objectives.
- **Interaction risk with TTT**: if MTP improves the pre-TTT model, it may or may not compose cleanly with the inherited score-first TTT recipe.

## Suggested next experiments

If this candidate looks promising, the next sequence I would test is:

1. `MTP_NUM_HEADS=1` versus `0` on the same seed and hardware.
2. `MTP_INIT_FROM_MAIN_HEAD=1` versus `0`.
3. `MTP_NUM_HEADS=2` if one head is neutral-to-positive.
4. Re-enable `TTT_ENABLED=1` only after confirming the pre-TTT stack improved.
