# Candidate: Late Core Sharing

## Hypothesis

The strongest clean 11-layer stack in this repo may be overspending artifact bytes on distinct **late** transformer weights that are partly redundant. This candidate keeps the full **11 logical layer applications**, but shares only the heavy late block cores (attention + MLP weights) across the last four logical layers while leaving the per-layer norms, residual mixing, scaling, skip path, and late value injection controls distinct. The saved bytes are partly reinvested in stronger lexical/value side channels.

## Why this is promising for this repository

- The best non-TTT base already looks mature architecturally: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`.
- Repo history shows that extra effective depth and richer lexical side channels both help when the artifact budget allows them.
- Prior repo notes also show that **naive full recurrence / looping** was weak under the 10-minute budget, so this candidate tries a narrower trade: preserve the same logical depth and wallclock profile, but reduce duplicated late parameters instead of unrolling more steps.
- Recent automated candidate issues already explored **AWQ / SmoothQuant / LSQ** export ideas and **early MLP sharing** ideas; this candidate instead tests a different structural trade focused on the late stack.

## Influences from prior records and prior candidate iterations

### Direct base

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - 11 layers, EMA, GPTQ-lite export, partial RoPE, LN scale, XSA on late layers, shared value embeddings, SmearGate, BigramHash.

### Additional record influences

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - established partial RoPE + LN scale as durable zero-byte wins.
- `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/`
  - showed that better compression can be profitably reinvested into model structure and larger lexical side channels.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - provided the stronger `LeakyReLU(0.5)^2` MLP activation that this candidate carries over into the non-TTT branch.

### Prior candidate iterations reviewed

- Issue `#544` proposed **partial heavy sharing in early non-XSA layers**.
- Issue `#559` proposed **mirror-shared U-Net blocks with LoRA relaxation**.
- Issue `#594` proposed **adjacent pair-shared transformer cores across the 11-layer stack**.
- Issue `#604` proposed **shared early MLPs**.
- Issue `#608` proposed **early shared MLPs plus alias-aware export**.
- Issues `#602`, `#610`, `#612`, and `#614` proposed **AWQ / SmoothQuant / LSQ-style quantization** directions.

This candidate differs from those by targeting only the **late XSA/VE-heavy substack**, avoiding alias-dedup export complexity by storing unique `block_cores` exactly once, and reinvesting more aggressively into **VE placement/dim plus a larger BigramHash** rather than into full-stack mirror sharing or early-layer sharing.

## External research that informed this candidate

- **ALBERT** (arXiv:1909.11942) showed that cross-layer parameter sharing can substantially reduce parameter count with modest quality loss.
- **Universal Transformer** (arXiv:1807.03819) supports the broader idea that repeated transformer computation can be parameter-efficient.
- **Intra-Layer Recurrence in Transformers for Language Modeling** (arXiv:2505.01855) is a cautionary data point: it reports the strongest recurrence results from earlier layers, which is one of the main risks for this late-sharing variant.

## What changed versus the chosen base implementation

1. **Late shared block cores**
   - New defaults: `SHARED_LATE_N=4`, `SHARED_LATE_UNIQUE=2`.
   - The resulting default core map is:
     - logical layers: `0,1,2,3,4,5,6,7,8,9,10`
     - physical cores: `0,1,2,3,4,5,6,7,8,7,8`
   - Heavy attention/MLP weights live in `block_cores`, while per-layer norms, residual mixing, scales, and optional gating stay in `blocks`.

2. **LeakyReLU(0.5)^2**
   - `MLP_NEGATIVE_SLOPE=0.5` by default.

3. **Stronger side channels**
   - `BIGRAM_VOCAB_SIZE` default increased from `2048` to `4096`.
   - `VE_DIM` default increased from `128` to `160`.
   - `VE_LAYERS` default moved from `9,10` to `8,9,10`.

4. **Candidate-local path defaults**
   - `DATA_PATH` and `TOKENIZER_PATH` now resolve relative to the repository root so the script can be launched directly from this candidate directory.

## How to run

From the repository root:

```bash
cd candidates/202604030600_late-core-sharing
RUN_ID=late_core_sharing \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful overrides:

```bash
SHARED_LATE_N=0 SHARED_LATE_UNIQUE=0          # disable sharing
BIGRAM_VOCAB_SIZE=2048 VE_DIM=128 VE_LAYERS=9,10
MLP_NEGATIVE_SLOPE=0.0                         # recover relu^2
```

## Main expected risks and tradeoffs

- Sharing the late heavy cores may remove too much layer-specific capacity; the gains from larger lexical/value side channels may not repay that loss.
- This is deliberately the opposite of recent literature that found **earlier-layer** recurrence strongest, so this should be treated as a targeted candidate, not a low-risk expected record.
- The shared cores also share `q_gain`, attention weights, and MLP weights; only the lightweight shell parameters remain per-layer.
- Artifact savings are real only if the structural sharing harms quality less than the recovered bytes help.

## Validation

### Commands

```bash
python -m compileall candidates/202604030600_late-core-sharing/train_gpt.py
python -m compileall candidates/202604030600_late-core-sharing
```

### Outcomes

- `compileall`: **passed**
- Runtime smoke test: **not feasible in this workflow environment** because the runner does not currently have the required runtime stack installed (`torch`, `numpy`, `sentencepiece`, and `flash_attn_interface` were all unavailable), so only syntax-level validation was safe.
