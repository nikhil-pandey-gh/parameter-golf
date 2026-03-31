# Early Shared MLP + Bigger Bigram

## Hypothesis

The strongest non-TTT training stack in this repo already spends most of its parameters in the per-layer MLPs. Recent parameter-sharing work suggests that generative transformers can often share depth-local capacity more profitably than they share everything uniformly, and the newest recurrence paper I found argues that extra reuse is most helpful in earlier layers. My hypothesis is that, for this repo's 11-layer U-Net-like stack, the earliest encoder MLPs are still redundant enough to share, and that the bytes saved are better reinvested into local lexical memory via a larger BigramHash table.

Concretely, this candidate shares the MLP weights for layers `0,1` and `2,3`, keeps all attention layers unique, keeps all late layers unique, switches the MLP activation to LeakyReLU(0.5)^2, and increases the default `BIGRAM_VOCAB_SIZE` from `2048` to `4096`.

## Why this is promising for this repository

The repo history points in a consistent direction:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` is the strongest non-TTT training-only base and already includes the mature 11L/XSA/partial-RoPE/LN-scale/EMA/GPTQ-lite stack.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` shows LeakyReLU(0.5)^2 is a cheap, real win on the same model family.
- `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/` and `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/` both suggest that extra local lexical memory from larger BigramHash tables can be valuable when the artifact budget allows it.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/` reported that naive full layer recurrence was bad, so this candidate deliberately avoids whole-model looping and instead applies selective early-layer sharing only.

That combination makes this a "parameter reallocation" candidate rather than a "more compute" candidate: compress redundant early MLP capacity, keep the successful late-stack intact, and spend the recovered bytes on a slightly bigger explicit local-context memory.

## External research that informed it

- **ALBERT** (`arXiv:1909.11942`) argues that cross-layer sharing can preserve quality while cutting parameters.
- **Subformer** (`arXiv:2101.00234`) is especially relevant here because it studies parameter sharing for *generative* transformers and finds sandwich-style sharing better than naive uniform sharing.
- **Intra-Layer Recurrence in Transformers for Language Modeling** (`arXiv:2505.01855`) reports that earlier layers benefit most from reuse, which is why this candidate shares only the earliest MLPs.
- **A Survey on Transformer Compression** (`arXiv:2402.05964`) reinforces the idea that efficient architecture design and quantization should be combined instead of treated separately.

## Chosen base implementation

I used `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py` as the implementation base rather than the 2026-03-23 TTT record. The 2026-03-22 script already has the strongest non-TTT stack in a relatively conventional layout, so it is the cleanest place to test a new architectural idea without mixing in parameter banking and evaluation-time adaptation.

## What changed versus the base

1. The per-layer MLP modules were lifted out of `Block` and stored in a shared `ModuleList`, so the saved state dict truly contains fewer MLPs instead of just aliasing duplicate parameter names.
2. The default sharing pattern is `SHARED_MLP_GROUPS=0,1;2,3`, which shares only the earliest encoder MLPs.
3. The MLP activation changed from `relu^2` to `LeakyReLU(0.5)^2`, following the later 2026-03-23 record.
4. The default `BIGRAM_VOCAB_SIZE` changed from `2048` to `4096` to spend part of the recovered parameter budget on a bigger local lexical table.
5. The default `LATE_QAT_THRESHOLD` is set to `0.0` so this candidate stays focused on the sharing hypothesis instead of relying on a compile-sensitive late-QAT path.
6. Logging now prints the active shared-MLP groups and the number of unique MLP modules.

## Files added

- `train_gpt.py` — self-contained candidate training/eval/export script.
- `README.md` — this note.

## How to run

From the repo root:

```bash
cd candidates/202603310740_early-shared-mlp-bigram
RUN_ID=early_shared_mlp_bigram \
SEED=1337 \
BIGRAM_VOCAB_SIZE=4096 \
SHARED_MLP_GROUPS=0,1\;2,3 \
MLP_NEGATIVE_SLOPE=0.5 \
LATE_QAT_THRESHOLD=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The important candidate-specific knobs are:

- `SHARED_MLP_GROUPS` — semicolon-separated layer groups that share MLP weights.
- `MLP_NEGATIVE_SLOPE` — LeakyReLU negative slope before squaring.
- `BIGRAM_VOCAB_SIZE` — larger explicit bigram memory funded by MLP sharing.

Evaluation/export behavior remains the same as the 2026-03-22 base: EMA application, GPTQ-lite-inspired int6 quantization, and sliding-window evaluation.

## Main risks / tradeoffs

- Early MLP sharing may regularize the model usefully, but it may also remove too much layer-specific capacity and blunt the gains from the larger BigramHash table.
- Sharing only the MLPs is intentionally conservative; if the effect is too small, the next follow-up would be to try a slightly wider sharing window or to share only the *up* projection while keeping each *down* projection unique.
- Because this candidate keeps the mature 11L stack but changes parameter allocation, the real question is not whether it trains, but whether the byte savings are converted into enough lexical bias to offset the lost early-layer specialization.

## Validation

I ran the lightweight checks that fit this environment:

```bash
python -m compileall candidates/202603310740_early-shared-mlp-bigram/train_gpt.py
python -m py_compile candidates/202603310740_early-shared-mlp-bigram/train_gpt.py
```

Outcome: both passed.

I also checked whether a real local smoke test was feasible:

```bash
python - <<'PY'
import importlib.util
for name in ['torch', 'sentencepiece', 'flash_attn_interface']:
    print(f'{name}={bool(importlib.util.find_spec(name))}')
PY
```

Outcome: `torch=False`, `sentencepiece=False`, `flash_attn_interface=False`.

So I did **not** claim a CPU startup smoke test here: the current environment does not have the required runtime ML dependencies installed, and the candidate intentionally targets the same CUDA/FlashAttention-style execution path used by the strongest recent records.
