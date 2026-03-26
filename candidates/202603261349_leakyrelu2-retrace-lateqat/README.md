# LeakyReLU^2 + Retraced Late QAT + Bigram3072

## Hypothesis

The strongest clean training-time base in this repository is the 2026-03-22 `11L EMA + GPTQ-lite + warmdown3500 + QAT@0.15` stack. This candidate keeps that stack intact, swaps the MLP activation from `ReLU^2` to `LeakyReLU(0.5)^2`, and fixes the repo's broken late-QAT path by forcing a one-time `torch.compile` retrace exactly when late fake quantization is enabled.

The bet is that three effects stack cleanly under the 10-minute / 16MB constraints:

1. `LeakyReLU(0.5)^2` improves per-step learning signal in the MLP, matching the gain seen in the current best record.
2. Late QAT should reduce the GPTQ-lite roundtrip gap, but only if it actually executes. Earlier records documented that the branch was constant-folded away.
3. A moderate `BigramHash` bump to `3072` follows the same direction as prior winning ablations without adding new infrastructure.

## Why this is promising for this repository

- The 2026-03-22 record is the best pre-TTT implementation base in `records/`, reaching `1.1228`/`1.1233` mean with GPTQ-lite, EMA, Tight SWA, Partial RoPE, LN scaling, XSA4, and VE128.
- The 2026-03-23 record showed that `LeakyReLU(0.5)^2` was a real gain on top of a very similar 11-layer stack.
- The 2026-03-21 README explicitly notes that its late-QAT flag never activated because `torch.compile` constant-folded `CastedLinear._qat_enabled` on the first trace.
- Recent low-bit training work such as BitNet b1.58 argues that quantization-aware training can preserve low-bit quality, so this repo still seems under-explored on the "make low-bit training actually happen" axis.

## Prior repository work that influenced this candidate

### Main bases

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`

### Specific takeaways reused here

- `LeakyReLU(0.5)^2` from the 2026-03-23 record.
- GPTQ-lite clip search + EMA + warmdown3500 from the 2026-03-22 record.
- The explicit postmortem that late QAT was dead code under `torch.compile` in the 2026-03-21 record.
- Bigger `BigramHash` tables were repeatedly helpful in 2026-03-20/23 experiments, so this candidate nudges the table upward rather than changing the architecture broadly.

### Prior candidates

There were no prior `candidates/` directories in this repository when this candidate was created.

## External research that informed it

- **BitNet b1.58** (`arXiv:2402.17764`): motivates making low-bit behavior part of training, not just post-training export. <https://arxiv.org/abs/2402.17764>
- **MobileLLM** (`arXiv:2402.14905`): reinforces that compact LMs are highly architecture-sensitive and benefit from disciplined, small changes on top of a strong deep-thin base. <https://arxiv.org/abs/2402.14905>
- **ALBERT** (`arXiv:1909.11942`) and **Universal Transformer** (`arXiv:1807.03819`) were considered because parameter sharing / recurrence are natural 16MB ideas, but the repository's own non-record recurrence attempts were clearly negative under a fixed 10-minute budget, so this candidate intentionally avoids that path for now.

## What changed vs. the chosen base implementation

Base file:
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

1. **MLP activation:**
   - from `ReLU^2`
   - to `LeakyReLU(0.5)^2`

2. **Late QAT fix:**
   - keep the existing late-QAT trigger (`scale < LATE_QAT_THRESHOLD`)
   - but when it fires, set `CastedLinear._qat_enabled = True` and immediately rebuild the compiled training graph with `torch._dynamo.reset()` + `torch.compile(...)`
   - this is intended to prevent the QAT branch from remaining dead after the first trace

3. **Bigger bigram table:**
   - default `BIGRAM_VOCAB_SIZE` from `2048` to `3072`

Everything else stays intentionally close to the 2026-03-22 stack so the effect is interpretable.

## How to run / evaluate

From this candidate directory:

```bash
RUN_ID=leakyrelu2_retrace_lateqat \
DATA_PATH=../../data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=3072 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
python -m torch.distributed.run --standalone --nproc_per_node=8 train_gpt.py
```

Notes:

- `QAT_ENABLED` can stay unset / `0`; this candidate is meant to turn QAT on automatically during the late phase and then retrace the compiled graph.
- EMA is still applied in-script with the same `0.997` decay used by the 2026-03-22 base.
- The script remains self-contained and runnable from the candidate directory.

## Main expected risks / tradeoffs

- **Retrace overhead:** rebuilding the compiled graph mid-run costs time and may eat some of the intended win.
- **Compile behavior risk:** this should fix the dead-branch issue, but `torch.compile` behavior can still vary across PyTorch versions.
- **Bigger bigram table may need retuning:** `3072` is evidence-backed, but not guaranteed optimal on this exact stack.
- **No new eval trick here:** unlike the 2026-03-23 record, this candidate does not add legal TTT, so it is a cleaner training-side experiment rather than a full record attempt.

## Validation

Commands run in this workflow:

```bash
python -m compileall candidates/202603261349_leakyrelu2-retrace-lateqat/train_gpt.py
```

Outcome:
- **Passed**.

Attempted additional smoke check:

```bash
python - <<'PY'
# import candidate module and run a tiny CPU forward/backward smoke
PY
```

Outcome:
- **Not feasible in this container** because both `/usr/bin/python` and `/usr/bin/python3` are missing `torch`, even though `requirements.txt` lists it. That prevented a runtime CPU smoke test here without changing the environment.
