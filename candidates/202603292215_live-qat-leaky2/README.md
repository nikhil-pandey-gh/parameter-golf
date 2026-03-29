# Candidate: live late QAT + LeakyReLU^2 on the 11L GPTQ-lite base

## Hypothesis

The current 11-layer training-only record is already strong, but it still leaves quantization as a visible bottleneck and uses the older ReLU^2 MLP. This candidate tests whether a cleaner transfer of two later wins can move that base forward:

1. port `LeakyReLU(0.5)^2`, which improved a closely related 11-layer stack in the later legal-TTT record, and
2. make late QAT actually reachable under `torch.compile` by preparing both a no-QAT graph and a QAT-specialized graph up front, then switching once the learning-rate scale enters the late-training regime.

The expected outcome is a smaller post-quantization gap without paying the full-run cost of always-on fake quantization, plus a modest training-time quality gain from the leaky-squared MLP.

## Why this is promising for this repository

Repository history shows a consistent pattern:

- the best training-only lineage moved from `11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271` to `11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` to `11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`,
- the `2026-03-21` record explicitly documented that its late-QAT branch was dead-code-eliminated under `torch.compile`, and
- the later `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` record showed that `LeakyReLU(0.5)^2` was worth another `-0.0021` on a very similar 11-layer stack.

So the repo evidence points toward a practical next step: keep the cleaner `2026-03-22` base, import the proven leaky-squared MLP change, and replace "logged as enabled" late QAT with an explicit graph switch that should survive compilation.

There was no existing `candidates/` directory at review time, so this candidate is branching only from the baseline and `records/`.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - chosen as the base because it is the strongest simpler training-only stack in the repo.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - contributed the `LeakyReLU(0.5)^2` MLP idea and the evidence that a larger `BigramHash` table can still help on a related 11-layer model.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - motivated the compile-safe late-QAT handling because its README notes that the original late-QAT branch never actually activated under `torch.compile`.
- `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/`
  - reinforced that larger bigram tables remained beneficial late into the record progression.

## External research that informed it

- **Primer: Searching for Efficient Transformers for Language Modeling** (`arXiv:2109.08668`)
  - Primer found that squared ReLU is one of the key simple changes that improves language-model efficiency. This candidate keeps the squared-activation bias but uses the repo-proven leaky variant to preserve negative-side gradient flow.
- **BitNet b1.58** (`arXiv:2402.17764`)
  - BitNet strengthens the case that low-bit-aware training can remain competitive with full precision. That makes "real" late-stage fake quantization a good fit for this repository's artifact-constrained objective.
- **Exclusive Self Attention** (`arXiv:2603.09078`)
  - XSA is already one of the strongest cheap wins in the repo, and the paper reports larger gains as sequence length grows. That supported staying on the 11-layer XSA base rather than pivoting to a broader attention redesign.

I also reviewed newer alternatives such as Scalable-Softmax (`arXiv:2501.19399`) and Diff Transformer (`arXiv:2410.05258`), but both would require deeper attention-kernel changes than this candidate's "adapt the current code" budget allows.

## What changed versus the chosen base

Starting point: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

1. **MLP activation**
   - changed from `ReLU^2` to `LeakyReLU(0.5)^2`.

2. **Late-QAT activation path**
   - instead of relying on a single compiled graph to notice a class-level boolean flip mid-run, the candidate prepares:
     - a default compiled graph with QAT off, and
     - a second compiled graph specialized for the late-QAT branch.
   - the late-QAT graph is warmed once before the timed loop so the actual switch does not spend its window compiling.
   - training then switches to the second graph once `scale < LATE_QAT_THRESHOLD`.

3. **BigramHash default**
   - bumped `BIGRAM_VOCAB_SIZE` from `2048` to `3072` as a modest transfer from the later top-line ablations.

Everything else stays intentionally close to the `2026-03-22` base: 11 layers, XSA on the last 4 layers, Partial RoPE, LN scale, EMA + tight SWA, shared late-layer value embeddings, GPTQ-lite int6 export, and stride-64 sliding-window evaluation.

## How to run / evaluate

From the repository root:

```bash
cd candidates/202603292215_live-qat-leaky2
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful knobs:

```bash
BIGRAM_VOCAB_SIZE=3072 \
LATE_QAT_THRESHOLD=0.15 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script resolves the usual dataset and tokenizer defaults relative to the repository root, so running from inside the candidate directory still uses:

- `DATA_PATH=./data/datasets/fineweb10B_sp1024`
- `TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model`

## Main risks and tradeoffs

- **Compile/runtime risk:** the dual-graph late-QAT switch is deliberately explicit, but it still depends on `torch.compile` behaving stably when two optimized graphs share one underlying model.
- **Bigram size risk:** `3072` is a modest bump, but it still spends some budget on hash features that could fail to transfer cleanly from the later parameter-banked stack.
- **No broader architecture change:** this candidate intentionally avoids riskier ideas like SSMax, Diff Transformer, or parameter sharing, so its upside is more likely incremental than dramatic.
- **Environment coupling:** the script still expects the normal GPU training environment used by the repo's record scripts.

## Validation

Commands run for this candidate in the current environment:

```bash
python -m compileall candidates/202603292215_live-qat-leaky2/train_gpt.py
python -c "import torch"
```

Outcomes:

- `python -m compileall ...` — passed.
- `python -c "import torch"` — failed in this workflow environment because `torch` is not installed here, so a runtime smoke test was not feasible without adding new heavyweight infrastructure. I therefore limited validation to syntax compilation and code review in this environment.
