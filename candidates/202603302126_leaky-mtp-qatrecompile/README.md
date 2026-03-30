# LeakyReLU^2 + training-only MTP + real late-QAT recompile

## Hypothesis

The strongest next non-TTT candidate in this repository is not a full architecture rewrite; it is a cleaner, better-trained version of the current 11-layer EMA + GPTQ-lite stack.

This candidate starts from the best pure-training lineage in `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` and adds three changes that are individually well-motivated but have not been stacked together here:

1. **LeakyReLU(0.5)^2** in the MLP, following the strongest activation ablation in the repo.
2. **A single training-only multi-token prediction head** (`t+2`) to improve sample efficiency during the 10-minute training window, then anneal it away and exclude it from export.
3. **A real late-QAT activation path** by recompiling the model when fake quantization turns on, avoiding the known `torch.compile` constant-folding issue from the earlier late-QAT record.

## Why this is promising for this repository

The records show a very clear pattern:

- the best non-TTT stack is already clustered around **11 layers + XSA + EMA + GPTQ-lite + warmdown tuning**;
- the largest remaining gains have come from **small, high-leverage tweaks** rather than wholesale rewrites;
- quantization quality still matters a lot, and one of the strongest QAT-lineage records explicitly documented that late QAT was accidentally dead due to compile-time folding;
- the best overall record got another meaningful gain from **LeakyReLU(0.5)^2**, but that result was entangled with legal TTT and the parameter-banking fork rather than cleanly tested in the strongest non-TTT branch.

This candidate therefore pushes on the highest-signal open lane: **make the best non-TTT branch train a little better, quantize a little better, and use the fixed wallclock budget more efficiently**.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - best pure-training base stack here: 11L, XSA4, EMA, GPTQ-lite int6, warmdown 3500.

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - strongest evidence in-repo that **LeakyReLU(0.5)^2** is worth real BPB.

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - important caution: the README documents that late QAT was effectively dead in that lineage because `torch.compile` folded the class flag.

- `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/`
  - evidence that a larger **BigramHash** can still help under the artifact cap, motivating a moderate increase from 2048 to 3072 buckets here.

## External research that informed it

- **Multi-Token Prediction** — Gloeckle et al., 2024, `arXiv:2404.19737`
  - argues that predicting multiple future tokens with auxiliary heads improves sample efficiency; this is especially attractive here because the extra head is **training-only** and can be excluded from the final artifact.

- **Scaling Law for Quantization-Aware Training** — Chen et al., 2025, `arXiv:2505.14302`
  - highlights that quantization error depends on model size, training tokens, and granularity, and that weight-side error matters increasingly as training progresses; this supports spending effort on a **real** late-QAT path instead of a nominal one.

- **SQuAT: Sharpness- and Quantization-Aware Training for BERT** — Wang et al., 2022, `arXiv:2210.07171`
  - reinforces the idea that flatter late-training solutions are friendlier to low-bit quantization, matching this repo's EMA/warmdown/QAT trends.

- **Transformers with Learnable Activation Functions** — Fang et al., 2022, `arXiv:2208.14111`
  - supports the broader repo lesson that activation choice is underexplored and can materially change transformer quality; this makes the leap from `relu^2` to `LeakyReLU^2` more than just a random tweak.

- **Small Language Models: Survey, Measurements, and Insights** — Lu et al., 2024, `arXiv:2409.15790`
  - useful framing for why this candidate favors a **small, implementation-light sample-efficiency improvement** over a riskier full architecture change.

I also considered more structural sharing ideas inspired by **Subformer** (`arXiv:2101.00234`) and **MASA** (`arXiv:2508.04581`), but I did not choose them for this candidate because the repository already contains negative evidence on simple recurrence under tight wallclock, and the implementation risk is much higher for a first next-candidate pass.

## What changed versus the chosen base implementation

Base implementation:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

1. **MLP activation**
   - `relu^2` -> `LeakyReLU(0.5)^2`

2. **Training-only multi-token prediction**
   - default `MTP_NUM_HEADS=1`
   - default `MTP_LOSS_WEIGHT=0.3`
   - default `MTP_ANNEAL_FRAC=0.15`
   - the auxiliary head is **not exported** in the final artifact

3. **Real late QAT**
   - keep the existing late-QAT gate, but when it activates, **recompile the training model**
   - this is meant to make the fake-quant branch actually participate in the compiled graph

4. **Bigger lexical side channel**
   - `BIGRAM_VOCAB_SIZE=3072` by default instead of 2048

5. **Practical runner improvements**
   - default dataset/tokenizer paths resolve relative to the repository root, so the script can be run directly from this candidate directory
   - FlashAttention import falls back to PyTorch SDP when unavailable
   - `SMOKE_TEST=1` runs a tiny CPU synthetic validation path that exercises forward/backward plus quantize/dequantize roundtrip without dataset access

## How to run or evaluate it

From this candidate directory:

```bash
cd candidates/202603302126_leaky-mtp-qatrecompile
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The default data and tokenizer paths point back to the repository root:

- `../../data/datasets/fineweb10B_sp1024`
- `../../data/tokenizers/fineweb_1024_bpe.model`

If your cached dataset lives elsewhere, override `DATA_PATH` and `TOKENIZER_PATH` as usual.

Useful knobs:

```bash
MTP_NUM_HEADS=1
MTP_LOSS_WEIGHT=0.3
MTP_ANNEAL_FRAC=0.15
BIGRAM_VOCAB_SIZE=3072
LATE_QAT_THRESHOLD=0.15
LATE_QAT_RECOMPILE=1
```

For a local dependency/runtime smoke check:

```bash
SMOKE_TEST=1 python train_gpt.py
```

## Validation run for this candidate

Local validation was intentionally lightweight and CPU-safe:

1. Syntax / bytecode compilation

```bash
python -m compileall candidates/202603302126_leaky-mtp-qatrecompile/train_gpt.py
```

Outcome: **passed**

2. CPU synthetic smoke test

```bash
SMOKE_TEST=1 python candidates/202603302126_leaky-mtp-qatrecompile/train_gpt.py
```

Outcome: **passed**

Observed output:

```text
smoke_test:ok loss:6.0740 reload_loss:4.8610
```

3. Existing low-cost repository syntax check

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603302126_leaky-mtp-qatrecompile/train_gpt.py
```

Outcome: **passed**

I did **not** run a real CUDA training job in this environment, so the candidate still needs a proper 8xH100 or smaller-GPU training trial to confirm the wallclock/quality tradeoff of the added MTP head.

## Main risks and tradeoffs

- **MTP may trade steps for sample efficiency.** If the extra head costs too much throughput, the gain from auxiliary supervision may be offset by fewer optimizer steps in 600 seconds.

- **The late-QAT recompile fix is plausible but still unbenchmarked here.** It addresses a known failure mode in the repo, but it still needs a real GPU run to confirm that the quantization gap shrinks in practice.

- **Bigger BigramHash is not guaranteed to be free.** It should still fit under the artifact budget, but only a real export confirms the final compressed size.

- **This is intentionally non-TTT.** It is aimed at improving the base trained model rather than competing with the repo's current legal-TTT-heavy best result directly.
