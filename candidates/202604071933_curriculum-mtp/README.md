# 202604071933_curriculum-mtp

## Hypothesis

The strongest recent record stack in this repository appears close to saturated on **architecture**, **quantization**, and **evaluation-time tricks**, but it still trains with a mostly standard next-token objective. A **small-model-friendly multi-token prediction (MTP) curriculum** should improve sample efficiency inside the fixed 10-minute training budget while leaving the exported artifact size effectively unchanged because the auxiliary MTP heads are stripped before serialization.

## Why this is promising here

Repository evidence points in the same direction:

- The best runs steadily improved by stacking **compression-funded capacity**, **EMA/SWA**, **GPTQ-lite**, **partial RoPE**, **XSA**, and finally **legal TTT**, reaching `1.1194` post-TTT bpb in `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`.
- Those records repeatedly show that **evaluation context** and **quantization-aware choices** matter a lot, while clearly slow ideas like weight recurrence are a bad fit for the 10-minute wall-clock budget.
- Several strong standalone record scripts already carried a dormant `MTP_NUM_HEADS` path, but none of the documented record READMEs actually turned it on for a scored run. That makes MTP one of the clearest underexplored knobs left in the current code lineage.

External research also supports trying this now:

- **Better & Faster Large Language Models via Multi-token Prediction** ([arXiv:2404.19737](https://arxiv.org/abs/2404.19737)) reports that predicting multiple future tokens from a shared trunk improves sample efficiency with little added training-time overhead.
- **Pre-Training Curriculum for Multi-Token Prediction in Language Models** ([arXiv:2505.22757](https://arxiv.org/abs/2505.22757)) specifically finds that **small language models struggle with naive MTP**, and that a **forward curriculum** helps them recover the benefit.
- **Predicting the Order of Upcoming Tokens Improves Language Modeling** ([arXiv:2508.19228](https://arxiv.org/abs/2508.19228)) is a useful cautionary signal: exact far-future token prediction can be too hard as an auxiliary loss. That argues for keeping horizons short and weighting near-future heads more heavily here.

## Main repository influences

This candidate is primarily based on:

1. `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
2. `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
3. Earlier 11-layer standalone records that already carried the dormant MTP hooks:
   - `records/track_10min_16mb/2026-03-20_11L_EfficientPartialXSA_FA3_SWA120/`
   - `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`
   - `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`

There were no prior runs under `candidates/` to reuse.

## What changed versus the chosen base implementation

Starting from the `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` script, this candidate makes one focused change:

- **Turns on training-only MTP by default**
  - `MTP_NUM_HEADS=2`
  - `MTP_LOSS_WEIGHT=0.15`
- **Adds a forward curriculum**
  - `MTP_CURRICULUM_FRAC=0.4`
  - MTP ramps from pure next-token training toward full auxiliary loss over the first 40% of the run, keyed off wall-clock progress when the 600s cap is active.
- **Biases the auxiliary task toward the nearest future**
  - `MTP_HEAD_WEIGHT_DECAY=0.5`
  - Head 1 gets the strongest weight, later horizons are downweighted, and deeper horizons ramp in more slowly via `curriculum_scale ** (k + 1)`.
- **Keeps the artifact budget behavior unchanged**
  - MTP heads are still excluded from `final_model.pt` / quantized export, just as the latent path already allowed in the record code.

Everything else intentionally stays aligned with the strongest current stack:

- 11 layers, 512d, 8H / 4KV
- LeakyReLU(0.5)^2 MLP
- Parameter banking + parallel Muon
- Partial RoPE, LN scale, XSA on late layers
- VE128 late-layer value embeddings
- GPTQ-lite int6 + lzma export
- Legal score-first TTT evaluation

## How to run

From the repository root:

```bash
cd candidates/202604071933_curriculum-mtp

DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The new MTP settings are already the defaults. To ablate them:

```bash
MTP_NUM_HEADS=0 MTP_LOSS_WEIGHT=0.0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Validation

Validated with:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604071933_curriculum-mtp/train_gpt.py
```

Outcome:

- `train_gpt.py`, `train_gpt_mlx.py`, `data/`, and the candidate `train_gpt.py` all compiled successfully after the candidate changes.
- A true CPU-only runtime smoke test was **not feasible** here because this standalone candidate hard depends on the CUDA/FlashAttention training stack used by the record it forks, plus dataset shards/tokenizer files at runtime.

## Main risks and tradeoffs

- **Step-time regression:** even training-only heads can cost enough compute to erase the sample-efficiency gain.
- **Objective mismatch:** MTP may improve pre-TTT modeling while not helping the final score as much once legal TTT is applied.
- **Schedule sensitivity:** the curriculum fraction and head decay are plausible defaults, not tuned optima.
- **`torch.compile` interaction:** this candidate uses a tensor buffer for curriculum state specifically to avoid the kind of constant-folding problem that broke the old late-QAT toggle, but it still deserves a real GPU run.

## Suggested next experiments if this helps

1. Sweep `MTP_CURRICULUM_FRAC` in `{0.25, 0.4, 0.55}`.
2. Sweep `MTP_NUM_HEADS` between `1` and `2` before trying anything larger.
3. If the pre-TTT score improves but post-TTT gain is flat, try decaying the MTP loss during the final warmdown window.
