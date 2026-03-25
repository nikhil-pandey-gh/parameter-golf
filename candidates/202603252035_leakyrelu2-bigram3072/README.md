# Candidate: LeakyReLU(0.5)^2 + Bigger BigramHash on the best non-TTT 11L stack

## Hypothesis

Port the recently successful **LeakyReLU(0.5)^2** MLP activation onto the strongest existing **non-TTT** stack, and pair it with a slightly larger **BigramHash** table (`2048 -> 3072`).

The expected win is a better tradeoff between:

- gradient flow in the 3x MLP,
- cheap local-context modeling from hashed bigrams,
- and the already-strong compression-aware recipe from the 11-layer EMA + GPTQ-lite record.

## Why this is promising for this repository

The repository review showed a clear pattern:

- the frontier already converged on **11 layers + 3x MLP + XSA + partial RoPE + LN scaling + EMA + mixed int6/int8 export**, and
- the safest remaining gaps were **LeakyReLU^2 on the best non-TTT stack** and **a larger/tuned BigramHash**, not large architectural rewrites.

In particular:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` is the best standard-training base I found.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` reports a meaningful gain from `LeakyReLU(0.5)^2` on a closely related frontier stack.
- multiple records show **BigramHash** is one of the cheapest persistent wins for this challenge.

I also explicitly reviewed depth-reuse / recurrent-layer ideas via the repo history and external papers, but the repo's own negative results suggest that under the strict **10-minute** budget, recurrence tends to lose too many optimization steps to be competitive here.

## Prior records and candidates that influenced this candidate

### Direct implementation base

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`

This candidate copies that script as the base because it is the strongest non-TTT implementation and already includes the modern house stack:

- 11 layers, 512d, 8H / 4KV
- 3x MLP
- XSA on the last 4 layers
- partial RoPE (`16/64`)
- LN scale
- VE128 on late layers
- EMA + tight SWA
- GPTQ-lite style clip search for int6 export
- sliding-window evaluation

### Additional repo evidence used for the new twist

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - source of the **LeakyReLU(0.5)^2** porting idea
- `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/`
  - evidence that a larger hashed bigram table can be worthwhile
- `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/`
  - evidence that BigramHash remains a robust low-cost inductive bias in this repo
- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/`
  - useful negative result: recurrence/depth looping was tried and called out as net-negative under the time cap

### Prior candidates

- No `candidates/` directory existed in this checkout before this addition.

## External research that informed the choice

I reviewed the following primary sources while selecting the idea:

- **Primer: Searching for Efficient Transformers for Language Modeling** (`arXiv:2109.08668`)
  - Important here because Primer identified **squared ReLU** as one of the simple modifications that can improve Transformer efficiency.
  - This supports keeping the repo's strong `relu^2`/`square-after-activation` bias while trying a better pre-square nonlinearity.
- **GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers** (`arXiv:2210.17323`)
  - Reinforces the repository trend that quantization quality is central, not a minor afterthought.
  - This candidate therefore keeps the strong GPTQ-lite export path from the 2026-03-22 base instead of replacing the compression stack.
- **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration** (`arXiv:2306.00978`)
  - Reinforces the idea that some channels/tensors matter disproportionately for low-bit export.
  - This supports making a conservative change to the model body while preserving the export strategy that already works well here.
- **ALBERT** (`arXiv:1909.11942`) and **Universal Transformer** (`arXiv:1807.03819`)
  - These made depth reuse / parameter sharing worth considering.
  - However, the repository's own experiments reported recurrence/depth reuse as a poor fit for the 10-minute budget, so I used them mainly as researched-but-rejected alternatives.

## What changed versus the chosen base implementation

Relative to `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`:

1. **MLP activation: `ReLU^2 -> LeakyReLU(0.5)^2`**
   - Old:
     - `x = torch.relu(self.fc(x))`
     - `return self.proj(x.square())`
   - New:
     - `x = F.leaky_relu(self.fc(x), negative_slope=0.5)`
     - `return self.proj(x.square())`

2. **Bigger default BigramHash table**
   - `BIGRAM_VOCAB_SIZE` default changed from `2048` to `3072`.
   - This is intentionally modest: large enough to test the repo's positive bigram trend, but not such a large jump that artifact risk becomes the whole story.

3. **Import/manual-forward attention fallback**
   - If `flash_attn_interface` is unavailable, the script now falls back to PyTorch `scaled_dot_product_attention`.
   - This only removes the hard dependency on FlashAttention for import-level or manual forward checks. The main training/eval entrypoint still requires CUDA, and on the intended CUDA setup with FlashAttention available, the fast path is still used.

4. **New env knob for the activation port**
   - `LEAKY_RELU_SLOPE` (default `0.5`) so the activation can be ablated without editing code.

## How to run or evaluate it

From the repository root:

```bash
cd candidates/202603252035_leakyrelu2-bigram3072
RUN_ID=leakyrelu2_bigram3072 \
DATA_PATH=../../data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful ablations:

```bash
# Turn the candidate back into the original activation while keeping the larger bigram table
RUN_ID=ablate_relu2 \
DATA_PATH=../../data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
LEAKY_RELU_SLOPE=0.0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py

# Compare against the old bigram size on the same code
RUN_ID=ablate_bigram2048 \
DATA_PATH=../../data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
BIGRAM_VOCAB_SIZE=2048 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Validation run in this workflow

### Commands run

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603252035_leakyrelu2-bigram3072/train_gpt.py
```

Attempted CPU smoke environment check:

```bash
python3 - <<'PY'
import torch
PY
python - <<'PY'
import torch
PY
```

### Outcomes

- `compileall` completed successfully for the existing root Python entrypoints, `data/`, and the new candidate `train_gpt.py`.
- A true CPU runtime smoke test was **not feasible in this runner** because neither `python` nor `python3` had `torch` installed.
- The candidate still includes an attention fallback so that once PyTorch is available, local import/manual-forward checks do not also depend on FlashAttention being present.

## Main expected risks or tradeoffs

- The `LeakyReLU(0.5)^2` gain was observed on a close but not identical frontier stack; the transfer to the 2026-03-22 base may be smaller than hoped.
- Increasing `BIGRAM_VOCAB_SIZE` can help local context modeling, but it also spends more bytes on embeddings that may or may not compress favorably enough to pay back the cost.
- This candidate intentionally avoids a deeper architectural rewrite. That makes it safer to evaluate, but it also means the upside may be incremental rather than dramatic.
- The attention fallback is only for portability and manual validation; it is not expected to match FlashAttention throughput on the target GPU environment.
