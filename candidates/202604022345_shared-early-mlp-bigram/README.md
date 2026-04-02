# Shared Early MLP + Bigger Bigram

## Hypothesis

The strongest non-TTT stack in this repo already squeezes a lot out of quantization, evaluation, and late-layer attention tweaks. A cleaner remaining axis is **parameter-efficient depth**: share only the earliest feed-forward blocks, keep per-layer attention and control tensors unique, and make the exporter alias-aware so the shared tensors are only paid for once in the artifact. The saved bytes can then fund a larger hashed bigram table while keeping the high-end 11-layer stack intact.

Concretely, this candidate shares the first **4 encoder-side MLPs**, keeps layer-specific attention / residual scales / skip paths, switches the MLP nonlinearity to **LeakyReLU(0.5)^2**, and raises the default `BIGRAM_VOCAB_SIZE` from **2048** to **3072**.

## Why this is promising here

The repo history shows a clear convergence:

- sliding-window evaluation, depth, 3x MLPs, fp16-friendly embeddings, SmearGate/BigramHash, EMA/SWA, XSA, Partial RoPE, LN scaling, and GPTQ-lite all helped;
- the 2026-03-22 and 2026-03-23 lines are the strongest practical bases;
- recurrence / sharing has **not** been meaningfully explored in the winning 11-layer line, even though the artifact budget is tight enough that weight reuse should matter.

This candidate is intentionally conservative about where sharing happens:

- **shared:** early MLP weights only;
- **not shared:** attention, RoPE/XSA behavior, residual/control tensors, skip weights, late decoder blocks.

That matches the intuition from compact-model research that early recurrent processing can be useful, while preserving the specialized late-layer stack that already works well in this repo.

## Prior repo experiments that influenced this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - strongest train/export stack without TTT;
  - provides the 11L + XSA4 + Partial RoPE + LN scale + EMA + GPTQ-lite backbone.
- **Activation + bigger bigram influence:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - LeakyReLU(0.5)^2 and a larger BigramHash were both positive there.
- **Architecture lineage:** `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - confirms Partial RoPE + LN scale are worth keeping.
- **Older backbone lineage:** `records/track_10min_16mb/2026-03-19_smeargate_orthoinit_muonwd/` and `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/`
  - establish SmearGate + BigramHash + 3x MLP as a durable base.
- **Prior candidates:** there was **no existing `candidates/` directory** when this candidate was created.

## External research that informed the choice

- **ALBERT** — Lan et al., 2019, arXiv:1909.11942  
  Cross-layer parameter sharing can cut memory and parameter count substantially without giving up strong language-model quality.
- **Universal Transformer** — Dehghani et al., 2018, arXiv:1807.03819  
  Recurrent depth is a viable way to add iterative computation while reusing weights.
- **Intra-Layer Recurrence in Transformers for Language Modeling** — Nguyen & Lin, 2025, arXiv:2505.01855  
  Reports that recurrence targeted at individual layers, especially **earlier layers**, is the most promising setting.
- Also reviewed but **not chosen as the main twist** because this repo already mined that space heavily:
  - **EfficientQAT** — arXiv:2407.11062
  - **Scaling Law for Quantization-Aware Training** — arXiv:2505.14302
  - **SiLQ** — arXiv:2507.16933

## What changed vs. the chosen base

Compared with `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate:

1. Shares the earliest encoder-side MLPs via `SHARED_MLP_LAYERS` (default **4**).
2. Makes the mixed int6 exporter **alias-aware**, so shared tensors are quantized and stored once, then reconstructed by alias on load.
3. Uses **LeakyReLU(0.5)^2** in the MLP instead of ReLU^2.
4. Increases the default hashed bigram table to **3072** buckets.
5. Resolves default dataset/tokenizer paths relative to the repo root so the script can be run directly from the candidate directory.
6. Adds a FlashAttention fallback to PyTorch SDPA for environments where `flash_attn_interface` is unavailable; the FA3 path remains unchanged when available.

Everything else stays intentionally close to the winning 2026-03-22 stack: 11 layers, XSA on the last 4 layers, Partial RoPE, LN scale, VE128 on layers 9-10, EMA, tight SWA, GPTQ-lite mixed int6 export, SmearGate, and BigramHash.

## How to run

From the repository root:

```bash
cd candidates/202604022345_shared-early-mlp-bigram
SEED=1337 \
SHARED_MLP_LAYERS=4 \
BIGRAM_VOCAB_SIZE=3072 \
LEAKY_RELU_SLOPE=0.5 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

If you are already inside `candidates/202604022345_shared-early-mlp-bigram/`, run the `torchrun ... train_gpt.py` command directly without the `cd`.

By default, `train_gpt.py` resolves:

- `DATA_PATH` to `../../data/datasets/fineweb10B_sp1024`
- `TOKENIZER_PATH` to `../../data/tokenizers/fineweb_1024_bpe.model`

so it can be launched directly from this folder. Override those environment variables if your data lives elsewhere.

## Evaluation notes

- The training/eval loop is inherited from the 2026-03-22 record stack.
- Final metrics are still reported through the script's `final_int6_roundtrip*` and `final_int6_sliding_window*` lines.
- `SHARED_MLP_LAYERS=0` disables the sharing experiment and should recover a near-base behavior with the other candidate defaults still applied.

## Main risks / tradeoffs

- Sharing early MLP weights may remove too much layer diversity and hurt optimization more than it helps the artifact budget.
- The exporter alias logic only helps if training quality survives the weight sharing; otherwise the saved bytes are not useful.
- A larger bigram table helps only if the extra lexical capacity is better than spending those bytes elsewhere.
- This candidate does **not** include the legal TTT / Parallel Muon stack from the current top record, so it is aiming for a strong **train/export** candidate rather than a drop-in SOTA replacement.

## Validation run here

Commands run in this repository:

```bash
python -m compileall candidates/202604022345_shared-early-mlp-bigram/train_gpt.py
python - <<'PY'
import importlib.util
print(importlib.util.find_spec("torch"))
PY
```

Outcomes:

- `compileall` **passed**.
- A deeper local import/forward/export smoke check was **attempted but blocked by the runner environment** because PyTorch is not installed here (`find_spec("torch") -> None`), so no CPU/GPU execution path was available in this workflow run.
