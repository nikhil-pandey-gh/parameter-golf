# MLA + RoPE + Bigram3072 candidate

## Hypothesis

Replace the strongest pre-TTT record's standard GQA K/V projections with **MLA-style latent K/V compression**, keep the repo's proven **partial RoPE** and deep-layer tricks, and spend part of the recovered parameter budget on a larger **BigramHash** table.

The specific bet is that, in this repository's 11-layer / 512d regime, **RoPE-preserving MLA will keep most of the quality of full K/V projections while freeing enough parameters to improve front-end token-pair priors without blowing the 16MB artifact budget**.

## Why this looks promising here

Repository review suggests three strong patterns:

1. The best non-TTT stack is already quite mature: `11L + XSA4 + EMA + Partial RoPE + LN scale + SmearGate + BigramHash + VE + GPTQ-lite`.
2. The best improvements lately are mostly **small, composable wins** layered onto that stack, not wholesale rewrites.
3. Naive recurrent depth already failed badly under the fixed 10-minute wallclock budget, so the next good efficiency idea should save parameters or memory **without adding extra recurrent passes**.

MLA fits that shape better than recurrence. It changes only the attention parameterization, preserves the single-pass training path, and plays naturally with the repo's existing GQA, RoPE, XSA, EMA, and mixed-precision export code.

## Prior repository influence

### Chosen base implementation

This candidate starts from the strongest pre-TTT training stack:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

That record already includes the strongest reusable non-TTT ingredients in this repo:

- 11 layers at 512d
- MLP 3x
- SmearGate + BigramHash
- XSA on the last 4 layers
- Partial RoPE (`16/64`) + layerwise LN scaling
- VE128 on late layers
- EMA + tight SWA
- GPTQ-lite style per-row clip search for int6 export
- sliding-window evaluation

### Other records that influenced the choice

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`
  - motivated borrowing **LeakyReLU(0.5)^2** and a larger **BigramHash** budget (`2048 -> 3072` was positive in their ablation).
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md`
  - reinforced that **Partial RoPE** and **LN scale** are real wins in this depth regime.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md`
  - confirmed that **layer recurrence** is a poor fit for this wallclock-constrained challenge.

### Prior candidates

There was **no existing `candidates/` directory** at review time, so this is the first candidate folder in the repository.

## External research informing this candidate

- **Mehta et al., "Latent Multi-Head Attention for Small Language Models" (arXiv:2506.09342)**
  - reports that **MLA + RoPE** is the right small-model variant, while MLA without RoPE underperforms.
  - their key result is that **half-rank latent dimensions** can preserve nearly all quality while materially reducing KV-state cost.
  - this candidate mirrors that recommendation by keeping RoPE and setting `MLA_LATENT_DIM=128` for a 512d model.

- **PLM: Efficient Peripheral Language Models Hardware-Co-Designed for Ubiquitous Computing (arXiv:2503.12167)**
  - another small-model design that combines **MLA** with **squared-ReLU-family activations** and efficiency-focused architecture choices.
  - this made the MLA + `LeakyReLU^2` combination feel like a reasonable repo-native experiment rather than two unrelated ideas.

## What changed versus the chosen base

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate makes five focused changes:

1. **MLA-style latent K/V compression**
   - adds `MLA_LATENT_DIM` (default `128`)
   - replaces direct `c_k` / `c_v` projections with:
     - `c_kv_latent: dim -> latent`
     - `k_from_latent: latent -> kv_dim`
     - `v_from_latent: latent -> kv_dim`
   - keeps the query path, GQA shape, RoPE application, FlashAttention call, and XSA path unchanged.

2. **Larger BigramHash table**
   - increases `BIGRAM_VOCAB_SIZE` default from `2048` to `3072`
   - the idea is to reinvest some of the MLA parameter savings into a stronger front-end token-pair prior, matching a trend that already helped in the current best record.

3. **LeakyReLU(0.5)^2 MLP**
   - changes the MLP activation from `relu(x)^2` to `leaky_relu(x, 0.5)^2`
   - exposes `MLP_NEGATIVE_SLOPE` (default `0.5`) for easy ablation.

4. **Run-from-candidate-directory path fix**
   - default `DATA_PATH` and `TOKENIZER_PATH` are now resolved from the repository root using `__file__`, so `train_gpt.py` works correctly when launched from this candidate directory.

5. **Late-QAT disabled by default**
   - sets `LATE_QAT_THRESHOLD=0.0` by default in this candidate
   - this keeps the experiment focused on MLA rather than a runtime quantization toggle whose behavior has been fragile under `torch.compile` in prior repo history
   - if desired, it can still be re-enabled explicitly as an ablation

I intentionally did **not** add new infrastructure, teacher models, or extra files. The quantization/export path, EMA/SWA flow, sliding-window eval, and optimizer structure are otherwise preserved from the base record.

## How to run

From this directory:

```bash
cd candidates/202603310102_mla-rope-bigram3072
RUN_ID=mla_rope_bigram3072 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful ablations:

```bash
# Recover the base attention parameterization
MLA_LATENT_DIM=0 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Smaller or larger latent bottleneck
MLA_LATENT_DIM=96 torchrun --standalone --nproc_per_node=8 train_gpt.py
MLA_LATENT_DIM=160 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Revert the activation if desired
MLP_NEGATIVE_SLOPE=0.0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Because the default data/tokenizer paths are resolved relative to the repository root, you only need to override `DATA_PATH` or `TOKENIZER_PATH` if your local layout differs from the repository default.

## Main risks / tradeoffs

- **Quality risk**: MLA may undercompress K/V too aggressively at `128` latent dims once combined with XSA, VE, and aggressive post-training quantization.
- **Speed risk**: parameter savings do not automatically guarantee better wallclock throughput; extra projections could erase some of the expected benefit.
- **Export interaction risk**: GPTQ-lite clip search and mixed int6 export were tuned on the original attention weights, so the new attention parameterization may quantize differently.
- **Attribution risk**: this candidate intentionally combines one new research-backed idea (MLA) with two repo-supported low-risk upgrades (Bigram3072 and LeakyReLU^2), so a future 8xH100 run should ablate those separately.

## Validation

### Commands run

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603310102_mla-rope-bigram3072/train_gpt.py
```

### Outcome

- `compileall` **passed** for the repository Python entrypoints plus this candidate script.

### Attempted smoke check

I also attempted a lightweight import-and-construct smoke test for the candidate module. That did **not** run in this workflow environment because the runner is missing repository Python dependencies before the model can even import:

- `ModuleNotFoundError: No module named 'numpy'`

`numpy` is listed in the repository `requirements.txt`, so this is an environment limitation rather than a candidate-script syntax failure.

### Why a CPU-only run was not feasible here

A faithful runtime smoke test is not available in this environment without changing the machine setup, because this script follows the repository's normal GPU path and expects:

- repository Python dependencies from `requirements.txt`
- CUDA availability
- FlashAttention runtime support (`flash_attn_interface`)

Given those constraints, `compileall` was the highest-signal validation available without introducing new infrastructure or mutating the shared environment.
