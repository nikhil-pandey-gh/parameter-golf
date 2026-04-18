# LWC-lite + LeakyReLU^2 late-QAT

## Hypothesis

The best pre-TTT stack in this repo already does late int6 QAT and GPTQ-lite export, but its fake-quant clip range is still fixed by per-row maxima. A tiny, learnable per-output-channel clip multiplier should let late-QAT reshape weights toward a lower-error int6 export, especially when paired with GPTQ-lite's percentile search at save time.

I also ported the record-proven `LeakyReLU(0.5)^2` MLP change because it is a near-free activation upgrade on the same 11-layer family.

## Why this is promising here

This repository's strongest consistent gains after sliding-window eval came from:

1. better quantization-aware training and export,
2. low-byte architectural improvements that preserve the 11L / 2048-token / MLPx3 recipe,
3. tiny changes that improve the final compressed round-trip rather than the raw fp32 model.

That makes quantizer refinement a better next bet than a brand-new backbone. The current best non-TTT record (`2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`) is already a clean export-aware base, and the overall record (`2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`) shows that `LeakyReLU^2` is a meaningful low-risk activation improvement.

## Influential records and prior experiments

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - best clean pre-TTT base,
  - introduces EMA + GPTQ-lite + later QAT trigger.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - proves `LeakyReLU(0.5)^2` helps on the strongest 11-layer stack,
  - but also adds parameter banking and legal TTT, which I intentionally did **not** copy here.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - confirms partial RoPE + LN-scale matter,
  - and shows the repo has already converged on the same architectural backbone used here.

No prior `candidates/` existed when this candidate was created.

## External research that informed it

- **LSQ** — *Learned Step Size Quantization* (Esser et al., 2019): learn quantizer parameters jointly with the model instead of fixing them by heuristic.
  - <https://arxiv.org/abs/1902.08153>
- **SmoothQuant** — *Accurate and Efficient Post-Training Quantization for Large Language Models* (Xiao et al., 2022/2023): shifting quantization difficulty with mathematically structured rescaling is powerful for LLMs.
  - <https://arxiv.org/abs/2211.10438>
- **AWQ** — *Activation-aware Weight Quantization for LLM Compression and Acceleration* (Lin et al., 2023): protecting the right channels matters more than uniform clipping.
  - <https://arxiv.org/abs/2306.00978>
- **OmniQuant** — *Omnidirectionally Calibrated Quantization for Large Language Models* (Shao et al., 2023/2024): learnable weight clipping (LWC) is an effective low-bit refinement on top of existing PTQ/QAT pipelines.
  - <https://arxiv.org/abs/2308.13137>

This candidate implements the smallest repo-compatible slice of those ideas: **LWC-lite** for late-QAT and export, not full calibration-time LET/rotation infrastructure.

## What changed versus the chosen base

Starting point: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes:

1. **LeakyReLU^2 MLP**
   - `relu(x)^2` -> `leaky_relu(x, 0.5)^2`
   - copied from the current best record's activation tweak.
2. **Learnable late-QAT clip parameters**
   - every `CastedLinear` now owns a small `qat_log_clip` vector (one value per output row),
   - when late-QAT is enabled, the fake int6 quantizer uses these learned clip multipliers instead of fixed row-max clipping alone.
3. **Export uses the learned clip candidate too**
   - GPTQ-lite percentile search is kept,
   - but export now also evaluates the learned clip candidate and keeps whichever reconstruction is best.
4. **Candidate script works from inside the candidate directory**
   - default dataset and tokenizer paths resolve relative to the script location instead of assuming repo-root CWD.

## How to run

From this candidate directory:

```bash
RUN_ID=lwc_leaky_qat \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Optional knobs worth ablation:

```bash
LATE_QAT_THRESHOLD=0.10   # later fake quant start
LATE_QAT_THRESHOLD=0.20   # earlier fake quant start
VAL_LOSS_EVERY=2000       # more frequent checkpoints while debugging
```

The defaults already keep the 11L / seq2048 / EMA / XSA4 / partial-RoPE / GPTQ-lite recipe from the base run.

## How to evaluate

The script still:

- writes `final_model.int6.ptz`,
- validates the quantized round-trip,
- reports sliding-window metrics with `stride=64`,
- logs the final `final_int8_zlib_roundtrip_exact` compatibility line used elsewhere in the repo.

## Main risks and tradeoffs

- The learned clip vectors add a small number of training-only parameters; they are cheap, but they do slightly increase the exported state because they are kept as passthrough tensors for strict round-trip loading.
- Late-QAT may simply prefer the old GPTQ-lite percentile choices, in which case this candidate will behave like a low-cost ablation rather than a clear upgrade.
- I deliberately did **not** combine this with legal TTT or parameter banking, so a strong result here would still need follow-up work to port the idea onto the full `2026-03-23` stack.

## Validation

Executed lightweight validation:

```bash
python -m compileall train_gpt.py
```

Outcome: **passed**.

Attempted CPU-side import / construction smoke check:

```bash
python - <<'PY'
import train_gpt
h = train_gpt.Hyperparameters()
print('data_path=', h.data_path)
print('tokenizer_path=', h.tokenizer_path)
m = train_gpt.GPT(
    vocab_size=32,
    num_layers=2,
    model_dim=8,
    num_heads=2,
    num_kv_heads=1,
    mlp_mult=2,
    tie_embeddings=True,
    tied_embed_init_std=0.01,
    logit_softcap=30.0,
    rope_base=10000.0,
    qk_gain_init=1.0,
    mtp_num_heads=0,
    mtp_loss_weight=0.0,
    bigram_vocab_size=0,
    bigram_dim=8,
    xsa_last_n=0,
    rope_dims=0,
    ln_scale=False,
    dtg=False,
    ve_enabled=False,
    ve_dim=8,
    ve_layers='',
)
print('tiny_params=', sum(p.numel() for p in m.parameters()))
PY
```

Outcome: **not feasible in this workflow environment**. The import failed immediately with `ModuleNotFoundError: No module named 'numpy'`, so the environment does not currently have the repository's Python dependencies installed. A fuller smoke test would also still require CUDA plus FlashAttention 3 for any real forward/training path.
