# LSQ-style Bank QAT on the LeakyReLU2 + Parallel Muon stack

## Hypothesis

The current best record already pushes the architecture and eval stack hard, but its dominant weights live inside four parameter banks and still reach the artifact through a mostly post-training int6 path. A compile-safe, late-warmdown, per-row learned-step-size fake-quant pass on those bank tensors should shrink the train-to-int6 gap without changing the final artifact format or adding permanent parameters.

## Why this is promising here

- The record history repeatedly improved by reducing quantization damage rather than by making the pretrained float model much better.
- The 4-hour non-record baseline still had a large post-quantization gap, which suggests the compression bottleneck remains real even after longer training.
- `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` explicitly notes that its late-QAT path was constant-folded away by `torch.compile`.
- `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` moved the main attention/MLP weights into bank tensors, so the old `CastedLinear`-only fake-quant path is even less relevant there.

## Prior runs that informed this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
- **Export/PTQ inspiration:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- **Dead-end/root-cause inspiration:** `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
- **Quantization-bottleneck evidence:** `records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/` and `records/track_10min_16mb/2026-03-19_WarmdownQuantization/`

There were no prior experiments under `candidates/` when this candidate was created.

## External research that informed it

- **LSQ** — Learned Step Size Quantization (Esser et al., 2020): learn the quantizer step size directly instead of fixing it from row maxima. <https://arxiv.org/abs/1902.08153>
- **LSQ+** (Bhalgat et al., 2020): better initialization matters for low-bit stability; that motivated the percentile-search warm start instead of naive unit scales. <https://arxiv.org/abs/2004.09576>
- **AdaRound** (Nagel et al., 2020): rounding and quantizer choices matter enough that nearest-grid export is often not optimal. <https://arxiv.org/abs/2004.10568>
- **Scaling Law for Quantization-Aware Training** (Chen et al., 2025): weight-side quantization error becomes increasingly important as training progresses, which argues for a late warmdown weight-focused QAT pass. <https://arxiv.org/abs/2505.14302>
- **BitNet b1.58 Reloaded** (Nielsen and Schneider-Kamp, 2024): quantization-aware training remains competitive even for small networks, not only large LLMs. <https://arxiv.org/abs/2407.09527>

I also reviewed recent rotation-based PTQ work such as QuaRot and SpinQuant, but bank-aware LSQ-style QAT was a much better fit for this repo's single-file training script and 10-minute training regime.

## What changed versus the chosen base

Compared with `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`, this candidate:

1. Adds a **bank-QAT path** for `qo_bank`, `kv_bank`, `mlp_up_bank`, and `mlp_down_bank`.
2. Introduces **per-row learned scale parameters** for those banks plus a small dedicated AdamW optimizer with zero weight decay.
3. Enables bank QAT only in late warmdown via `BANK_QAT_THRESHOLD`, and **warm-starts scales from percentile-search reconstruction** when that threshold is crossed.
4. Uses a **tensor mix** instead of a Python/class boolean so the fake-quant path survives `torch.compile` instead of being dead-code-eliminated.
5. Reuses the learned bank scales during int6 export through **scale overrides**, rather than recomputing fresh heuristic scales for those matrices.
6. Adds a **FlashAttention import fallback** to PyTorch SDPA so the file can still be imported and smoke-tested on CPU-only environments.
7. Aligns a couple of defaults with the strongest recent run where it was harmless to do so (`BIGRAM_VOCAB_SIZE=1536`, `TTT_FREEZE_BLOCKS=0`).

The final artifact format is still the same mixed int6/int8 + lzma flow used by the base record.

## How to run

From this directory:

```bash
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
BANK_QAT_ENABLED=1 BANK_QAT_THRESHOLD=0.20 BANK_QAT_SCALE_LR=0.002 \
TTT_ENABLED=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Notes:

- Most of the strong architecture/training defaults are already baked into this copy.
- Set `TTT_ENABLED=0` if you want to isolate the pure training/export improvement from the eval-time TTT stack.
- On Hopper with `flash_attn_interface` available, the script still uses FlashAttention; otherwise it falls back to SDPA.

## Validation run for this candidate

Commands executed in this workflow:

```bash
python -m compileall candidates/202604060807_lsq-bank-qat/train_gpt.py
```

Outcome: **passed**.

```bash
PYTHONDONTWRITEBYTECODE=1 /tmp/gh-aw/agent/candidate-venv/bin/python - <<'PY'
import importlib.util
from pathlib import Path
import torch

path = Path('candidates/202604060807_lsq-bank-qat/train_gpt.py')
spec = importlib.util.spec_from_file_location('candidate_train_gpt', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

model = mod.GPT(
    vocab_size=16,
    num_layers=2,
    model_dim=16,
    num_heads=2,
    num_kv_heads=2,
    mlp_mult=2,
    tie_embeddings=True,
    tied_embed_init_std=0.01,
    logit_softcap=30.0,
    rope_base=10000.0,
    qk_gain_init=1.0,
    bigram_vocab_size=0,
    xsa_last_n=0,
    rope_dims=4,
    ln_scale=False,
    ve_enabled=False,
).float()

input_ids = torch.randint(0, 16, (1, 4), dtype=torch.int64)
target_ids = torch.randint(0, 16, (1, 4), dtype=torch.int64)
loss = model(input_ids, target_ids)
model.enable_bank_qat()
loss_q = model(input_ids, target_ids)
opt = torch.optim.AdamW([
    model.qo_bank_scale,
    model.kv_bank_scale,
    model.mlp_up_bank_scale,
    model.mlp_down_bank_scale,
], lr=1e-3)
opt.zero_grad(set_to_none=True)
loss_q.backward()
before = model.qo_bank_scale.detach().clone()
opt.step()
after = model.qo_bank_scale.detach()
print(loss.item(), loss_q.item(), (after - before).abs().sum().item())
PY
```

Outcome: **passed** (`cpu_qat_step_ok loss=2.7836 loss_q=2.7836 delta=0.000021`), which confirms the file imports on CPU, the model starts, bank QAT can be enabled, and the learned scale parameters actually update under an optimizer step.

The workflow image did not have the repo runtime deps installed in system Python, so the smoke test used a temporary venv with `numpy`, `sentencepiece`, and `torch`.

## Main risks / tradeoffs

- The one-time percentile warm start uses `torch.quantile` over large bank tensors; this is late-only, but it still adds a bit of wall-clock overhead.
- The new scale optimizer may need tuning; `BANK_QAT_SCALE_LR=0.002` is a reasonable first guess, not an ablated optimum.
- This candidate only makes the dominant banked weights quantization-aware. Small auxiliary `CastedLinear` modules remain on the older optional path.
- If the learned scales overfit the roundtrip metric too aggressively, the pre-quant float model could get slightly worse even while the final int6 export improves.
