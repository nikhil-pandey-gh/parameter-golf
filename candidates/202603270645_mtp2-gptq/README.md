# MTP2 + EMA + GPTQ-lite warmdown3500

## Hypothesis

A small amount of **training-only multi-token prediction (MTP)** should improve sample efficiency on this repository's 600-second training budget without increasing the exported artifact size. The extra MTP heads are used only during training and are explicitly excluded from export, so the 16 MB budget remains focused on the main next-token model.

## Why this is promising for this repository

The record history shows that most recent gains came from compression-aware training, local inductive biases, and evaluation tricks on top of a strong 11-layer stack. External research suggests there is still underused headroom in the **training objective** itself: the MTP literature reports better token-efficiency from predicting multiple future tokens, which is especially attractive in a hard wall-clock regime.

This repo already carried dormant MTP support in several late-record codepaths, but every reviewed run kept it disabled (`MTP_NUM_HEADS=0`). That makes this candidate a clean next step: turn on a training-only objective that fits the existing code structure and costs no export bytes.

## Prior records that influenced this candidate

This candidate is based directly on `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`, which is the strongest reviewed non-TTT stack and already includes:

- 11 layers at 512d with U-Net skips
- XSA on the last 4 layers
- partial RoPE + LN scale
- SmearGate + BigramHash + shared value embedding
- EMA + tight SWA
- GPTQ-lite style per-row clip search for int6 export

Other reviewed records that shaped the choice:

- `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`: shows that small, zero- or low-byte training tweaks can still move this stack.
- `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`: shows the leaderboard still has room from better training/eval ideas, but TTT adds substantial implementation and eval complexity.
- `2026-03-20_11L_EfficientPartialXSA_FA3_SWA120`: explicitly logged `MTP_NUM_HEADS=0`, confirming MTP machinery existed but was not actually explored in the reviewed records.

## External research that informed it

- **Better & Faster LLMs via Multi-token Prediction** (arXiv:2404.19737): motivates auxiliary future-token heads as a sample-efficiency improvement during pretraining.
- **Multi-Token Prediction Needs Registers** (arXiv:2505.10518): reinforces that lightweight future-token objectives remain a live research direction for language modeling efficiency, even if the full register mechanism is more invasive than needed here.
- Supporting compact-model context from the review also included ALBERT and DeFINE style parameter-efficiency papers, but MTP was the best fit because it improves training efficiency without consuming artifact bytes.

## What changed versus the chosen base implementation

Relative to the `2026-03-22` base:

- enabled **2-head MTP by default** with `MTP_NUM_HEADS=2`
- set a more conservative default auxiliary weight `MTP_LOSS_WEIGHT=0.15`
- added **horizon weights** via `MTP_HEAD_WEIGHTS=1.0,0.5`, so the +1-token head dominates the +2-token head
- kept the existing export behavior that **drops `mtp_heads` from the saved artifact**
- added a small **FlashAttention fallback** to PyTorch SDPA so the script can still import and run a CPU smoke test when `flash_attn_interface` is unavailable
- added logging for the active attention backend and MTP head weights

## How to run or evaluate it

From the repository root:

```bash
cd candidates/202603270645_mtp2-gptq
SEED=1337 \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful overrides for ablations:

```bash
MTP_NUM_HEADS=0 MTP_LOSS_WEIGHT=0.0             # recover the base objective
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.10            # lighter auxiliary objective
MTP_HEAD_WEIGHTS=1.0,0.25                       # make the +2 horizon gentler
```

## Validation

Validation run in this workflow:

```bash
python -m compileall candidates/202603270645_mtp2-gptq/train_gpt.py
python - <<'PY'
import importlib.util
from pathlib import Path
import torch
spec = importlib.util.spec_from_file_location('cand', Path('candidates/202603270645_mtp2-gptq/train_gpt.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
model = mod.GPT(
    vocab_size=128,
    num_layers=4,
    model_dim=64,
    num_heads=4,
    num_kv_heads=2,
    mlp_mult=2.0,
    tie_embeddings=True,
    tied_embed_init_std=0.02,
    logit_softcap=30.0,
    rope_base=10000.0,
    qk_gain_init=1.0,
    mtp_num_heads=2,
    mtp_loss_weight=0.15,
    mtp_head_weights='1.0,0.5',
    bigram_vocab_size=64,
    bigram_dim=16,
    xsa_last_n=1,
    rope_dims=8,
    ln_scale=True,
    dtg=False,
    ve_enabled=False,
)
model.eval()
x = torch.randint(0, 128, (2, 16))
y = torch.randint(0, 128, (2, 16))
with torch.no_grad():
    loss = model(x, y)
print(float(loss))
PY
```

Observed outcome in this workflow:

- `python -m compileall candidates/202603270645_mtp2-gptq/train_gpt.py` **passed**.
- The CPU smoke test command could **not** be completed on this runner because the workflow environment does not have `torch` installed (`ModuleNotFoundError: No module named 'torch'` before the candidate module import ran).
- No full training run was attempted here because the real training path requires CUDA plus the FineWeb shards/tokenizer.

If you run the smoke command in the normal Parameter Golf environment (where PyTorch is installed), it should instantiate the model and exercise the SDPA fallback path when `flash_attn_interface` is unavailable.

## Main expected risks / tradeoffs

- MTP improves token efficiency in the literature, but it still adds **training-time FLOPs and parameters** during optimization, so the gain must beat the lost steps.
- The existing MTP implementation predicts future tokens from the same hidden states without extra register tokens, so it is a **minimal** version of the research idea, not the strongest published variant.
- The best stack in the repo already sits near a locally optimized operating point; it is plausible that MTP needs follow-up tuning of `MTP_LOSS_WEIGHT`, warmdown behavior, or `MTP_NUM_HEADS` to pay off.
