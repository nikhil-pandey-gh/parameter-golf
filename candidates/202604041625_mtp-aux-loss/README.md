# 202604041625_mtp-aux-loss

## Hypothesis

Add a **training-only multi-token-prediction (MTP) auxiliary loss** to the strongest non-TTT local stack so the model gets more learning signal per token under the fixed 600s budget, while keeping the exported artifact unchanged by stripping the auxiliary heads before serialization.

I also keep the candidate on the strongest currently demonstrated local footing by swapping the base stack's `relu^2` MLP for **LeakyReLU(0.5)^2**, which is already a proven zero-parameter win elsewhere in this repo.

## Why this is promising here

The repo evidence is clear about two things:

1. Tiny frontier gains now mostly come from **small, cheap, orthogonal tweaks** rather than large architectural rewrites.
2. Slow ideas lose badly under the 10-minute wallclock, so new ideas need to improve **sample efficiency** more than they hurt step time.

MTP is attractive because it is:

- **train-time only**: the auxiliary heads are excluded from the exported persistent weights, and the artifacts now record export-time `model_kwargs` with `mtp_num_heads=0`,
- **already close to the repo's code path**: recent 11-layer record branches carry dormant MTP support but never enable it,
- **orthogonal** to the current winning stack of EMA + XSA + partial RoPE + GPTQ-lite + low-bit export.

## Local experiments that influenced this candidate

- **Base implementation**: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - best non-TTT stack in this checkout,
  - already includes the strongest compression-aware recipe here: 11 layers, XSA, partial RoPE, LN scale, EMA, GPTQ-lite, warmdown 3500, late-QAT threshold 0.15.
- **Activation change borrowed from**: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - demonstrated that LeakyReLU(0.5)^2 is a cheap positive delta on a nearby frontier branch.
- **Compression-aware history**:
  - `2026-03-18_FP16Embed_WD3600`
  - `2026-03-19_WarmdownQuantization`
  - these runs reinforced that export quality matters as much as pre-quant loss, which makes export-free training auxiliaries especially appealing.

There were **no prior `candidates/` directories** in this repository when this candidate was created.

## External research that informed it

- **Gloeckle et al., _Better & Faster Large Language Models via Multi-Token Prediction_ (2024)**  
  Core motivation for using future-token auxiliary supervision to improve training efficiency.
- **Mehra et al., _On multi-token prediction for efficient LLM inference_ (arXiv:2502.09419, 2025)**  
  Useful caution: hidden states are specialized for next-token prediction, so MTP should be kept lightweight and not assumed to be free.
- **Gerontopoulos et al., _Multi-Token Prediction Needs Registers_ (arXiv:2505.10518, 2025)**  
  Reinforces that minimal-extra-parameter lookahead objectives can work well, even though this candidate uses the repo's simpler auxiliary-head path instead of register tokens.

## What changed vs the chosen base implementation

Compared with `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate:

1. **Enables MTP by default**
   - `MTP_NUM_HEADS=1`
   - `MTP_LOSS_WEIGHT=0.1`
2. **Bootstraps the MTP head from the default tied input/output projection**
   - under the default `TIE_EMBEDDINGS=1` setting, the auxiliary head now starts from the same tied matrix used for logits instead of zeros,
   - intended to save short-run optimization budget.
3. **Switches the MLP activation to LeakyReLU(0.5)^2**
   - chosen because the repo already shows this as a zero-size positive tweak.
4. **Makes the script runnable from inside the candidate directory**
   - default dataset and tokenizer paths resolve relative to the repo root instead of the current working directory.
5. **Adds a CPU/no-FlashAttention import fallback**
   - uses PyTorch SDPA when `flash_attn_interface` is unavailable, which helps lightweight local smoke imports.

Everything else is intentionally kept as close as possible to the 2026-03-22 stack.

## How to run

From this candidate directory:

```bash
cd candidates/202604041625_mtp-aux-loss
RUN_ID=mtp_aux_loss torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful overrides:

```bash
MTP_NUM_HEADS=0              # disable the auxiliary loss
MTP_LOSS_WEIGHT=0.05         # lighter MTP
MTP_INIT_FROM_LM_HEAD=0      # ablate the copied initialization
DATA_PATH=/path/to/dataset
TOKENIZER_PATH=/path/to/tokenizer.model
```

Artifacts and logs are written into this candidate directory:

- `logs/`
- `final_model.pt`
- `final_model.int6.ptz`

Both model artifacts include `model_kwargs` metadata describing the export-time architecture (including `mtp_num_heads=0`).

## Main expected risks / tradeoffs

- **Step-time overhead**: even one auxiliary vocab head costs extra compute; if the sample-efficiency win is too small, total progress may regress under the 600s cap.
- **Objective mismatch**: next-token hidden states are not guaranteed to be ideal for token+2 prediction.
- **Interaction risk**: LeakyReLU² and MTP are individually plausible here, but their combination is not yet ablated in this repo.
- **No artifact cost does not mean no training cost**: excluding MTP heads from export only helps if the extra training signal outweighs the slower updates.

## Validation

| Command | Outcome |
|---|---|
| `python -m compileall candidates/202604041625_mtp-aux-loss/train_gpt.py` | Passed |
| CPU import/forward smoke via `importlib` + tiny `GPT(...)` instantiation | Attempted, but this runner's local Python environment does not have `torch` installed (`ModuleNotFoundError: No module named 'torch'`), so a true CPU smoke test was not feasible here |
