# Sparse training-only MTP on the 2026-03-23 stack

## Hypothesis

Add a **single sparse multi-token prediction (MTP) auxiliary head** to the strongest current 11-layer stack so the model gets denser training signal without paying any final artifact cost. The extra head is **training-only**: it is optimized during training, then stripped before export and quantization.

## Why this is promising here

The record lineage already looks saturated on the obvious knobs:

- sliding-window evaluation is already standard,
- tied-embedding protection and int6-friendly export are already strong,
- the best stacks already combine 11 layers, XSA, partial RoPE, LN scale, EMA, GPTQ-lite-style export, and LeakyReLU^2,
- recurrence and slower MLP variants like SwiGLU have already looked unattractive under the 10-minute wall-clock.

What has **not** been seriously explored in the record summaries is MTP, even though the strongest script lineage already contained dormant MTP/export scaffolding. That makes this repository unusually well-positioned to test MTP with only a small code delta.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`: strongest overall stack; this candidate starts from its LeakyReLU^2 + parameter-banked 11L backbone.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`: strongest no-TTT lineage; reinforces that low-overhead post-training improvements still matter once the backbone is strong.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`: partial RoPE + LN scale remain part of the modern winning recipe.
- `records/track_10min_16mb/2026-03-20_11L_EfficientPartialXSA_FA3_SWA120/` and `records/track_10min_16mb/2026-03-19_smeargate_orthoinit_muonwd/`: reinforce that zero/low-parameter architectural tweaks and the SmearGate + BigramHash front end are durable wins.

## External research informing the idea

- **Gloeckle et al., _Better & Faster Large Language Models via Multi-token Prediction_ (arXiv:2404.19737)**: argues that predicting multiple future tokens with independent heads improves sample efficiency and downstream capability.
- **DeepSeek-V3 Technical Report (arXiv:2412.19437)**: explicitly includes a multi-token prediction training objective in a frontier training stack.
- **Mehra et al., _On multi-token prediction for efficient LLM inference_ (arXiv:2502.09419)**: shows that naive retrofitting is hard and that joint training matters, which matches this repository's from-scratch setting.
- **Gerontopoulos et al., _Multi-Token Prediction Needs Registers_ / MuToR (arXiv:2505.10518)**: strengthens the case that parameter-light MTP variants are worth exploring when added capacity is constrained.

The repository-specific twist here is **sparsity**: compute the auxiliary loss on every Nth position so the objective stays compatible with the 10-minute training budget.

## What changed vs the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes:

1. **Candidate-local repo-root defaults** so the script runs from this candidate directory without overriding `DATA_PATH` or `TOKENIZER_PATH`.
2. **Enable one auxiliary MTP head by default**:
   - `MTP_NUM_HEADS=1`
   - `MTP_LOSS_WEIGHT=0.1`
3. **Add sparse supervision** with `MTP_TOKEN_STRIDE=4`, so only every 4th eligible position contributes to the auxiliary loss.
4. **Fix optimizer wiring** so `mtp_heads` are actually optimized. The base record already excluded them from export, but they were not wired into an optimizer group.
5. Keep the existing **export-time MTP stripping**, so the auxiliary head does not count toward the serialized/quantized artifact.

## How to run

From this candidate directory:

```bash
cd candidates/202604051722_sparse-mtp
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful ablations:

```bash
# Disable the auxiliary objective
MTP_NUM_HEADS=0 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Denser auxiliary supervision
MTP_TOKEN_STRIDE=2 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Validation

Commands run:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604051722_sparse-mtp/train_gpt.py
python - <<'PY'
import importlib.util
print(f"torch_installed={importlib.util.find_spec('torch') is not None}")
print(f"flash_attn_interface_installed={importlib.util.find_spec('flash_attn_interface') is not None}")
PY
```

Outcome:

- `compileall`: passed for the baseline scripts, `data/`, and this candidate's `train_gpt.py`
- CPU smoke test: not feasible on this runner because neither `torch` nor `flash_attn_interface` is installed, and the script is written for the CUDA/FlashAttention runtime used by the challenge environment

## Main risks and tradeoffs

- Even sparse MTP may still cost enough throughput to erase any sample-efficiency gain.
- A single future-token head may be too weak; more heads may be too expensive.
- This candidate focuses on **training** improvements only; it does not add a new quantization or evaluation trick.
