# MTP Auxiliary Heads on the 2026-03-23 Record Stack

## Hypothesis

Add a **training-only multi-token prediction (MTP)** auxiliary loss to the strongest end-to-end record stack so the backbone learns faster within the fixed 600s training budget, while the final exported artifact size stays effectively unchanged because the auxiliary heads are dropped before quantized export.

## Why this is promising for this repository

- Repo history shows that **sample efficiency and evaluation-aware tricks** matter more than raw architectural novelty under the 10-minute cap.
- Repo history also shows that **layer recurrence / weight sharing** was a regression under fixed wall-clock, and that some more aggressive quantization ideas are fragile or expensive to wire correctly here.
- MTP is attractive because it is **orthogonal** to the winning stack in this repo: it changes the **training objective**, not the deployed model shape.
- The candidate keeps the current best full-stack recipe (LeakyReLU² + legal score-first TTT + parameter banking + parallel Muon), then adds a **cheap auxiliary objective** whose parameters are excluded from export.

## Record influences

There were **no prior `candidates/`** in this repository at the start of this run, so this candidate is informed entirely by the existing `records/` lineage.

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - strongest **non-TTT** training stack in the repo
  - reinforced that the durable wins are 11L depth, MLP 3x, XSA, partial RoPE, LN scale, VE128, EMA/SWA, and GPTQ-lite-aware export
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - strongest **end-to-end** result in the repo
  - contributed LeakyReLU(0.5)^2, legal score-first TTT, BigramHash(1536), parameter banking, and parallel Muon
- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/`
  - documented that **SwiGLU was slower and net-negative** on the throughput-constrained path
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
  - documented that **layer recurrence / layer reuse regressed badly** under a fixed wall-clock budget

Those dead ends are why this candidate avoids the more invasive external ideas like cross-layer sharing or recurrent depth unrolling, even though they remain interesting in the abstract.

## External research that informed the choice

- **Fabian Gloeckle et al., “Better & Faster Large Language Models via Multi-token Prediction” (arXiv:2404.19737)**  
  argues that predicting multiple future tokens with independent heads on a shared trunk improves sample efficiency and can be used as an auxiliary training task.
- **DeepSeek-AI, “DeepSeek-V3 Technical Report” (arXiv:2412.19437)**  
  explicitly states that DeepSeek-V3 uses a **multi-token prediction training objective for stronger performance**.
- **John Kirchenbauer et al., “Multi-Token Prediction via Self-Distillation” (arXiv:2602.06019)**  
  strengthens the case that MTP is a lightweight way to improve or repurpose autoregressive LMs without changing the final deployed interface.

I considered two other strong external directions and rejected them for this repo iteration:

1. **Cross-layer sharing / recurrent depth**: promising byte-for-byte in general, but prior repo experiments already showed recurrence was a bad trade against lost optimizer steps.
2. **QuaRot / SpinQuant-lite style rotation-aware low-bit training**: compelling, but it requires broader quantization-path surgery than this candidate and is riskier for a first `candidates/` drop.

## What changed vs the chosen base implementation

Chosen base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes:

1. **Enable MTP by default**
   - `MTP_NUM_HEADS=1`
   - `MTP_LOSS_WEIGHT=0.1`
2. **Promote the best-record runtime defaults into the candidate script**
   - `ITERATIONS=9000`
   - `BIGRAM_VOCAB_SIZE=1536`
   - `TTT_ENABLED=1`
   - `TTT_FREEZE_BLOCKS=0`
3. **Make the script runnable from the candidate directory**
   - dataset and tokenizer defaults are resolved by walking ancestors from `__file__` until the repo root is found
4. **Add a FlashAttention fallback**
   - if `flash_attn_interface` is unavailable, attention falls back to PyTorch SDPA and enables the math / mem-efficient SDP backends needed for that path
5. **Keep export size behavior unchanged**
   - `mtp_heads` are still excluded from the exported state dict before quantized packaging

## How to run / evaluate

From the repository root:

```bash
cd candidates/202604060609_mtp-aux-heads
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The defaults are set to the candidate configuration. Important overrides for ablations:

```bash
# disable the new idea
MTP_NUM_HEADS=0 MTP_LOSS_WEIGHT=0.0

# disable expensive eval-time adaptation
TTT_ENABLED=0

# try a stronger auxiliary objective
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.1
```

The script still performs the same overall pipeline as the 2026-03-23 base:

- train under the 600s wall-clock cap,
- export an int6-compressed artifact,
- run roundtrip + sliding-window evaluation,
- then run legal score-first TTT evaluation by default.

## Main risks / tradeoffs

- **Training throughput risk**: even one auxiliary head adds extra vocab projections and cross-entropies; the MTP gain must outweigh any lost steps.
- **Eval budget remains heavy**: legal TTT is still a major part of end-to-end evaluation time.
- **Auxiliary-objective tuning risk**: too much MTP weight or too many heads could regularize the model in the wrong direction for BPB.
- **Portability is improved, not fully generalized**: the SDPA fallback helps imports and non-Flash environments, but the full training loop still requires CUDA.

## Validation

| Command | Outcome |
| --- | --- |
| `python -m compileall candidates/202604060609_mtp-aux-heads/train_gpt.py` | **Passed** |
| `cd candidates/202604060609_mtp-aux-heads && python - <<'PY' ...` (CPU import / forward smoke) | **Blocked by environment**: local workflow Python raised `ModuleNotFoundError: No module named 'torch'` before the smoke could run |

The smoke-test failure is an environment limitation in this workflow container, not a repo-level dependency omission: `requirements.txt` includes `torch`, and the repository README notes that the target evaluation environment has the required packages installed.
