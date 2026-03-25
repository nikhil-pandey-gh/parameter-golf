## Candidate: 11L EMA + GPTQ-lite + next+2 MTP auxiliary head

### Hypothesis

The strongest recent non-TTT stack in this repository already contains dormant multi-token prediction (MTP) support, but every documented run leaves it disabled (`MTP_NUM_HEADS=0`). Enabling a single training-only auxiliary head that predicts the token two steps ahead should improve sample efficiency under the fixed 600s training budget without changing the exported artifact, because the extra head is excluded from serialization.

### Why this is promising here

This repo's frontier is already heavily optimized on quantization, sliding-window evaluation, partial RoPE, EMA/SWA, and small architectural priors. The remaining headroom is increasingly about getting more useful gradient signal per training step. Recent primary-source MTP results report better sample efficiency with no inference-time burden when auxiliary heads are used only during training.

I intentionally keep this candidate conservative:

- start from the best self-contained non-TTT record,
- add only **one** auxiliary horizon (`+2` token prediction),
- keep the MTP heads training-only so artifact size is unchanged,
- avoid deeper recurrence or MoE-style routing that would steal steps under the 10-minute cap.

### Prior repository influences

Primary base:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`

Additional influences:

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` for the mature 11-layer partial-RoPE/LN-scale stack.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` as evidence that training-time efficiency ideas still matter after the quantization stack is mature.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/` as a warning that classic layer recurrence is a bad fit for the hard wall-clock budget here.

There were no prior experiments under `candidates/` at the time this candidate was created.

### External research that informed this candidate

- Fabian Gloeckle et al., **"Better & Faster Large Language Models via Multi-Token Prediction"** (arXiv:2404.19737, 2024). The paper argues that predicting multiple future tokens with auxiliary heads improves sample efficiency while keeping deployment overhead low when those heads are training-only.
- This candidate also borrows the repository's existing instinct to protect export quality: like the prior records' quantization-only improvements, MTP is used here as a training-side helper while the shipped model remains the standard next-token model.

### What changed vs the chosen base implementation

Compared with `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`:

- default `MTP_NUM_HEADS` changes from `0` to `1`
- default `MTP_LOSS_WEIGHT` changes from `0.2` to `0.1`
- documentation is updated to treat MTP as the main hypothesis instead of a dormant code path

Everything else stays intentionally aligned with the proven base: 11 layers, seq 2048, EMA + tight SWA, GPTQ-lite int6 export, partial RoPE, XSA on the last 4 layers, BigramHash, SmearGate, and VE layers 9-10.

### How to run or evaluate

From this directory:

```bash
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Important defaults in this candidate:

- `MTP_NUM_HEADS=1`
- `MTP_LOSS_WEIGHT=0.1`
- `NUM_LAYERS=11`
- `TRAIN_SEQ_LEN=2048`
- `WARMDOWN_ITERS=3500`
- `EVAL_STRIDE=64`

To disable the new idea and recover the base behavior:

```bash
MTP_NUM_HEADS=0 MTP_LOSS_WEIGHT=0.0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Main expected risks / tradeoffs

- Even one extra auxiliary head adds output-projection and cross-entropy work during training, so too much MTP could reduce step count enough to cancel the sample-efficiency gain.
- The repository's target metric is still single-step next-token BPB; an overly strong auxiliary loss could bias optimization away from the exact export objective.
- Because this candidate reuses the mature 11-layer stack, the upside may be incremental rather than dramatic.

### Validation

Ran:

- `python -m compileall candidates/202603251033_mtp-next2/train_gpt.py` ✅

Attempted but blocked:

- CPU import / tiny-forward smoke test with `flash_attn_interface` stubbed and `xsa_last_n=0`
  - not feasible in this workflow container because the available `python` environment does not have `torch` installed (`ModuleNotFoundError: No module named 'torch'`)

So the candidate was syntax-checked successfully, but a runtime smoke test could not be completed in the current container.
