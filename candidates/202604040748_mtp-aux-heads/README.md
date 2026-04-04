# MTP Auxiliary Heads on 11L EMA + GPTQ-lite + LeakyReLU²

## Hypothesis

Adding a small **training-only multi-token prediction (MTP)** objective to the best pure-training 11-layer stack should improve sample efficiency inside the fixed 600s budget. The key appeal for Parameter Golf is that the auxiliary `mtp_heads` are **excluded from export**, so the candidate can spend extra train-time compute without paying extra artifact bytes at evaluation time.

This candidate also carries over **LeakyReLU(0.5)²** from the latest overall record because it is the strongest repo-proven zero-byte activation tweak on a closely related stack.

## Why this is promising for this repository

- The best non-TTT checkpoint family in `records/` is still the 11-layer EMA + GPTQ-lite line:
  `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md`.
- The latest overall record shows that **LeakyReLU²** is a meaningful one-line improvement on the modern frontier stack:
  `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`.
- Multiple frontier `train_gpt.py` snapshots already contain dormant MTP hooks, but the published configs leave `MTP_NUM_HEADS=0`. This candidate turns that unused path into the main experiment instead of inventing new infrastructure.
- Negative results in prior records suggest avoiding more aggressive recurrent sharing or heavier architectural rewrites under the 10-minute cap, so MTP is a better fit than recurrence-style ideas.

## Prior records that influenced this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`
- **Activation carry-over:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`
- **Dead-end to avoid:** recurrence under fixed wallclock was negative in
  `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md` and
  `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md`

There were **no prior `candidates/` directories** in the repository when this candidate was created.

## External research that informed it

- **Fabian Gloeckle et al., “Better & Faster Large Language Models via Multi-token Prediction” (arXiv:2404.19737)**  
  The paper reports that predicting multiple future tokens with independent auxiliary heads on a shared trunk improves sample efficiency and helps induction-style behavior, while keeping the core model unchanged.
- I also reviewed **nGPT** (arXiv:2410.01131) and **LSQ** (arXiv:1902.08153) as alternate directions, but MTP was the best fit here because it targets training efficiency without broad architecture or quantization changes.

## What changed versus the chosen base

Relative to `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`:

1. **Enabled export-free MTP by default**
   - `MTP_NUM_HEADS=1`
   - `MTP_LOSS_WEIGHT=0.1`
2. **Switched the MLP activation to LeakyReLU(0.5)²**
3. **Made FlashAttention optional for local validation**
   - if `flash_attn_interface` is unavailable, attention falls back to PyTorch SDPA and can use whichever built-in backend is available
   - the intended H100 path still uses FlashAttention when installed

Everything else stays close to the strong 11-layer EMA + GPTQ-lite + partial-RoPE + XSA4 + VE128 + BigramHash stack so the new variable is mostly the auxiliary training signal.

## How to run

From this candidate directory:

```bash
cd candidates/202604040748_mtp-aux-heads
RUN_ID=mtp_aux \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful ablations:

```bash
# Disable the new idea
MTP_NUM_HEADS=0 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Stronger auxiliary signal if throughput headroom exists
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.1 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script still performs the usual post-training quantization, roundtrip validation, and sliding-window evaluation.

## Validation

- `python3 -m compileall candidates/202604040748_mtp-aux-heads/train_gpt.py` — **passed**
- A minimal CPU/import smoke test was **not feasible in this runner** because the local Python environment does not have `torch` or `sentencepiece` installed. The candidate does remove `flash_attn_interface` as a hard local dependency by adding an SDPA fallback.

## Main risks and tradeoffs

- The extra MTP head may reduce total step count enough to erase its sample-efficiency gain.
- MTP evidence is strongest on generative/code-style tasks; the transfer to FineWeb BPB may be real but smaller.
- LeakyReLU² and MTP are individually plausible here, but their interaction with EMA + GPTQ-lite + late-QAT may still need a small sweep over `MTP_NUM_HEADS` and `MTP_LOSS_WEIGHT`.
