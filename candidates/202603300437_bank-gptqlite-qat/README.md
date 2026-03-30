# Late Bank GPTQ-lite QAT

## Hypothesis

The strongest current stack in this repository already has most of the right architectural and evaluation tricks, but its biggest weight tensors still do **not** train against the same low-bit quantizer used at export time. This candidate tests whether turning on **compile-safe, late int6 fake quantization for the banked attention and MLP weights** can reduce the remaining post-training quantization gap and improve post-quant sliding-window BPB.

Concretely: start from the March 23 record, keep its 11-layer / XSA / partial-RoPE / LeakyReLU^2 / EMA / legal TTT recipe, and add a late-training bank-QAT path whose per-row scales are chosen with the same percentile search used by the repo's GPTQ-lite export.

## Why this is promising for this repository

Repository evidence points in the same direction:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` is the current best 10-minute stack (`val_bpb: 1.1194`) and is the closest practical base.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md` shows GPTQ-lite clip selection still buys measurable post-training compression quality.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md` explicitly notes that one late-QAT attempt was compile-folded away and had no real effect.
- `records/track_10min_16mb/2026-03-19_MLP3x_QAT_Int6_SlidingWindow/README.md` and `records/track_10min_16mb/2026-03-19_smeargate_orthoinit_muonwd/README.md` show that broader int6 QAT can help on earlier, weaker stacks.
- `records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/README.md` shows that extra training alone does not erase the quantization bottleneck.

The missing piece is that the March 23 record uses large parameter banks (`qo_bank`, `kv_bank`, `mlp_up_bank`, `mlp_down_bank`) directly in `F.linear(...)`, so its existing `CastedLinear._qat_enabled` switch only touches the much smaller auxiliary projection layers.

## Prior experiments that influenced this candidate

No prior `candidates/` directory existed when this workflow started.

The most relevant record influences were:

- `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` for the base architecture and training/eval stack.
- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` for GPTQ-lite percentile clip search.
- `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` for the compile-folding warning on late QAT.
- `2026-03-19_MLP3x_QAT_Int6_SlidingWindow` and `2026-03-19_smeargate_orthoinit_muonwd` as earlier proof that broader low-bit QAT is plausible in this benchmark.

## External research that informed it

The idea is grounded in a small set of weight-quantization papers that map cleanly onto this repo:

- **LSQ** argues that matching training to low-bit deployment improves low-precision performance, and that the quantizer configuration itself matters: <https://arxiv.org/abs/1902.08153>
- **GPTQ** motivates spending a little extra compute on better low-bit scale selection for weight matrices: <https://arxiv.org/abs/2210.17323>
- **AWQ** highlights that weight-only quantization quality depends on protecting the right channels, not just using a naive global clip: <https://arxiv.org/abs/2306.00978>
- **QDrop** supports the broader intuition that exposing training to quantization noise can improve robustness at very low precision: <https://arxiv.org/abs/2203.05740>

I also considered more aggressive shared-weight / recurrent-depth ideas from ALBERT and Universal Transformer, but the repository already contains a negative recurrence result under a fixed wall-clock budget, so this candidate stays closer to the current winning recipe.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate's `train_gpt.py`:

- Added `BANK_QAT_ENABLED` and `BANK_QAT_PERCENTILES` knobs.
- Added GPTQ-lite-style per-row scale selection for the four large parameter banks.
- Added fake int6 quantization for bank slices during training via STE once late QAT turns on.
- Recompiled the model right when late QAT activates so the bank-QAT branch is not optimized away by `torch.compile`.
- Kept the existing small `CastedLinear` late-QAT path, so both small auxiliary projections and large banks can train against low-bit noise.
- Dropped the copied base file's inert SWA knob so the candidate's documented controls match the code that actually runs.
- Added a narrow FlashAttention import fallback to PyTorch SDPA so import-level validation is possible on machines without `flash_attn_interface`.
- Updated the default `DATA_PATH` and `TOKENIZER_PATH` to resolve from the repository root, so the script can be run directly from this candidate directory.

## How to run or evaluate

Run from this candidate directory:

```bash
cd candidates/202603300437_bank-gptqlite-qat

NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
BANK_QAT_ENABLED=1 BANK_QAT_PERCENTILES=0.9990,0.9995,0.9999,0.99999,1.0 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Notes:

- The default dataset and tokenizer paths now resolve to the repository root automatically, so no extra path overrides are needed for the standard setup.
- `flash_attn_interface` is still preferred for real GPU runs. The SDPA fallback is mainly for portability and lightweight validation.
- To measure the effect before TTT, rerun with `TTT_ENABLED=0`.

## Main expected risks or tradeoffs

- Recompiling once when bank QAT activates costs some wall-clock time in a tight 600-second budget.
- The selected QAT scales are frozen when late QAT turns on. If the weights keep drifting materially after that point, the fake quantizer may become slightly stale.
- The best current score uses heavy legal TTT, so it may be hard to tell whether a gain came from better training-time quantization or better downstream TTT dynamics without separate ablations.
- The new SDPA fallback is not meant to be a record-path replacement for FlashAttention on H100s.

## Validation

Commands run in this workflow:

```bash
python -m compileall candidates/202603300437_bank-gptqlite-qat/train_gpt.py
```

Outcome:

- `python -m compileall .../train_gpt.py` succeeded.
- A minimal CPU forward-pass smoke test was attempted, but this runner does not have the repository's required `torch` dependency installed (`requirements.txt` lists `torch`), so an import-level execution smoke test was not feasible in this environment without the missing training stack.
