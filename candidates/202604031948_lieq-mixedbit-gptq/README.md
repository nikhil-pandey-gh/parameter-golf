# LieQ-inspired mixed-bit GPTQ-lite on the 1.1194 stack

## Hypothesis

The current best training stack in this repo is already very strong; the next cheap win is more likely to come from **where** we spend quantization precision than from another broad architectural change. This candidate keeps the `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` recipe intact, but replaces its uniform blockwise int6 export with a **layer-sensitive mixed-bit plan**:

- keep the most sensitive late projections at **int8**,
- push the least sensitive early projections down to **int5**,
- leave the rest at **GPTQ-lite int6**.

## Why this looks promising here

Repository history says the biggest durable gains came from better export/eval choices on top of the same compact GPT backbone: sliding-window eval, mixed int6/int8 export, int5/int6 tradeoffs, EMA, GPTQ-lite clip search, and finally the LeakyReLU^2 + legal TTT stack. The main dead ends were bigger architectural departures with unclear compute tradeoffs, including layer recurrence on the non-record 5090 exploration.

This candidate leans into that pattern instead of fighting it: keep the best-known 11-layer banked model and only change the quantization allocation.

## Prior repo work that influenced this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
- **Quantization baseline to beat:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- **Earlier mixed-precision export:** `records/track_10min_16mb/2026-03-19_MixedQuant_Int6Int8_SlidingWindow/`
- **Int5 funding idea:** `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/`
- **No prior `candidates/` directory existed when this candidate was created.**

## External research that informed it

- **LieQ / "Exploring Layer-wise Information Effectiveness for Post-Training Quantization in Small Language Models" (arXiv:2508.03332)** argues that small LMs benefit from hardware-friendly mixed precision with more bits reserved for functionally critical layers.
- **"Scaling Law for Quantization-Aware Training" (arXiv:2505.14302)** finds that quantization bottlenecks concentrate in specific layers, with FC2-style projections being especially sensitive to outliers; mixed precision helps close that gap.
- **APTQ / "Attention-aware Post-Training Mixed-Precision Quantization for Large Language Models" (arXiv:2402.14866)** motivates spending extra precision on attention-sensitive parts rather than using a uniform bitwidth everywhere.
- **EfficientQAT (arXiv:2407.11062)** reinforces the same broader lesson: selective quantization treatment matters more than uniform low-bit compression when the budget is tight.

## What changed versus the chosen base implementation

Relative to `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`, this candidate only changes the export policy:

1. Adds four mixed-bit knobs:
   - `MPQ_INT8_MLP_PROJ_LAST_N=2`
   - `MPQ_INT8_ATTN_OUT_LAST_N=2`
   - `MPQ_INT5_ATTN_QKV_FIRST_N=5`
   - `MPQ_INT5_MLP_UP_FIRST_N=3`
2. Reuses the record's existing bank unroll/re-roll path so quantization can still act on per-layer tensors.
3. Generalizes the existing GPTQ-lite percentile clip search from fixed int6 to variable **int5 / int6 / int8**.
4. Logs the chosen mixed-bit plan before serializing the artifact.

Default mixed-bit allocation:

- **int8**
  - last 2 `mlp.proj` tensors
  - last 2 `attn.proj` tensors
- **int5**
  - first 5 `attn.c_q`, `attn.c_k`, `attn.c_v` tensors
  - first 3 `mlp.fc` tensors
- **int6**
  - every other block attention/MLP matrix
- **int8 / fp pass-through**
  - embeddings and the same control tensors the base record already preserves

## Why this differs from existing records

This repo has already tried:

- **uniform block int6 + int8 embeddings**
- **global int5 MLP / int6 attention**
- **uniform GPTQ-lite int6 clip search**

It has **not** yet tried a **late-layer rescue + early-layer demotion** policy on the strongest banked 1.1194 stack. That is the novel twist here.

## How to run

From this directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
MPQ_INT8_MLP_PROJ_LAST_N=2 MPQ_INT8_ATTN_OUT_LAST_N=2 \
MPQ_INT5_ATTN_QKV_FIRST_N=5 MPQ_INT5_MLP_UP_FIRST_N=3 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Evaluation notes

- The final training/eval flow is still the same as the 2026-03-23 record:
  - train under the 600s wallclock cap,
  - export a compressed artifact,
  - roundtrip dequantize and score it,
  - run stride-64 sliding evaluation,
  - optionally run legal score-first TTT.
- The mixed-bit artifact is written as `final_model.intmix.ptz`.

## Main expected risks and tradeoffs

- **Artifact risk:** int8 rescue on late layers may help BPB but still lose the size battle if lzma compression does not recover enough bytes from the int5 demotions.
- **Sensitivity risk:** the early-layer int5 assumptions are theory-backed but unverified on this exact stack; first-layer Q/K/V may be more fragile than expected.
- **Interaction risk:** legal TTT changes the post-export loss landscape, so the best mixed-bit allocation for no-TTT eval may not be the best after adaptation.
- **Tuning risk:** the chosen `2 / 2 / 5 / 3` allocation is a principled first shot, not yet an ablated optimum.

## Validation

Commands run in this repository:

```bash
python -m compileall candidates/202604031948_lieq-mixedbit-gptq/train_gpt.py
```

Outcomes:

- `compileall`: passed
- Minimal CPU-only smoke test: not feasible without changing the candidate, because it intentionally preserves the base record's CUDA-only training/eval path and real-data assumptions.
