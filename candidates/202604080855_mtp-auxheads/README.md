# Core 11L + LeakyReLU^2 + MTP Aux Heads

## Hypothesis

Add a small number of **training-only multi-token-prediction (MTP) heads** to the strongest pre-TTT core stack so the trunk gets denser future-token supervision during the fixed 10-minute training budget, while **excluding those heads from export** so the final artifact stays effectively unchanged. I also fold in the **LeakyReLU(0.5)^2** MLP change from the current best record because it was a clean win on the same model family and is orthogonal to MTP.

## Why this is promising here

The records show a consistent pattern:

- early wins came from **more context** and **better evaluation** (`Seq2048`, `Seq4096`, sliding-window eval),
- mid-stage wins came from **compression-funded capacity** and **quantization-aware smoothing** (int6, 3x MLP, BigramHash, EMA, warmdown),
- the most recent wins came from **cheap, no-artifact architectural tweaks** like XSA, Partial RoPE, LN scaling, and LeakyReLU^2.

MTP fits that pattern well:

- it is a **training-time-only** change,
- it already fits this codebase's trunk/head structure,
- its auxiliary heads are already designed to be **dropped before serialization**,
- and none of the existing records actually ran with MTP enabled even though the core stack already contains dormant support for it (`mtp_num_heads:0` in the record logs).

## Prior records that influenced this candidate

- **Primary base:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - strongest clean train/export stack before the leaderboard moved into heavier eval-time TTT.
- **Activation borrowed from:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - LeakyReLU(0.5)^2 was the largest single ablated improvement in that README.
- **Architecture lineage kept intact from the record family:**
  - `2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271`
  - `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`
  - `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`

There were no prior `candidates/` directories in this repository when this candidate was created.

## External research that informed it

- **Fabian Gloeckle et al., 2024 — _Better & Faster Large Language Models via Multi-token Prediction_ (arXiv:2404.19737)**  
  Core motivation for this candidate: independent future-token heads on a shared trunk can improve sample efficiency with no inference-time requirement to keep those heads.
- **Somesh Mehra et al., 2025 — _On multi-token prediction for efficient LLM inference_ (arXiv:2502.09419)**  
  Useful caution: frozen NTP trunks are specialized, but **jointly training** MTP heads with the backbone is the right regime; this candidate trains from scratch rather than retrofitting a frozen model.
- **Anastasios Gerontopoulos et al., 2025 — _Multi-Token Prediction Needs Registers_ (arXiv:2505.10518)**  
  Reinforces that lightweight MTP variants can stay parameter-efficient and compatible with standard language-model training.

I also considered more invasive ideas from the same research pass, especially **shared-MLP parameter reuse** and **stronger LSQ+/AdaRound/QDrop-style quantization**, but those would require broader export or quantization changes. MTP was the strongest idea that could be tested by adapting the existing training code directly.

## What changed vs. the chosen base

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`:

1. **Enable MTP in the documented candidate run**
   - candidate run uses `MTP_NUM_HEADS=2`
   - candidate run uses `MTP_LOSS_WEIGHT=0.15`
2. **Switch the MLP activation from ReLU^2 to LeakyReLU(0.5)^2**
   - this mirrors the strongest ablation from the 2026-03-23 record.
3. **Disable late QAT by default**
   - `LATE_QAT_THRESHOLD=0.0`
   - the 2026-03-21 record README notes that an earlier compiled late-QAT path was constant-folded away; this candidate isolates the MTP effect instead of pretending that late QAT is already proven in this stack.

Everything else stays on the proven 11-layer core stack: XSA on the deepest 4 layers, Partial RoPE, LN scaling, BigramHash, VE layers, EMA + tight SWA, GPTQ-lite-style int6 export, and sliding-window eval.

## How to run / evaluate

From this candidate directory:

```bash
python -m pip install zstandard  # if your environment does not already have it
```

Then run:

```bash
SEED=1337 \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
ARTIFACT_COMPRESSOR=zstd \
MTP_NUM_HEADS=2 \
MTP_LOSS_WEIGHT=0.15 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The compressor is explicit on purpose: `ARTIFACT_COMPRESSOR=zstd` gives deterministic artifact bytes instead of silently changing formats based on what happens to be installed.

Useful comparisons:

- **Disable the new idea:** `MTP_NUM_HEADS=0`
- **Ablate the activation:** replace the LeakyReLU^2 line in `train_gpt.py` back to ReLU^2
- **Sweep MTP strength:** try `MTP_NUM_HEADS=1` or `MTP_LOSS_WEIGHT=0.10`

## Validation

- `python -m compileall train_gpt.py` — **passed**
- Minimal CPU-only runtime smoke test — **not feasible in this environment**
  - this inherited stack hard-requires CUDA and FlashAttention 3, and the repository does not provide a CPU fallback path for the 11L Hopper-oriented record code.

## Main risks / tradeoffs

- **Step-time overhead:** extra vocabulary projections during training may cost enough wallclock to erase some of the sample-efficiency gain.
- **Model-size sensitivity:** the MTP literature is strongest at larger scales; a 16MB artifact model may benefit less.
- **Interaction risk:** LeakyReLU^2 and MTP are individually plausible, but the combination has not been tuned in this repo yet.
- **Quantization gap remains an open axis:** this candidate intentionally avoids mixing in a new quantization method so the result is easier to attribute.
