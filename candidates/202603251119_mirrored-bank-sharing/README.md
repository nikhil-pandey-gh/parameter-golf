# Mirrored Bank Sharing on the LeakyReLU² + Legal TTT stack

## Hypothesis

The current best stack in this repository already separates **large matrix weights** from **layer-local control tensors** via parameter banks. That makes it a good fit for a more artifact-aware architecture: **share only the heavy Q/K/V/O and MLP bank slices across mirrored layers, but keep each layer's RMSNorms, `q_gain`, `attn_scale`, `mlp_scale`, `resid_mix`, XSA behavior, and value-embedding scales local**.

The concrete bet is that this preserves most of the expressivity of the 2026-03-23 banked stack while cutting duplicated bank bytes enough to fund a slightly larger virtual stack and a larger local-feature table. In this candidate, the saved budget is spent on:

- **12 virtual layers** instead of 11, using a mirrored share map
- **3072 BigramHash buckets** instead of the banked stack's 1536 best-run setting
- keeping the rest of the strong frontier stack intact: **LeakyReLU², EMA + tight SWA, partial RoPE, LN scale, XSA on late layers, VE128, GPTQ-lite int6 export, and legal score-first TTT**

This is deliberately **not** naive recurrence. The model still executes all virtual layers distinctly; only the heaviest matrix parameters are shared.

## Why this looks promising for this repository

Repository review showed a very consistent progression:

- early gains came from **longer context and better evaluation**,
- then from **quantization-aware model design**,
- then from **11-layer / 3x-MLP / BigramHash / XSA / EMA / partial-RoPE** stacks,
- and finally from **LeakyReLU² + legal TTT + parameter banking**.

What is *not* saturated yet is **artifact-budget reallocation**. The top scripts aggressively optimize training speed and export quality, but they still mostly assign one unique heavy bank slice per virtual layer. External research suggests that **cross-layer sharing can be very parameter-efficient when layer-local controllers remain free to specialize**, which matches this codebase unusually well because the banked architecture already isolates the large weights from the small per-layer controls.

This candidate also addresses a dead-end from prior exploration: a non-record 1x5090 run found that simple layer recurrence was bad because it traded away too many optimizer steps. Mirrored bank sharing differs in two ways:

- it does **not** loop a single block repeatedly with the same controller state,
- and it keeps **per-layer control tensors unique**, so mirrored layers can still specialize.

## Influential prior work in this repo

There were **no prior experiments under `candidates/`** in this checkout, so the direct influences are record folders:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - provided the base script and the strongest current stack: parameter banking, legal score-first TTT, LeakyReLU², VE, partial RoPE, XSA, EMA/SWA, GPTQ-lite export
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - reinforced that export-side quantization improvements still matter at the frontier
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - showed that partial RoPE + LN scale are worth keeping
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`
  - established the 11L / XSA / EMA direction the later winners built on
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
  - provided a useful negative result on naive recurrence, which this candidate is intentionally avoiding

## External research that informed the idea

- **ALBERT**: parameter sharing across layers can dramatically reduce serialized model size while preserving performance if the sharing is structured well.
  - https://arxiv.org/abs/1909.11942
- **Universal Transformer**: depth reuse can be effective when the repeated transformation is still allowed to operate over multiple steps.
  - https://arxiv.org/abs/1807.03819
- I also reviewed more recent compact-model directions during the research pass (QAT scaling laws, MTP curricula, and PTQ refinements). Those were interesting, but mirrored matrix sharing looked like the strongest idea that was both **novel relative to this repo** and **implementable with the existing banked code path**.
  - QAT scaling law: https://arxiv.org/abs/2505.14302
  - Better & Faster LLMs via MTP: https://arxiv.org/abs/2404.19737
  - Curriculum MTP for SLMs: https://arxiv.org/abs/2505.22757

## What changed versus the chosen base implementation

Base implementation: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. **Mirrored bank sharing**
   - Heavy bank tensors now allocate only `ceil(NUM_LAYERS / 2)` unique slices.
   - For the default 12-layer configuration, the share map is:
     - `[0, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0]`
   - Each virtual layer still keeps its own local controls.

2. **Deeper default virtual stack**
   - `NUM_LAYERS` default changed from `11` to `12`.

3. **Larger default BigramHash table**
   - `BIGRAM_VOCAB_SIZE` default changed from `2048` to `3072` to spend part of the saved artifact budget on stronger local token-pair features.

4. **Value embedding defaults updated for 12 layers**
   - `VE_LAYERS` default is now `10,11`.

5. **Candidate-friendly defaults**
   - `ITERATIONS=9000`
   - `TTT_ENABLED=1`
   - `TTT_FREEZE_BLOCKS=0`
   - default data/tokenizer paths are derived from the candidate script location, so the script can be launched either from this folder or from repo root via its path.

6. **Quantization/export helpers updated**
   - unbank/rebank helpers now operate on the number of **unique shared bank slices**, not the number of virtual layers.

7. **TTT freezing semantics preserved under sharing**
   - if `TTT_FREEZE_BLOCKS > 0`, the script now also masks gradients for any shared bank slice referenced by a frozen layer, so mirrored sharing does not silently bypass the freeze option.

## How to run / evaluate

From this candidate directory:

```bash
cd candidates/202603251119_mirrored-bank-sharing
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The defaults assume the repository dataset layout is present under:

- `../../data/datasets/fineweb10B_sp1024`
- `../../data/tokenizers/fineweb_1024_bpe.model`

Useful ablation toggles:

```bash
# isolate the architectural change without evaluation-time TTT
TTT_ENABLED=0 torchrun --standalone --nproc_per_node=8 train_gpt.py

# compare with a smaller bigram table if the extra local features seem too slow/heavy
BIGRAM_VOCAB_SIZE=2048 torchrun --standalone --nproc_per_node=8 train_gpt.py

# compare against the old depth while keeping sharing machinery
NUM_LAYERS=11 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Main expected risks / tradeoffs

- **Oversharing risk**: mirrored layers may want different QKV/MLP weights even if layer-local scales and norms remain unique.
- **Step-time risk**: 12 virtual layers may cost enough throughput that the extra representational reuse is cancelled by fewer optimizer steps.
- **Interaction risk with XSA / VE / TTT**: all three were tuned on a non-shared bank layout, so their best settings may shift.
- **Quantization distribution shift**: fewer unique bank slices changes the per-slice statistics seen by GPTQ-lite export.

## Validation

Commands run during this workflow:

```bash
python -m compileall candidates/202603251119_mirrored-bank-sharing/train_gpt.py
```

Outcome:

- **Passed**: Python bytecode compilation succeeded.

Attempted extra smoke check:

```bash
python - <<'PY'
import importlib.util
from pathlib import Path
path = Path('candidates/202603251119_mirrored-bank-sharing/train_gpt.py')
spec = importlib.util.spec_from_file_location('candidate_train_gpt', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
PY
```

Outcome:

- **Not feasible in this runner as-is**: the environment is missing the repository runtime dependencies (`numpy`, `torch`, and `sentencepiece`), and network/package install is not available from the shell tool here. Because of that, I validated syntax/compilation only, not full runtime startup.

## Suggested next experiments

1. **TTT disabled** first, to see whether the mirrored-sharing change helps the underlying train/export stack before legal TTT amplifies or masks it.
2. **11 virtual layers with sharing** as a lower-risk control.
3. **Share only MLP banks** or **only attention banks** if full sharing proves too aggressive.
4. If sharing looks promising, combine it with one of the training-only ideas that costs no export bytes, especially **lightweight MTP**.
