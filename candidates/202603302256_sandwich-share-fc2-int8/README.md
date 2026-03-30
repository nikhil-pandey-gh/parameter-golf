# Candidate: Sandwich Shared Cores + FC2 Int8

## Hypothesis

The strongest recent training-only stack in this repository is already close to the 16 MB artifact limit, so a good next move is not "more layers at any cost" but **artifact-aware parameter sharing that preserves layer-specific behavior**.

This candidate shares only selected transformer **attention+MLP cores** across layers, while keeping each layer's own RMSNorms, residual mix, learned scales, skip structure, and value-embedding routing. The saved artifact budget is then spent on a more conservative quantization choice for the most QAT-sensitive weights: **`mlp.proj.weight` (the FC2/down-projection) stays int8 instead of int6**.

## Why this is promising for this repository

The repo's winning trend is clear: the frontier moved by combining deeper/better architectures with increasingly careful compression. Recent records improved BPB mostly by reducing quantization damage rather than by adding large new subsystems.

This candidate targets exactly that regime:

- it keeps the proven 11-layer U-Net/XSA/EMA/Partial-RoPE/LN-scale family;
- it avoids the obvious dead end of naive recurrence that costs training steps in fixed wallclock;
- it uses sharing only where it buys artifact budget, then spends those bits on the MLP down-projection that recent QAT work identifies as a major bottleneck.

## Prior repository work that influenced this candidate

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md`
  - Chosen as the main implementation base because it is the strongest clean training-time stack before legal TTT.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`
  - Contributed the `LeakyReLU(0.5)^2` MLP activation, which was the strongest recent low-cost training-side gain.
- `records/track_10min_16mb/2026-03-19_MixedQuant_Int6Int8_SlidingWindow/README.md`
  - Reinforced the repo-wide lesson that **mixed precision beats uniform low-bit quantization**, especially for sensitive weights.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md`
  - Confirmed Partial RoPE and LN scaling remain part of the best stack.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md`
  - Important negative result: naive layer recurrence hurt because it traded away too many training steps. This candidate avoids that by **sharing parameters across the same 11 effective layers** instead of adding extra recurrent depth.

There were no prior `candidates/` directories in the repository when this candidate was created.

## External research that informed this candidate

- **Subformer** (`arXiv:2101.00234`)
  - Showed that **sandwich-style parameter sharing** can outperform naive cross-layer sharing in generative transformers.
- **Basis Sharing** (`arXiv:2410.03765`)
  - Motivated the idea that cross-layer sharing is especially attractive in the high-compression regime.
- **ALBERT** (`arXiv:1909.11942`)
  - Classic evidence that cross-layer sharing can reduce parameter count without collapsing model utility.
- **Scaling Law for Quantization-Aware Training** (`arXiv:2505.14302`)
  - Specifically highlighted FC2/down-projection sensitivity and supported spending recovered bits on a safer mixed-precision choice there.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

This candidate adds:

1. **Sandwich-style shared block cores**
   - Default shared groups: `1,4;2,3;7,10;8,9`
   - Only the heavy **attention + MLP core** is shared.
   - Each layer keeps its own norms, residual mix, learned residual scales, DTG gate, and position in the U-Net stack.
   - Groups are constrained not to cross the XSA boundary.

2. **Artifact deduplication for shared weights**
   - The export quantizer now detects aliased tensors and stores only one quantized copy.
   - This is the key change that turns in-memory sharing into actual artifact-size savings.

3. **Targeted int8 protection for FC2**
   - By default, any weight matching `mlp.proj.weight` is quantized with the int8 path instead of int6.
   - The rest of the strong mixed-quantization stack is preserved.

4. **LeakyReLU(0.5)^2 MLP**
   - Carried forward from the latest best record so the candidate does not regress to the older ReLU^2 choice.

5. **Practical execution fixes**
   - Default data/tokenizer paths are resolved from the repository root, so `train_gpt.py` can be launched from inside this candidate directory.
   - FlashAttention is optional at import time; the script falls back to PyTorch SDPA when FlashAttention is unavailable.
   - A `SMOKE_TEST=1` path is included for local CPU-only sanity checks on machines that already have PyTorch installed; the full training path still requires `numpy` and `sentencepiece`.

## How to run

From this candidate directory:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful knobs:

```bash
MLP_NEGATIVE_SLOPE=0.5
FORCE_INT8_PATTERNS=mlp.proj.weight
SHARED_LAYER_GROUPS="1,4;2,3;7,10;8,9"
```

If you change `NUM_LAYERS` or `XSA_LAST_N`, you should usually also update `SHARED_LAYER_GROUPS` so the groups still make architectural sense. To **disable** the built-in default sharing pattern entirely, set:

```bash
SHARED_LAYER_GROUPS=
```

For local non-GPU sanity checking on a machine with PyTorch installed:

```bash
SMOKE_TEST=1 python train_gpt.py
```

## Main expected risks and tradeoffs

- **Under-sharing vs over-sharing**
  - Too little sharing may not buy enough artifact budget to matter.
  - Too much sharing can collapse layer specialization and undo the gains.

- **Compression savings may be best used elsewhere**
  - FC2 int8 is motivated by recent QAT evidence, but the recovered bytes might be better spent on another sensitive tensor family.

- **Training dynamics can change even when compute depth is unchanged**
  - Shared cores receive gradients from multiple layer positions, which may regularize or may create optimization interference.

- **Default sharing groups are tuned to the current 11L/XSA4 layout**
  - They are intentionally conservative and should be revisited if the stack changes materially.

## Validation

Commands run for this candidate in the workflow:

```bash
python -m compileall candidates/202603302256_sandwich-share-fc2-int8/train_gpt.py
```

Outcome:

- **Passed** on this workflow runner.

Attempted smoke validation:

```bash
SMOKE_TEST=1 python candidates/202603302256_sandwich-share-fc2-int8/train_gpt.py
```

Outcome:

- The workflow runner did **not** have a usable local PyTorch install.
- An isolated venv was created, but fetching a CPU PyTorch wheel was blocked by the runner's network restrictions.
- As a result, a real CPU smoke execution was **not feasible on this runner**.
- The smoke path remains in the script for local validation on a machine with PyTorch available.
