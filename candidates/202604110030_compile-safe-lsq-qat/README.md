## Compile-Safe LSQ-Style Late QAT

### Hypothesis

The strongest non-TTT stack in this repository is already close to the current frontier, but its remaining gap is still dominated by **post-training int6 export error** rather than raw model quality. A compile-safe, export-aligned **late quantization-aware training** path should reduce that gap more reliably than another architecture-wide rewrite.

This candidate starts from the 11-layer EMA + GPTQ-lite + Partial RoPE + LN scale stack and replaces the fragile late-QAT switch with a **tensor-gated LSQ-style weight fake-quant path** that survives `torch.compile(fullgraph=True)` and reuses the learned per-row scales at export time.

### Why this is promising here

Repository evidence points in the same direction:

1. `records/track_10min_16mb/2026-03-19_WarmdownQuantization/README.md` explicitly argues that quantization error is the dominant bottleneck once the model is reasonably trained.
2. `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md` notes that its late-QAT branch never actually activated because `torch.compile` constant-folded the boolean switch away.
3. The recommended fork base, `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`, still uses the same class-level boolean QAT pattern, so it is a natural place to test a real fix.
4. External research supports learned quantizer scales over fixed heuristics: **LSQ** (arXiv:1902.08153), **LSQ+** (arXiv:2004.09576), and the 2025 QAT scaling-law study (arXiv:2505.14302) all point to quantizer configuration as a major driver of low-bit performance.

I considered shared-depth recurrence from the external research pass, but the repository's own exploratory history already flags recurrence as a bad wallclock trade under this challenge.

### Prior records that influenced this candidate

- **Primary base:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- **Bug signal / motivation:** `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
- **Quantization bottleneck framing:** `records/track_10min_16mb/2026-03-19_WarmdownQuantization/`
- **Current ceiling reference:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`

There were no prior `candidates/` directories in the repository when this candidate was created.

### External research

- **LSQ** — *Learned Step Size Quantization* (arXiv:1902.08153)
- **LSQ+** — asymmetric learned quantization with more stable initialization (arXiv:2004.09576)
- **Unified QAT scaling law** — quantization error still matters even as model/data scale increase, especially for weight quantization (arXiv:2505.14302)

### What changed vs. the chosen base

1. **Compile-safe late QAT**
   - Replaced the old `CastedLinear._qat_enabled` class boolean with a **tensor-valued `qat_mix` gate** on each linear.
   - The fake-quant path is now arithmetic, not Python control flow, so it is compatible with `torch.compile(fullgraph=True)`.

2. **LSQ-style learned per-row scales**
   - Each `CastedLinear` now carries a learned `qat_scale` vector.
   - During training, weights are fake-quantized with an STE round pass and a gradient-scaled learned step size.

3. **Late ramp instead of hard switch**
   - `late_qat_threshold` now controls a smooth ramp from 0 to 1 as the LR multiplier decays below the threshold.
   - `QAT_ENABLED=1` still forces full-strength QAT from step 0 if you want an ablation.

4. **Export alignment**
   - If late QAT actually engaged during training, export reuses the learned per-row scales for int6 tensors instead of discarding them and reverting entirely to the GPTQ-lite percentile search.
   - The learned `qat_scale` tensors are treated as training-only state and are stripped from the exported artifact.

5. **Candidate-directory execution**
   - Default `DATA_PATH` and `TOKENIZER_PATH` now resolve from the script location, so this file can be launched directly from inside `candidates/202604110030_compile-safe-lsq-qat/`.

### How to run

From this candidate directory:

```bash
cd candidates/202604110030_compile-safe-lsq-qat
RUN_ID=compile_safe_lsq_qat \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful ablations:

```bash
# Force QAT from step 0
QAT_ENABLED=1 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Keep the new compile-safe training path but export with GPTQ-lite scales only
QAT_EXPORT_SCALES=0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Evaluation notes

This keeps the base script's evaluation flow:

- EMA application before export/eval
- int6 roundtrip evaluation
- stride-64 sliding-window evaluation

The main measurement to watch is whether the **post-export int6 gap** shrinks relative to the base stack.

### Main risks and tradeoffs

1. **Training overhead:** the compile-safe fake-quant path does extra per-row scale work every training step.
2. **Short-run instability:** learned scales may need more late-phase steps than a 600s run gives them.
3. **Quantizer mismatch risk:** LSQ-style learned scales may improve roundtrip quality but still interact differently with zstd compression than the base GPTQ-lite heuristic.
4. **Small-gain possibility:** if the remaining gap is mostly architectural rather than quantization-driven on this stack, the effect may be modest.

### Validation

Commands run in this environment:

```bash
python -m compileall candidates/202604110030_compile-safe-lsq-qat/train_gpt.py
```

Outcome: **passed**.

Attempted additional smoke test:

```bash
python - <<'PY'
# import candidate module, stub FlashAttention, build a tiny CPU model, run
# forward/backward + int6 roundtrip smoke
PY
```

Outcome: **blocked by environment** (`ModuleNotFoundError: No module named 'torch'`), so no runtime smoke test was possible here without adding new infrastructure.
