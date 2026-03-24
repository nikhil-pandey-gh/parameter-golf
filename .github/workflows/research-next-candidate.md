---
name: Research Next Candidate
description: Research and implement the next parameter-golf candidate in a new timestamped candidates folder.
on:
  workflow_dispatch:
permissions:
  contents: read
  issues: read
  pull-requests: read
engine: codex
tools:
  github:
    mode: remote
    toolsets: [default]
  bash: true
  edit:
  web-fetch:
  web-search:
network:
  allowed:
    - defaults
    - python
timeout-minutes: 60
safe-outputs:
  create-pull-request:
    title-prefix: "[agentic candidate] "
    draft: true
    if-no-changes: error
---

# Research Next Candidate

When this workflow runs, act as a strong ML systems researcher and coding agent for this repository.

## Goal

Study the current baseline, all previous record submissions, and all previous candidate iterations. Then do deep external research for the strongest next idea that fits this repository's constraints and implement it in a new directory named `candidates/YYYYMMDDHHMM_<short-slug>` using the current UTC timestamp to the minute, for example `candidates/202603240915_partial-rope-refresh`.

## Required repository review

1. Read `train_gpt.py` in the repository root.
2. Review every relevant prior experiment under `records/`, especially each run's `README.md`, `train_gpt.py`, and any logs or metadata that reveal what helped or failed.
3. Review every prior experiment under `candidates/` if that directory already exists.
4. Extract the main design patterns, winning trends, and dead ends before choosing a new idea.

## External research requirements

- Do deep research on the latest ideas that could help tiny language models under strict parameter and artifact limits.
- Focus on ideas relevant to this challenge, including quantization, Quantization-Aware Training (QAT) variants, compression-aware training, parameter sharing, recurrent or depth-reuse designs, small-model attention alternatives, tokenizer or embedding efficiency, optimizer improvements, evaluation-aware techniques, and other compact-model tricks.
- Prefer ideas that can plausibly improve validation bits-per-byte under the repository's 16MB artifact budget and 10-minute training goal.
- Prefer ideas that can be implemented by adapting the repository's current code rather than introducing broad new infrastructure.
- Avoid repeating an existing record or candidate unless you have a clear new twist and explain that twist explicitly.

## Implementation requirements

1. Choose the single best next candidate based on both repository evidence and external research.
2. Create `candidates/YYYYMMDDHHMM_<short-slug>/`. Create the `candidates/` directory first if it does not exist.
3. Add a `README.md` in that candidate directory that covers:
   - the hypothesis,
   - why it is promising for this repository,
   - which records or prior candidates influenced it,
   - which external research informed it,
   - what changed versus the chosen base implementation,
   - how to run or evaluate it,
   - the main expected risks or tradeoffs.
4. Add a self-contained `train_gpt.py` for the candidate that can be run from the candidate directory.
5. Only add extra files when they are genuinely needed for the candidate to run.
6. Do not edit or delete existing record folders.
7. Do not replace the root `train_gpt.py` unless absolutely necessary. Prefer copying and adapting code into the new candidate folder.
8. Keep the implementation precise and minimal while still being complete.

## Validation

- Run lightweight validation that already fits this repository, such as `python -m compileall candidates/YYYYMMDDHHMM_<short-slug>/train_gpt.py train_gpt.py`, plus any other existing low-cost checks that do not require new infrastructure.
- If a safe smoke check is possible without extra infrastructure, run it.
- Record the validation commands and outcomes in the new candidate `README.md`.

## Pull request output

Create a draft pull request containing the new candidate files.

In the pull request description, include:

- the chosen idea,
- why it differs from the existing records and candidates,
- the most relevant research that motivated it,
- what files were added,
- what validation you ran,
- the biggest uncertainties and suggested next experiments.
