---
name: Sync Upstream
description: Merge the latest openai/parameter-golf upstream changes into this fork through an auto-merge PR.
engine:
  id: copilot
  model: gpt-5.4
  args:
    - --reasoning-effort
    - xhigh
on:
  workflow_dispatch:
  schedule: daily on weekdays
permissions:
  contents: read
  pull-requests: read
tools:
  github:
    mode: remote
    toolsets: [default]
  bash: true
checkout:
  fetch-depth: 0
network:
  allowed:
    - defaults
timeout-minutes: 30
safe-outputs:
  create-pull-request:
    title-prefix: "[upstream sync] "
    draft: false
    auto-merge: true
    if-no-changes: warn
---

# Sync Upstream

When this workflow runs, act as a careful repository maintenance agent for this fork.

## Goal

Bring the latest changes from `openai/parameter-golf` into this repository's default branch by preparing a pull request that can auto-merge when checks pass.

## Requirements

1. Work from the checked out repository and detect the local default branch before making changes.
2. Treat `https://github.com/openai/parameter-golf.git` as the canonical upstream remote and `main` as the upstream branch unless repository metadata clearly indicates otherwise.
3. Use a fresh branch named `automation/upstream-sync-YYYYMMDDHHMM` using the current UTC timestamp to the minute.
4. Add the upstream remote if it does not exist, otherwise update its URL to `https://github.com/openai/parameter-golf.git`.
5. Fetch the latest refs needed from both `origin` and `upstream`.
6. Start the sync branch from the current default branch tip, then merge the upstream branch into it with a normal merge commit when changes exist. Do not rewrite history.
7. If the branches are already in sync, stop cleanly without creating a pull request.
8. If a merge conflict occurs, stop and explain the conflict clearly in the final workflow output instead of forcing a resolution.

## Validation

- Run `gh aw compile --strict` after the merge to ensure agentic workflows still compile.
- Run `python -m compileall train_gpt.py train_gpt_mlx.py data` as a lightweight repository check when those paths still exist after the merge.
- Record the validation commands and outcomes in the pull request body.

## Pull request requirements

If the merge succeeds and produces changes:

1. Create a pull request from the sync branch into the repository default branch.
2. Enable auto-merge on that pull request.
3. Use a concise title that states the upstream repository and branch being synchronized.
4. In the PR body include:
   - the upstream repository and branch,
   - the upstream commit SHA merged,
   - the base branch used in this repository,
   - whether the merge created a merge commit or found no changes,
   - the validation commands you ran and their outcomes,
   - any follow-up risk or manual review notes.

## Safety requirements

- Keep changes limited to the upstream synchronization work. Do not make unrelated edits.
- Do not force-push.
- Do not bypass merge conflicts, failing validation, or protected branch expectations.
