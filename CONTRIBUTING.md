# Team contribution workflow

Read `README.md`, `docs/EVALUATION_PROTOCOL.md`, and `docs/SUBMISSION_FORMAT.md` first.

## Your branch and folder
- Coordinator: `feature/gbfrs` → `gbfrs/`
- Member 2: `feature/fuzzy-rough` → `fuzzy_rough/`
- Member 3: `feature/svm-baseline` → `baseline_svm/`

Create or check out your feature branch, work locally, commit meaningful checkpoints, push your branch, and open a **pull request to main**. Do **not push to main directly**. The coordinator reviews before merging.

## Required handoff
- Fully runnable source code with exact commands and test notes.
- Source citation and algorithm description; reproduce the *actual* published method or clearly identify adaptations.
- Software versions, hardware, random seeds, inner-CV hyperparameter search and chosen values.
- One held-out prediction for each image over all five outer folds (1,875 total) in the **existing offline team-package schema**.
- Per-fold selected original feature names, scores, and measured timing.
- No hand-written/fabricated prediction results or reported accuracies.

## Frozen experimental rules
1. Every experiment starts with 288 numerical columns beginning `original_` or `wavelet-`; no other predictor columns.
2. Use `fold5_kfold` with values 1–5 as the held-out test fold, and patient_id for inner grouped CV.
3. Fit scaling and feature selection *inside* training folds only.
4. Do not optimize on outer-test labels, add features, run image-level random splitting, use PCA, oversample or silently change SVM candidate parameters.
5. Check the exact CSV SHA-256 in `docs/DATASET.md` before running.

## Public repository safety
Do not commit BUSBRA.zip, raw images, the team CSV, patient metadata, private chats, credentials, .env, or large intermediate outputs. Keep model predictions in the privately shared offline comparison folder until publication permissions are checked. Submit safe source code through GitHub PRs.

**Branch protection is not yet enforced by GitHub:** team members must follow this policy until the organization owner enables a ruleset requiring PR review.
