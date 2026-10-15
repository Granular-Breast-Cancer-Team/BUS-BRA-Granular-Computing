## Method
- [ ] Published GBFRS + SVM (coordinator)
- [ ] Conventional fuzzy-rough + SVM (member 2)
- [ ] All-feature SVM (member 3)

## Required reviewer checklist
- [ ] Exactly 288 original feature candidates; no extra engineered inputs
- [ ] Frozen CSV fingerprint matches docs/DATASET.md
- [ ] Same patient-disjoint source outer folds (1..5)
- [ ] Inner patient-grouped 3-fold tuning, using macro-F1
- [ ] Same SVM grid from evaluation_contract.json
- [ ] Scaling/feature selection trained only within training folds
- [ ] Reproducible source code, versions, run command and tested settings
- [ ] Full held-out predictions, selected names, parameters and timings submitted privately
- [ ] Actual experimental results only; no promised scores

## Explanation, results and known limitations

## Coordinator review notes
