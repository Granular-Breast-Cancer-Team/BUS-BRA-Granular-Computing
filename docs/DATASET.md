# BUS-BRA dataset provenance and immutable input contract

The input file is the **exact verified** `BUSBRA_features_ready.csv` distributed in the team's BUSBRA_Team_Handoff.zip.

- Local path: `inputs/BUSBRA_features_ready.csv` (not committed).
- **SHA-256** `7f22dccf4be1f8920a0e9fbd46ae31ed814f1be138e55dc936b00b76d67324ef`
- **1,875 rows, 295 columns**, of which **288 are numerical predictor columns** (prefixes `original_`, `wavelet-`).
- Remaining seven columns: `image_id`, `patient_id`, `source_case`, `diagnosis`, `pathology`, `fold5_kfold`, `fold10_kfold`. They are **NOT predictors**.
- 1,268 benign image rows (class 0); 607 malignant (class 1). 1,064 patient groups.
- The source `fold5_kfold` is patient-disjoint. **Do not use the source `valid_*` indicators** (not grouped).
- 42 exact redundant numeric columns were documented. They remain in the frozen input; apply only training-only feature-selection decisions.
- PNG scans lack physical spacing. Shape features are not guaranteed to represent physical millimetres.
- No missing/nonfinite numeric features and no extraction failures in the preparation report.

**Checksum on Mac:** `shasum -a 256 inputs/BUSBRA_features_ready.csv`

**Checksum on Windows PowerShell:** `Get-FileHash inputs/BUSBRA_features_ready.csv -Algorithm SHA256`

Only start training if the fingerprint matches. Do not modify the CSV in place.

## Citation and permission
Official source https://zenodo.org/records/8231412 ; Gómez-Flores, Gregorio-Calas, Pereira, *Medical Physics* (2024), DOI https://doi.org/10.1002/mp.16812. The downloaded archive includes `LICENSE.txt`; the full notice was not supplied in our chat. Check that notice and retain attribution before **publicly** distributing source or derived data. This repository intentionally does not publish the prepared CSV at present.
