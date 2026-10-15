\# All-Feature SVM Baseline



\## Purpose



This baseline evaluates an SVM using all 288 prepared BUS-BRA radiomic features. It provides a reference for comparison with conventional fuzzy-rough and granular-ball fuzzy-rough feature selection. No feature selection is performed.



\## Data



The frozen input contains 1,875 image records from 1,064 patient groups: 1,268 benign images and 607 malignant images. Diagnosis is encoded as 0 for benign and 1 for malignant.



Only the 288 names in `inputs/allowed\_feature\_columns.txt` are predictors. Identifiers, diagnosis, pathology, and fold columns are excluded.



Image cleaning and radiomic feature extraction were completed before this baseline experiment.



\## Evaluation Method



The experiment uses five predefined, patient-separated outer folds. Each image is tested once, and images from one patient remain together.



Within each outer-training portion, three-fold `StratifiedGroupKFold` selects SVM settings using mean macro-F1. Inner splitting uses `shuffle=True` and `random\_state=42 + outer\_fold`.



A pipeline fits `StandardScaler` and `SVC` within each training partition. Validation and test data do not influence fitted scaling values.



The shared grid contains 32 candidates:



\- Kernel: linear or RBF.

\- C: 0.1, 1, 10, or 100.

\- RBF gamma: scale, 0.01, or 0.1.

\- Class weights: None or balanced.



The winning pipeline is refitted on the complete outer-training portion before predicting the outer-test images.



\## Results



| Outer fold | Accuracy | Balanced accuracy | Macro-F1 | Malignant sensitivity | Benign specificity |

|---|---:|---:|---:|---:|---:|

| 1 | 0.8883 | 0.8896 | 0.8765 | 0.8934 | 0.8858 |

| 2 | 0.9169 | 0.9073 | 0.9056 | 0.8800 | 0.9346 |

| 3 | 0.8579 | 0.8089 | 0.8274 | 0.6667 | 0.9512 |

| 4 | 0.8110 | 0.7925 | 0.7880 | 0.7395 | 0.8455 |

| 5 | 0.8512 | 0.8445 | 0.8331 | 0.8264 | 0.8626 |

| Mean | 0.8650 | 0.8486 | 0.8461 | 0.8012 | 0.8960 |

| Sample standard deviation | 0.0400 | 0.0497 | 0.0457 | 0.0965 | 0.0456 |



These are image-level scores with patients kept separate. Means give each outer fold equal weight.



All folds selected a linear kernel with C = 0.1. Fold 3 selected equal class weights; the other folds selected balanced weights.



Recorded total fold time summed to approximately 575 seconds on the machine used. Runtime depends on hardware and system load.



\## Limitations



Malignant sensitivity varied from 66.67% to 89.34% across folds. Overall accuracy therefore does not fully describe malignant-case detection.



Fold standard deviations describe variation between folds; they are not confidence intervals. Results have not been validated on an independent external dataset.



\## Reproduction



Run commands from the project root using Python 3.11.9.



Install dependencies:



```bat

python -m pip install -r baseline\_svm\\requirements
