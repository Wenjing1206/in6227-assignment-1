# IN6227 Assignment 1

## Variant 2: reusable AI classification SKILL

The reusable SKILL and its separately executed experiment are available in [variant2/](variant2/README.md). Start with [SKILL.md](variant2/classification-report-agent/SKILL.md) for the agent workflow, [the Chinese walkthrough](variant2/docs/SKILL-walkthrough.zh.md) for a paragraph-by-paragraph explanation, and [the saved metrics](variant2/results/metrics.json) for evidence. Variant 2 selects by CV macro-F1; the original Variant 1 below selects by Average Precision. Their configurations and results must not be mixed.

## Variant 1: original implementation

A reproducible comparison of **Logistic Regression** and **Random Forest** on the supplied binary classification dataset. The workflow covers training-data exploration, missing values, categorical encoding, hyperparameter selection, independent test evaluation, and verification of the reported metrics.

## Reproduce the experiment

Use Python 3.12. Extract the course-provided `dataset.zip` and put `train.csv` and `test.csv` in `data/`. The original dataset is not distributed in this public repository.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python src/run_experiment.py --data-dir data --output-dir results --jobs 2
python src/verify_results.py
```

The final verification command expects the standard `results/` directory. Runtime depends on the computer; this run took about two minutes for the Random Forest search, excluding setup and verification. No GPU or API key is needed. The code has Chinese comments explaining the decisions.

## Experimental design

- Keep the supplied train/test split. Remove three training rows and one test row with missing labels; never impute the target. There are 31,109 labelled training rows and 13,333 labelled test rows.
- Use all 15 features: seven numerical and eight categorical. Retain extreme values because the dataset has no domain validity rules; keep literal `Unknown` values as categories.
- Fit median numerical imputation, mode categorical imputation and OneHotEncoder **inside** each training fold. Apply RobustScaler to numerical features for Logistic Regression; leave them unscaled for Random Forest. Previously unseen categories map to all-zero encoded values. There are 82 encoded features after refitting on the full training set.
- Use identical three-fold shuffled StratifiedKFold splits for both models; seed 42. Select hyperparameters using mean validation **Average Precision (AP)**, with `yes` as the positive class. AP is not trapezoidal PR-AUC.
- Logistic Regression: L2 regularization with `lbfgs`; C in {0.1, 1, 10}; class weight in {None, balanced}; tolerance 1e-4 and at most 2,000 iterations. Selected C=10 with no class weighting; final fit converged in 146 iterations.
- Random Forest: 300 trees, Gini criterion, bootstrap sampling, sqrt feature subsampling; maximum depth in {12, None}; minimum leaf size in {1, 5}; class weight in {None, balanced}. Selected unlimited depth, minimum leaf size 5 and no class weighting. Fit stops after the fixed number of trees; individual trees obey their split/leaf constraints.
- There are 6 and 8 candidate configurations respectively, giving 42 CV fits and two full-training refits. Class weighting is evaluated rather than assumed beneficial. No resampling, feature selection, threshold tuning or probability calibration is performed.
- Record model selection in `selection_before_test.json` before loading the test CSV. Evaluate the two fixed models at probability threshold 0.5 and perform no changes based on test results.
- Compute a paired, stratified test-row bootstrap (1,000 resamples, seed 42) for the AP difference. Its interval is conditional on these fitted models and assumes independent rows; it does not capture training, split, or distribution-shift uncertainty.

## Results

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| CV AP mean | 0.7086 | 0.7136 |
| CV AP standard deviation | 0.0109 | 0.0134 |
| Test AP | 0.7023 | 0.7084 |
| Test ROC-AUC | 0.8887 | 0.8900 |
| Test Accuracy | 0.8401 | 0.8388 |
| Test Balanced Accuracy | 0.7477 | 0.7440 |
| Test Precision | 0.7003 | 0.6997 |
| Test Recall | 0.5717 | 0.5634 |
| Test F1 | 0.6295 | 0.6242 |

The majority-class baseline has Accuracy 0.7624, Balanced Accuracy 0.5 and positive-class F1 0. Random Forest was selected by CV AP. Its test AP advantage is 0.0061, with a 95% paired bootstrap percentile interval of [-0.0015, 0.0128]. This does not establish a clear ranking advantage on this test sample. Logistic Regression has slightly better F1 and Recall at the prespecified threshold. The ranking criterion and the fixed-threshold decision criterion measure different properties.

Confusion matrices have actual classes in rows and predicted classes in columns, in the order `[no, yes]`:

- Logistic Regression: `[[9390, 775], [1357, 1811]]`
- Random Forest: `[[9399, 766], [1383, 1785]]`

The dataset provides no provenance, feature timing or entity identifiers, so this experiment cannot establish causal effects or deployment validity. Exact feature-row overlap between train and labelled test data was checked and was zero. Entity-level leakage cannot be ruled out from this schema alone.

## Files and verification

- `src/run_experiment.py`: complete experiment and plotting.
- `src/verify_results.py`: recomputes all seven reported test metrics directly from row-level predictions, using counting, rank sums, and grouped precision/recall increments rather than sklearn metric functions.
- `tests/test_data_integrity.py`: guards against invalid labels, target imputation, held-out preprocessing contamination and unseen-category failures.
- `results/metrics.json`: full-precision results, sample counts, selected parameters, package versions and input SHA-256 values.
- `results/selection_before_test.json`: selection record created before test loading.
- `results/cv_*.csv`: scores for every candidate and fold.
- `results/eda_train.json`, `numeric_summary.csv`, `numeric_correlations.csv`: training EDA evidence.
- `results/precision_recall.png`, `confusion_matrices.png`: figures computed from predictions.
- `results/verification.json`: independent numeric verification outcome.

Raw course data, row-level test predictions and the report containing student identifiers are not committed. Running the experiment regenerates `results/test_predictions.csv` locally. The same dataset and pinned environment should reproduce the results to numerical precision, although timings and tiny floating-point differences may vary across machines.

## References

- [LogisticRegression documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [RandomForestClassifier documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Average Precision documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
- [Data leakage and Pipeline guidance](https://scikit-learn.org/stable/common_pitfalls.html)

## Authorship and assistance

Prepared with AI assistance for implementation and report drafting. All reported results were produced by executing the code, and the saved predictions were checked with the independent verification script. This statement does not imply a student manual review that has not been performed.
