# Variant 2: Original executable workflow

这是仓库中最初的可运行 Variant 2 实现及其独立实验结果。它不等同于 [`variant2-recorded-run/`](../variant2-recorded-run/README.md) 的第二次记录，也不等同于后来只发布说明的 [3–5 classifier SKILL](tabular-classification-variant2/SKILL.md)。本目录的 SKILL 从 dataset path 出发，由 AI Agent Harness 决定 preprocessing 和模型；Python helper 执行记录下来的配置，不调用 LLM API。

Start with **[SKILL.md](classification-report-agent/SKILL.md)**. The [Chinese walkthrough](docs/SKILL-walkthrough.zh.md) explains its paragraphs, their implementation and the limits of the current helper.

## Workflow and boundaries

Dataset path → profile → AI-authored decisions.json → training-only cross-validation → fixed-model test evaluation → independent verification → generated PDF and visual review.

The helper accepts CSV/TSV, a train/test directory, or ZIP; XLSX/Parquet additionally need openpyxl/pyarrow. It supports binary and multiclass IID classification with configurable Logistic Regression, Random Forest or Extra Trees. A new dataset requires fresh target, leakage, split, metric and model decisions. Grouped, temporal, multilabel or other special structures require the agent to adapt the helper before running them; they are not silently treated as IID data.

The provided assignment train/test split is preserved. Three training rows with missing labels are excluded. One unlabeled test row receives predictions but is excluded from scoring. Imputation and category vocabulary are fitted inside CV folds; Logistic Regression additionally standardizes numerical features. The runner uses most-frequent imputation for categories and median imputation for numerical columns in this run. IQR-flagged observations are retained without domain evidence that they are invalid.

## Reproduce

Run from this `variant2/` directory. The recorded environment used Python 3.14.0; exact installed dependencies are in `requirements-lock.txt`. The broader supported dependency ranges are in `classification-report-agent/requirements.txt`.

```bash
# 中文备注：建立独立环境并安装本次运行的固定依赖。
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt

# 中文备注：先审查新数据，不直接沿用示例的模型决策。
python classification-report-agent/scripts/run.py profile --data /path/to/dataset.zip --out run

# 中文备注：使用 AI 根据 profile 编写并经审阅的 decisions.json。
python classification-report-agent/scripts/run.py run --data /path/to/dataset.zip --config decisions.json --out run
python classification-report-agent/scripts/verify.py --run run
python classification-report-agent/scripts/report.py --run run --config decisions.json --out run/report.pdf

# 中文备注：验证复用和缺失/未知类别等实际行为。
python -m unittest discover -s tests -v
```

Use the supplied `decisions.json` to reproduce this specific assignment experiment. For new datasets the agent must author a new configuration; `SKILL.md` describes the required fields. Fill in student identity, repository and model/interface metadata locally before creating a submission report. A missing value is marked explicitly, not invented. The generated report must be rendered and checked; its generator rejects reports over two pages.

To use this package in an Agent Harness, supply the path to `classification-report-agent/SKILL.md` and the dataset path. Keep its scripts and requirements alongside the markdown. Deterministic reproduction from an existing configuration needs no LLM API key.

## Recorded results

Selection criterion: **macro-F1** in the same three stratified CV folds, seed 42. The test metrics were computed after all candidate settings and the winner were fixed. Test files were available for data-quality profiling, so this is not a claim that the file was never read before final evaluation.

| Model | CV macro-F1 | Test Accuracy | Test macro-F1 | Test yes Recall |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.7536 | 0.7832 | 0.7458 | 0.8415 |
| Random Forest | 0.7719 | 0.8258 | 0.7721 | 0.7156 |
| Majority baseline | Not tuned | 0.7624 | 0.4326 | 0.0000 |

Random Forest was selected by CV macro-F1. Logistic Regression has higher minority-class recall, so Random Forest is not preferable under every possible error-cost objective. Class weighting was used but not compared against an unweighted ablation in this experiment.

- Logistic Regression: balanced class weights; C in {0.1, 1, 10}; selected C=0.1; maximum 2,000 iterations.
- Random Forest: balanced_subsample; 160 trees; maximum depth in {12, None}; minimum leaf size 2; selected unlimited depth.
- All 15 predictors retained, with seven numerical and eight categorical columns. No exact cross-split feature duplicates found.
- Evaluated test size: 13,333. Random Forest confusion matrix: `[[8744, 1421], [901, 2267]]`, rows actual / columns predicted in `[no, yes]` order.

## Evidence and files

`results/metrics.json` contains full-precision scores, candidate/fold CV scores, confusion matrices, software versions and input hashes. `results/training_decisions.json` preserves the configuration whose hash was stored during training; `decisions.json` may later have its report metadata updated without changing the experiment. `results/independent_verification.json` records count-based checks of Accuracy, macro-F1 and Balanced Accuracy from local row-level predictions. Tests cover multiclass/internal holdout, unlabeled test predictions, unseen categories, missing training targets, and saved-model preprocessing behavior.

Course data, row-level predictions, the large trained model and reports containing student identifiers are not committed. They are regenerated locally. The report's Reflection must describe the student's actual oversight and manual checking; automated verification is not a substitute for that personal account. Human Reflection is being completed separately and is not asserted here.
