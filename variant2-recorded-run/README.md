# IN6227 Variant 2 — Reusable Classification Skill

A reusable Markdown skill for tabular classification and automated, evidence-grounded reports. The workflow inspects a dataset, validates its target, selects preprocessing and contrasting classifiers, prevents train/test leakage, evaluates results, and generates a PDF report of at most two pages by default.

本仓库准备材料对应 IN6227 Assignment 1 Variant 2。`SKILL.md` 是可用于不同 tabular classification datasets 的工作流；`scripts/` 是该 skill 为本次给定 train/test ZIP 生成的实验实现，不能把该次实现当成支持所有数据格式的通用 CLI。

## Skill and workflow

Use `SKILL.md` in a compatible agent environment and provide an actual dataset path. For example:

```text
使用 $tabular-classification-report
输入 dataset_path 为实际的数据文件路径。
按 IN6227 Variant 2 完成 classification workflow 与报告。
```

The skill requires target validation where semantics are ambiguous, fold-local preprocessing, justified model selection, comparable evaluation, and traceable report metrics. Human Reflection must be based on the student's actual actions and checks.

The root skill is the current installed revision. The recorded run used the earlier snapshot in `skill-at-run/SKILL.md`. Both file hashes are in `docs/provenance.json`; no new experiment was run merely to prepare this repository.

## Recorded experiment

The supplied ZIP contains `dataset/train.csv` and `dataset/test.csv`. The run retained this split and treated `label` as the target, with `yes` as the positive class. After excluding four missing targets, there were 31,109 training and 13,333 test cases. Seven numeric and eight categorical predictors were processed inside Pipelines. Two classifiers each received four parameter candidates and identical three-fold stratified CV; DummyClassifier was a separate baseline.

| Model | CV macro-F1 | Test macro-F1 | Test accuracy | Yes precision | Yes recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| DummyClassifier | 0.4318 | 0.4326 | 0.7624 | Undefined* | 0.0000 |
| LogisticRegression | 0.7616 | 0.7619 | 0.8387 | 0.6965 | 0.5694 |
| RandomForest | 0.7621 | 0.7602 | 0.8012 | 0.5557 | 0.8153 |

*Dummy predicts no positives; the stored classification report uses zero_division=0. The probability summary preserves the undefined precision as null.

RandomForest was selected by CV before test scoring, but the CV difference was only about 0.0005. LogisticRegression had slightly higher test macro-F1. RandomForest recovered more positives at the cost of more false positives; these results do not establish a clear winner. No threshold tuning or probability calibration was performed. Missing domain documentation limits conclusions about feature availability and semantic leakage.

## Reproduce the recorded experiment

Run the commands below from `variant2-recorded-run/`. Use Python 3.12 and place the supplied dataset archive at `data/dataset.zip`. The original dataset is not distributed in this preparation package.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/run_analysis.py --dataset data/dataset.zip --output runs/reproduction --target label --positive yes
python scripts/verify_results.py --run-dir runs/reproduction
python scripts/render_report.py --run-dir runs/reproduction
```

Choose a new output directory for each experiment. The analysis refuses to overwrite an existing metrics file. Report metadata may be supplied with `--name`, `--matric`, `--repository`, and `--llm-version`; missing values remain explicitly marked. Re-rendered PDFs must be visually reviewed.

`results/` contains aggregate evidence from the completed run. The 52 checks recorded in `results/verification.json` were performed against the original local row-level predictions. Those predictions are not included here; rerunning the experiment generates new local predictions and allows all checks to be repeated. Results may vary slightly across environments; the exact package versions are recorded in `requirements.txt` and the run manifest.

## Files

- `SKILL.md`: current reusable skill.
- `skill-at-run/SKILL.md`: snapshot used for the recorded experiment.
- `scripts/`: experiment, independent metric verification, and report rendering.
- `results/`: aggregate metrics, all CV candidates, data profile, warnings and verification record.
- `docs/probability-and-reflection-guide.md`: three real prediction examples and a manual-check guide.
- `docs/provenance.json`: skill hashes and explanation of the portable manifest.

The repository copy of `run_manifest.json` replaces the local absolute dataset path with `data/dataset.zip` and omits host-specific command arguments. Original local evidence is unchanged. The SHA-256 of the source archive is retained for matching the input.

## Submission status

The local two-page report has been generated and visually checked, but it is not included as a finalized submission here. Student name, matric number, repository URL, exact LLM model/version, and genuine personal Reflection still need completion. This directory is published as a separately recorded run in this repository. It has not been submitted to NTULearn.
