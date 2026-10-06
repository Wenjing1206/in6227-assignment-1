# IN6227 Assignment 1 — Variant 2

A reusable AI SKILL for end-to-end tabular classification and automated report generation, with recorded decisions, cross-validation, independent metric verification, and reproducibility tests.

## Start here

- **[SKILL.md](variant2/classification-report-agent/SKILL.md)** — instructions for the AI Agent Harness.
- **[Implementation and reproduction guide](variant2/README.md)** — setup, workflow, results and limitations.
- **[逐段中文讲解](variant2/docs/SKILL-walkthrough.zh.md)** — how each SKILL paragraph works.
- **[Experiment decisions](variant2/decisions.json)** and **[verified results](variant2/results/metrics.json)** — the rationale and numerical evidence for this run.

The AI selects and justifies preprocessing and models; Python executes the recorded configuration. The supplied experiment selects Random Forest by training-only CV macro-F1. Its test macro-F1 is 0.7721, while Logistic Regression has higher minority-class recall. Model preference depends on the evaluation objective.

## Reproduce

Follow the commands in [variant2/README.md](variant2/README.md), starting from the `variant2/` directory. The recorded environment uses Python 3.14.0 and the included dependency lockfile. Supply the course dataset locally.

Raw course data, row-level predictions, the trained model and student submission reports are not included in the repository. The report and Reflection are completed locally using verified results and the student's actual review.

## Additional recorded run

A separately recorded experiment and the current reusable skill are available in **[variant2-recorded-run/](variant2-recorded-run/README.md)**. This directory preserves its own code, dependency versions, CV results and the skill snapshot used for that run. It does not replace the original `variant2/` implementation or its results.

- [Current SKILL.md](variant2-recorded-run/SKILL.md) and [skill snapshot used for this run](variant2-recorded-run/skill-at-run/SKILL.md).
- [Prediction probability examples and Reflection verification guide](variant2-recorded-run/docs/probability-and-reflection-guide.md).
- [Recorded aggregate metrics](variant2-recorded-run/results/metrics.json): test macro-F1 is 0.7619 for LogisticRegression and 0.7602 for RandomForest. RandomForest was selected before test scoring by a small training-CV advantage.

These values belong to the additional run and must not be combined with the original experiment's scores or configuration. Reproduction commands for this run start from `variant2-recorded-run/`.
