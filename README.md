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
