# IN6227 Assignment 1 · Variant 2
[`SKILL.md`](variant2/SKILL.md) 是 reusable workflow，`scripts/` 是本次 binary 数据的执行实现，`results/` 保存可公开的 aggregate evidence。

这次比较 LogisticRegression、RandomForest、GaussianNB 三个正式 classifier，另用 DummyClassifier 作 baseline；模型根据 train-only CV macro-F1 选择，test 留作最终评估。数据清理、preprocessing、模型取舍和指标的依据见 [项目说明](variant2/README.md)与[汇总结果](variant2/results/metrics.json)。对新 dataset 必须重新判断 target、feature 语义和 split；当前脚本不是对任何 tabular dataset 都可直接运行的通用程序。
