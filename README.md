# IN6227 Assignment 1 · Variant 2

**现在请从 [`variant2-standalone/`](variant2-standalone/README.md) 开始。**这里单独放置最新的三模型 Variant 2 项目：`SKILL.md`、执行脚本、依赖和可公开的汇总 evidence。之前的两次实验仍保留为历史记录；各次配置、CV 分数和 test 分数不能混用。

| 目录 | 用途 | 从这里开始 |
| --- | --- | --- |
| [`variant2-standalone/`](variant2-standalone/README.md) | **当前主版本**：三种正式 classifier 加 Dummy baseline；有 SKILL、代码、CV/test 汇总结果和复算记录。 | [项目说明](variant2-standalone/README.md) · [SKILL.md](variant2-standalone/SKILL.md) |
| [`variant2/`](variant2/README.md) | **历史版本 A**：AI 编写 `decisions.json`，Python helper 执行 binary/multiclass workflow；有测试和 aggregate results。 | [原实现与复现说明](variant2/README.md) |
| [`variant2-recorded-run/`](variant2-recorded-run/README.md) | **历史版本 B**：另一轮 train/test 实验，保留当时的 skill snapshot、代码和 aggregate evidence。 | [运行记录与复现说明](variant2-recorded-run/README.md) |

早先上传的 [`variant2/tabular-classification-variant2/SKILL.md`](variant2/tabular-classification-variant2/SKILL.md) 保留原路径，以免已有报告链接失效；当前主版本的同一份工作流和配套脚本请以 `variant2-standalone/` 为准。三模型脚本是这次 binary 数据的实现，新数据仍须重新检查 target、split 和 feature 语义。

原始课程数据、逐行 predictions、训练后的模型，以及含个人信息的最终提交 PDF 均未上传。Reflection 需要基于学生实际完成的 Human oversight、Critical evaluation 和 manual check；仓库中的自动验证结果不能代替个人经历。本仓库不代表已向 NTULearn 提交。
