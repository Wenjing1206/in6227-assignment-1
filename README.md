# IN6227 Assignment 1 · Variant 2

这个仓库保存 Variant 2 的 reusable SKILL、两次**独立**的分类实验记录，以及相应的复现说明。三个目录的用途不同；模型配置、CV 分数和 test 分数不能混为一次实验。

| 目录 | 用途 | 从这里开始 |
| --- | --- | --- |
| [`variant2/`](variant2/README.md) | 最初的可运行实现：AI 编写 `decisions.json`，Python helper 执行 binary/multiclass workflow；有测试和 aggregate results。 | [实现与复现说明](variant2/README.md) · [配套 SKILL](variant2/classification-report-agent/SKILL.md) |
| [`variant2-recorded-run/`](variant2-recorded-run/README.md) | 后来单独保存的 train/test 实验：包含当时的 skill snapshot、代码、CV 明细和 aggregate evidence。 | [运行记录与复现说明](variant2-recorded-run/README.md) · [当次 SKILL](variant2-recorded-run/skill-at-run/SKILL.md) |
| [`variant2/tabular-classification-variant2/`](variant2/tabular-classification-variant2/SKILL.md) | 新整理的 **3–5 个正式 classifier** 工作流说明。此目录目前只发布 `SKILL.md`；对应的新一轮脚本、逐行预测和 PDF 保存在本地，未作为这个仓库中的可复现实验发布。 | [新 SKILL.md](variant2/tabular-classification-variant2/SKILL.md) |

**想直接运行已发布代码**：从 [`variant2/README.md`](variant2/README.md) 开始；如果要复现第二次已记录实验，按 [`variant2-recorded-run/README.md`](variant2-recorded-run/README.md) 操作。**想参考最新的 3–5 模型工作流要求**：阅读 [`variant2/tabular-classification-variant2/SKILL.md`](variant2/tabular-classification-variant2/SKILL.md)。它是工作流指令，不含新的执行脚本。

原始课程数据、逐行 predictions、训练后的模型，以及含个人信息的最终提交 PDF 均未上传。Reflection 需要基于学生实际完成的 Human oversight、Critical evaluation 和 manual check；仓库中的自动验证结果不能代替个人经历。本仓库不代表已向 NTULearn 提交。
