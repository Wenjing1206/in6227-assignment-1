---
name: tabular-classification-variant2
description: Inspect a tabular classification dataset, build leakage-safe sklearn Pipelines, compare 3-5 candidate classifiers, select by train-only validation, and produce a verified PDF report of at most two pages. Use for IN6227 Assignment 1 Variant 2 or a comparable supervised classification task.
---

# Tabular classification workflow

接收 dataset path，可选 target、positive label、output directory、student metadata。按 dataset inspection → target/feature detection → data-quality diagnosis → preprocessing decision → model comparison → evaluation → PDF → review 执行。与用户用中文沟通，专业术语保留英文；生成 Python code 时写中文备注。原始文件和项目 `sources/` 只读。

## 判断与边界

- 先查文件格式、行列、字段类型、target 候选、类别分布、missing、duplicate、class imbalance；保存可核对的 profile。用户指定的 target 优先。自动推断只有唯一且明确的 `target`/`label`/`class` 候选时才成立；歧义必须询问，不通过“最后一列”猜测。
- 数值型 target 的类别语义不明时先确认。缺失 target 行不可用于 supervised training，记录排除数量。重复观测、outlier、高 cardinality 和 `Unknown`/`Other` 不自动删除。检查 ID、时间、群组和可能的 post-outcome feature；预测时可用性无证据时写出限制。
- 使用已有的 train/test split；无 split 时根据 group/time 语义选安全划分，否则用 stratified split。只在 training 内以 CV 选 hyperparameters 和 best model；test 保持冻结，只在选型后评估一次。所有从数据学习的 imputation、encoding、scaling 和 resampling 放在 `Pipeline` 中，CV 每 fold 重新 fit。
- 按 profile 决定 preprocessing：numeric 可用 median imputation；nominal category 可用显式 missing category + `OneHotEncoder(handle_unknown='ignore')`；需 scaling 的模型只对 numeric scaling。记录替代方案和取舍。
- 比较 3–5 个正式 classifier，另设 `DummyClassifier` baseline。候选应有不同 inductive bias，且用相同 CV folds、primary metric 和可比的预算。class imbalance 时不能只看 accuracy；binary 任务默认优先 macro-F1，并同时报告 balanced accuracy、per-class precision/recall/F1、confusion matrix，概率可用时补充 ROC-AUC 和 average precision。错误成本已知时调整 primary metric。
- 在读取 test metrics 前保存 best-model 选择与 CV evidence。保存每行 test prediction、score、row ID，以及 split、参数、seed、依赖版本、warnings 和数据 hash。若 CV 差异很小，说明 ranking 不稳定，不把最小差距当显著优势。

## 报告与人工复核

- PDF 主报告最多两页，内容包括数据与目标、质量问题与处理理由、`Pipeline`、3–5 个模型与 tuning、CV/test 指标、误差分析、局限和复现信息。所有数字从本次 evidence 读取。渲染每页检查字体、溢出、表格、页数和关键数值，并复算 confusion matrix 与 metrics。
- 记录 agent 实际查看的 intermediate outputs 及具体检查结果。自动复算和 agent 检查不能冒充学生本人的 manual verification。
- 课程 Reflection 覆盖 Human oversight、Critical evaluation、Trustworthiness。只写实际发生的用户决策和用户亲自核验；尚未取得这些信息时写可提交前补全的 draft，并给出一个具体可复算的检查项，不虚构第一人称经历。
- 只有用户明确要求时才上传 GitHub；发布前检查仓库和范围。上传范围以用户本次指令为准。不要上传原始数据、逐行 predictions 或私人信息，除非用户明确要求。

## IN6227 提交格式

最终交付为**单个 PDF**：主报告不超过两页，短 Reflection 放在后续页且不计入主报告页数。第一页顶部必须有 matric number、full name、`IN6227-Assignment-1`、`Variant-2`。主报告应标明实际使用的 LLM model name/version、LLM interface 与可确认的 interface version，并附上 GitHub 中此 `SKILL.md` 的链接。缺失的身份、版本或个人 Reflection 要标为待补，不能把 draft 当成最终提交版。

本目录附带的 Python scripts 是这次 binary 数据的参考实现。面对 multiclass、group/time 依赖、特殊缺失编码或其他新数据结构，先由 AI 根据 inspection 改写 candidate models、split、Pipeline 与 metrics；不能直接复用本次固定的 yes/no 配置并宣称已泛化。

执行时将脚本和结果保存在独立项目目录中；不得把一次数据集的固定字段、类别、模型结果写成通用规则。
