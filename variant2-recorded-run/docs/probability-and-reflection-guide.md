# 单条预测概率与 Reflection 核验

## 真实预测示例

`P_yes` 是模型对该条样本属于 yes 的估计。`P_no = 1 - P_yes`；本次默认规则为 P_yes > 0.5 时预测 yes。

| row_id | 实际 label | LogisticRegression P_yes | RandomForest P_yes | 解释 |
| --- | --- | --- | --- | --- |
| test:41 | yes | 84.60% | 94.36% | 两个模型都预测 yes，均正确。 |
| test:5 | yes | 35.24% | 70.67% | LogisticRegression 预测 no；RandomForest 预测 yes。本条 RandomForest 正确。 |
| test:3 | no | 48.42% | 76.70% | LogisticRegression 预测 no，正确；RandomForest 预测 yes，产生 false positive。 |

row_id 中冒号后的数字是从 0 开始的原始数据行索引；CSV 文件中对应行号为该数字加 2（计入表头）。样例是为解释不同预测情形而选择，不能代替完整 test set 的指标。

例如 test:3：RandomForest P_yes=76.70%，P_no=23.30%，所以预测 yes，但真实 label=no。高概率也可能预测错误。本次没有运行 probability calibration，不能声称所有 76.70% 的预测都具有相同的实测发生率。

LogisticRegression 将转换后的 features 线性组合后通过 sigmoid 得到概率；RandomForest 对各棵树叶节点的类别概率取平均。训练中的 class_weight 会影响结果。仅凭一个最终概率值，不能判断某个 feature 是导致预测的原因。

## 按实际 evidence 手动核验

在 results/metrics.json 找到 models → RandomForest → test → confusion_matrix，数值为 [[8100, 2065], [585, 2583]]。行是真实类别，列是预测类别；两者顺序均为 [no, yes]。

| 实际 / 预测 | no | yes |
| --- | ---: | ---: |
| no | TN=8100 | FP=2065 |
| yes | FN=585 | TP=2583 |

1. 用计算器检查总数：8100+2065+585+2583=13333，对照 test_evaluable_rows。
2. 检查 Recall：2583/(2583+585)=0.8153409091，即 81.53%。
3. 检查 Precision：2583/(2583+2065)=0.5557228916，即 55.57%。
4. 在原本保存在本地的 predictions.csv 中筛选 true_label=yes 且 RandomForest_label=yes，应有 2583 行。此完整逐行文件未放入 GitHub 准备包。

## 将核验转为 Reflection

Human oversight：如实说明你选择 skill、提供 dataset、要求解释预测概率，以及亲自做了哪些检查。不要把阅读 agent 给出的答案描述成独立复算。

Critical evaluation：比较 results/metrics.json 中两个模型的 cv_macro_f1_mean。RandomForest 只高出约 0.000511，而训练更慢、test false positives 更多。你是否认为这足以支持选择它？错误成本未知时，你更重视找出 yes 还是减少 false positives？你的理由比“模型复杂所以更好”更重要。

Trustworthiness：亲自完成上述哪一项，就写哪一项，包含数据来源、算式和结果。尚未检查的项目保持待核验。

请补充：我实际检查了【项目】，计算结果为【结果】；我对【决定】的看法是【意见及理由】；若重跑，我会【改进】。收到你的真实反馈后，才能形成可提交的第一人称 English Reflection。