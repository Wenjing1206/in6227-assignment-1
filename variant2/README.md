# IN6227 Assignment 1 · Variant 2

这是单独整理的 Variant 2 三模型版本。`SKILL.md` 描述从 dataset inspection 到两页 PDF 的工作流；`scripts/` 保存本次 binary train/test 数据使用的具体实现；`results/` 只保留可公开的汇总 evidence。原始数据、逐行 predictions、个人 Reflection 和含姓名的 draft PDF 不在此目录。

## 从哪里开始

| 文件 | 作用 |
| --- | --- |
| [`SKILL.md`](SKILL.md) | AI workflow 与质量边界；要求 3–5 个正式 classifier，另有 Dummy baseline。 |
| [`scripts/inspect_dataset.py`](scripts/inspect_dataset.py) | 读取数据并输出 target、feature types、missing、duplicates、imbalance 等 profile。 |
| [`scripts/run_workflow.py`](scripts/run_workflow.py) | 在 sklearn `Pipeline` 内预处理，训练 LogisticRegression、RandomForest、GaussianNB，以 train-only CV 选模型，再评估 test。 |
| [`scripts/verify_results.py`](scripts/verify_results.py) | 从保存的 predictions 独立复算 metrics、confusion matrix 与 split 记录。 |
| [`scripts/render_report.py`](scripts/render_report.py) | 从 evidence 生成两页以内的 PDF；生成后仍需逐页目视检查。 |
| [`results/metrics.json`](results/metrics.json) | 本次三个模型与 Dummy baseline 的 CV/test 指标。 |
| [`results/selection_before_test.json`](results/selection_before_test.json) | 读取 test 指标之前冻结的选型记录。 |

本次数据有 31,109 个可用于训练的样本和 13,333 个可评估 test 样本。预先指定的 primary metric 是 macro-F1。LogisticRegression 的 CV mean 为 0.7616246862，RandomForest 为 0.7616244592，差距仅 0.0000002270；自动选择 LogisticRegression 不意味着它有稳定优势。数值依据在 `results/` 中，新数据不能沿用这些结论。

## 复现

把 dataset 放在本地 `data/`，选择新的 `runs/` 目录，安装 [`requirements.txt`](requirements.txt) 所列版本，再运行：

```sh
python scripts/inspect_dataset.py --dataset data/dataset.zip --output runs/example/profile.json
python scripts/run_workflow.py --dataset data/dataset.zip --output runs/example --target label --positive yes
python scripts/verify_results.py --run-dir runs/example
python scripts/render_report.py --run-dir runs/example --name 'YOUR NAME' --matric 'YOUR MATRIC' --repository 'YOUR SKILL URL'
```

这些命令针对包含 `train.csv` 和 `test.csv` 的 ZIP。inspection 另支持 CSV、TSV、Excel 和 Parquet；后两类需相应 optional dependencies。当前训练脚本只实现 binary classification 和可近似 IID 的 split；若新数据涉及 multiclass、实体重复或时间顺序，应按 `SKILL.md` 改写模型与评估设计，不能直接把这份脚本当作通用处理器。

生成的 PDF 在 `runs/example/`。最终课程提交仍需填写真实 matric number、可确认的 LLM/interface 版本，并加入学生亲自完成的 Human oversight、Critical evaluation 与 manual check。仓库内的程序验证不能代替个人 Reflection。
