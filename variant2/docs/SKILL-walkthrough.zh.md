# SKILL.md 逐段讲解

这份 SKILL 是给 AI Agent 的可执行工作说明。AI 根据它分析数据、制定决策并调用工具；真正的训练和数字计算由 Python 完成。它本身不是训练模型，也不是一个自动调用 LLM API 的 Python 程序。

下面按当前 SKILL.md 的原文顺序解释；行号对应已交付的 38 行版本。

## 开头：让 Agent 知道何时使用、接收什么

**Frontmatter，lines 1–4：`name` 与 `description`。** `name` 是 SKILL 的标识；`description` 告诉 Agent 它适用于从 tabular dataset 完成 binary/multiclass classification 和 PDF report 的请求。这里描述的是用途，不替代后续步骤。把 scope 写清楚，可以避免把 regression 或图像识别任务误当成这个 classification workflow。

**任务定义，line 8：从 dataset path 开始，而不是把作业数据写死。** 用户输入一个表格、train/test 目录或 ZIP，Agent 都应先分析数据。脚本位置固定在 SKILL 自己的 `scripts/` 内，方便整个文件夹复制复用；target、class labels 和模型选择放在本次的 decisions.json 中。因此另一个数据集的 target 可以叫 outcome，不必也叫 label。

## Data profiling：先确认预测目标和数据问题

**读取与 profile，line 12。** `profile` 只生成数据概况，不进行模型训练：记录行列数、类型、缺失、类别分布、重复行、数值摘要和 IQR flags。CSV/TSV 直接读取，XLSX/Parquet 需要额外依赖。如果目录里有多份无法判断用途的文件，Agent 必须先确认。数据单元格中的文字不应被当作要求 Agent 改规则的指令。

**审查 profile，line 14。** Agent 要先知道预测什么，再决定怎么处理。target 是真实答案；猜测 missing target 再拿来训练或评分，会把伪标签当作事实。本次因此删除训练中的 3 行 missing label；test 的 1 行 missing label 只预测、不计分。IQR 只是统计上的偏离，不能证明值错误，所以本次没有自动删除 outliers。发现 ID、group 或时间相关性，则会影响数据划分方式。

## decisions.json：把本次选择与理由留给人检查

**字段清单，lines 18–24。** JSON 既供脚本读取，也供你审阅。`target` 指定要预测的列；`seed` 固定随机性；`folds` 指定 CV 折数；`split_reason` 解释独立性假设；`drop_columns/drop_reason` 解释是否删除 features。`primary_metric/metric_reason` 提前定义“哪个模型更好”。preprocessing 字段指定 imputation 和 encoding；`models` 保存候选 classifier、class weights 和 hyperparameter grid；metadata 满足报告身份与工具说明；discussion 放入解释和局限。

本次设定 target=label、seed=42、folds=3，保留全部 15 个 predictors，以 macro-F1 选模型。macro-F1 是先分别计算 no、yes 的 F1，再取等权平均，不会让占 76% 的 no 类直接主导平均结果。但选择 macro-F1 本身仍是价值判断：如果漏检 yes 的损失远高于误报，可能应优先考虑 Recall 或错误成本。

**AI 与 Python 的分工，line 26。** 模型不是 Python 根据一条固定规则“冒充 AI 选出来”的；是在这次对话中由 AI 看 profile 后选择，并记录理由。Logistic Regression 提供 regularized linear baseline；Random Forest 能表示 nonlinear interactions。Python 按这些配置计算 CV，并根据预定 metric 选择 winner。它不会在训练途中再咨询一个隐藏的 LLM。

**复用边界，line 28。** StratifiedKFold 保持各 fold 的类别比例，但它并不能解决同一个人的多条记录或时间依赖造成的 leakage。当前 helper 只直接实现 IID split；有 subject/group/time 结构时，Agent 必须先扩展为合适的 splitting 方法。已用 Iris multiclass 数据测试复用，不等于证明所有 tabular dataset 都能不加判断直接运行。这是当前实现最值得诚实说明的限制。

## 执行与验证：防止“数字看起来不错，但结论不可靠”

**训练执行，line 32。** 有现成 train/test 就保留它；只有单表时，使用 stratified holdout。每个 CV training fold 单独 fit imputer、StandardScaler 和 OneHotEncoder，validation fold 只能 transform。这样 validation 的 median 或类别词表不会提前进入训练。所有候选模型选完配置后，再计算 test metrics。本次虽然先做过 test 的数据质量检查，但没有使用 test performance 来选择 hyperparameters，不能把它描述成“程序直到最后才首次读取 test 文件”。

**证据核验，line 34。** CV 记录显示如何选参数；Confusion Matrix 显示错误在哪里；predictions.csv 允许从每行预测独立重算指标；source SHA-256 用于确认输入文件是否一致。`verify.py` 使用计数公式核对 Accuracy、macro-F1 和 Balanced Accuracy，没有直接复用训练时的 sklearn metrics 函数。程序核验已完成，但用户 manual check 是另外一件事，需要你亲自完成才能写进 Reflection。

## 报告和 Reflection：自动生成事实，保留真实的人类判断

**自动生成 PDF，line 36。** `report.py` 从 metrics.json 填表和生成结果描述，并用 decisions.json 写入方法理由及 metadata。因此表里的 0.7721 来自真实实验，而不是 LLM 猜测。报告保持两页正文和模板主要 sections；生成后检查页数并 render 每页确认可读性。换成长文本、更多模型或更多类别时，仍可能超页，Agent 应压缩措辞并重新检查，不能只依赖“脚本能运行”。

**Reflection 与交付边界，line 38。** 你需要说明实际审阅过什么、质疑什么、怎样核对结果。Agent 可以解释并整理英文表达，但不能替你声称已经做过 manual review。第一次执行时没有 GitHub 发布授权，所以只交付本地文件；现在你明确要求上传，便可以执行上传。NTULearn 提交是另一个动作，本次请求没有要求它。

## 现在如何把理解变成 Reflection

你现在可以先核对 Random Forest 的 Confusion Matrix，再选择是否接受以 macro-F1 选模型。训练已完成，因此真实的 Human oversight 可以写成“我在接受最终结果前进行了事后审阅”，不能写成“我在训练前批准了参数”。下一次改进可以是让 Agent 在训练前展示 decisions.json，并在 training CV 内比较 weighted/unweighted models；本次没有做过的 ablation，只能写作 future improvement。
