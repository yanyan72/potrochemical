# STATUS

## 当前阶段

Milestone 0–3 已完成；Milestone 4 的第一版 E1 可信度校准与检测评价也已完成。项目目前处于“E1 基线已评价、预测实验暂缓”的阶段：单一 normal 工况参考能够发现多数人工污染，但会把合法的 high-temperature、high-vibration 和 compound 工况大量误判为传感器异常。因此下一步先完成工况条件化参考，再决定是否进入 persistence、线性回归和 LSTM。

全部数据和结果均为 `synthetic/simulated`。真实数据状态为 `Pending external data / 等待外部数据`。

## 已实际完成

- 可复现的四变量、多工况 synthetic 时序仿真；
- `spike`、`bias`、`drift`、`missing`、`random_replacement` 五类异常及掩码、事件元数据；
- train/val/test 完整序列划分；
- 只用 clean train 拟合的中位数插补和标准化；
- 只用 `train + normal + clean` 建立的 Ledoit–Wolf 参考中心、收缩协方差和精度矩阵；
- 对 4,800 个观测点及 800 个参考点计算平方马氏距离；
- 只用 800 个训练参考距离估计 q90/q99；
- `high/uncertain/low` 分组及连续可信度 `trust=exp(-d²/(2q90))`；
- 4,800 行逐点审计表，含序列、时间、工况、污染标签/类型、距离、可信度和分组；
- train/validation 的 q90/q99 检测指标、分异常类型召回率、分工况未污染点误报率；
- 第一张 validation distance/trust 时间序列图；
- E1 无泄漏、单调性、边界、指标和端到端测试；
- 正式 E1 run 与独立临时目录复现审计，六个核心产物逐字节一致。

## 已创建框架但尚未实现

- `data/raw/real/`：仅保留真实数据接口和字段说明；
- `results/models/`：目录存在，但尚无预测模型；
- `configs/baseline.yaml`：包含后续模型规划，相应高级算法尚未实现。

## 已计划但尚未开始

- 工况条件化参考集合及 E1 重评；
- PCA 二维可视化；
- persistence、线性回归、普通 LSTM；
- 中位数修正 LSTM、可信度修正 LSTM；
- 不同污染比例的鲁棒性和消融实验；
- 物理动力学约束、风险预警、轻量完整性记录和区块链扩展。

## Pending external data / 等待外部数据

- 实验室真实数据接入和清洗；
- 企业传感器、化验或加速老化数据；
- 真实工业验证和外部有效性评价。

这些内容不阻塞 synthetic 算法开发，但不得用仿真结果替代真实工业结论。

## 最新 E1 正式结果

- run：`e1trust_20260910T035417735017Z_29f039af`；
- code version：`6dc612a9bd485c729761376a19f1088f21fbae8c`；
- 数据来源：synthetic；30 条序列 × 160 步，共 4,800 行；
- 校准来源：800 个 `normal_train_clean_reference_distances`；
- q90：`7.497480759602013`；q99：`13.833465760805584`；
- trust 范围：`0.0` 至 `0.9984665059040305`；
- 分组计数：high 1,192、uncertain 116、low 3,492；
- validation q99：Precision `0.0967`、Recall `0.9302`、F1 `0.1752`、PR-AUC `0.4797`、FPR `0.8547`；
- validation 污染率：`0.0896`；
- validation q99 分类型召回：bias `1.00`、drift `1.00`、missing `0.75`、random replacement `0.9091`、spike `1.00`；
- validation 未污染点 q99 误报率：normal `0.0078`，compound/high-temperature/high-vibration 均为 `1.00`；
- test 只保存逐点分数，不计算指标、不参与方法调整；
- 独立 `/tmp` 重跑的逐点 CSV、校准 JSON、三张指标表和 PNG 均与正式结果逐字节一致。

## 结果解释

PR-AUC 高于 validation 污染率，说明平方马氏距离具有一定排序信息；但统一 q90/q99 对合法非 normal 工况产生严重误报。当前 E1 不能用于可信度修正或作为预测模型输入，也不能声称异常检测已经有效。问题来自参考定义与工况覆盖，而不是通过放宽测试集阈值来解决。

## 正式结果路径

- 逐点结果：`data/processed/e1trust_20260910T035417735017Z_29f039af/pointwise_trust_scores.csv`；
- 校准参数：`data/processed/e1trust_20260910T035417735017Z_29f039af/trust_calibration.json`；
- 元数据：`data/processed/e1trust_20260910T035417735017Z_29f039af/metadata.json`；
- 配置快照：`data/processed/e1trust_20260910T035417735017Z_29f039af/config_snapshot.yaml`；
- 检测指标：`results/tables/e1trust_20260910T035417735017Z_29f039af/detection_metrics.csv`；
- 分异常类型召回：`results/tables/e1trust_20260910T035417735017Z_29f039af/corruption_type_recall.csv`；
- 分工况误报：`results/tables/e1trust_20260910T035417735017Z_29f039af/condition_false_positive_rates.csv`；
- 时间序列图：`results/figures/e1trust_20260910T035417735017Z_29f039af/distance_trust_timeseries.png`。

## 当前限制与风险

- 全部实验为仿真，不能代表真实装置、产品或实验室性能；
- clean oracle 参考不等于真实部署可见数据；
- 第一版参考仅含 5 条 normal train 序列，800 个时间点并非 800 个独立实验；
- 单一 normal 参考无法覆盖合法非 normal 工况，造成严重工况混淆；
- M1 标准化来自全部 clean train，非 normal train 会间接影响参考坐标尺度；
- test 尚未评价，这是有意的数据泄漏控制，不是遗漏；
- 当前 `.venv` 使用 Python 3.14、scikit-learn 1.9.0，尚未在建议的 Python 3.11 复验；
- 尚未证明可信度能改善预测，也尚无任何 LSTM、PINN、Transformer、区块链平台或界面结果。

## 验证状态

- E1 专项测试：`11 passed`；
- 最终全套测试：`24 passed in 4.12s`；
- 语法编译：通过；
- 已知未解决的代码错误或失败测试：无；
- 科学方法问题：统一 normal 参考的跨工况误报过高，必须在下一里程碑处理。

## 下一步唯一优先任务

实现工况条件化 E1 参考：对每个已知 synthetic condition，仅使用对应的 clean train 序列拟合中心、Ledoit–Wolf 协方差、q90/q99 和 tau，再按相同 train/validation 指标和工况误报表重评。完成该门控前，不开始预测模型。
