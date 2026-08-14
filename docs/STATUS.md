# STATUS

## 当前阶段

Milestone 0 的可复现仿真与五类异常、Milestone 1 的无泄漏插补与标准化、Milestone 2 的正常参考集合与 Ledoit–Wolf 收缩协方差均已完成。项目处于 E1 可信度识别的统计基础阶段；尚未计算马氏距离、q90/q99、连续可信度或异常识别指标，也未进入预测模型、物理约束、区块链或界面。

## 已实际完成

- 四变量、多工况、离散质量递推的 synthetic time series；
- train/val/test 完整序列划分和五类异常注入；
- clean、observed、quality、condition、split、mask 和异常事件保存；
- clean-train feature median 插补与 mean/sample-std 标准化；
- train + normal 完整序列正常参考集合；
- 使用 Milestone 1 参数标准化 clean 参考值；
- Ledoit–Wolf 中心、收缩协方差、精度矩阵和数值诊断；
- JSON/NPZ/YAML、输入/输出哈希和唯一 run 持久化；
- 当前全套 12 项自动测试通过；
- M2 正式 run 在独立临时目录重跑，全部参考数组和参数逐值一致。

## 已创建框架但尚未实现

- `data/raw/real/`：只有真实数据接口和字段说明；
- `results/tables/`、`results/models/`：尚无指标表或模型；
- `configs/baseline.yaml`：含后续可信度和模型规划，相应算法未全部实现。

## 已计划但尚未开始

- 马氏距离、q90/q99 分组和连续可信度；
- PCA/距离/可信度图；
- Precision、Recall、F1、PR-AUC 和每类异常分析；
- persistence、线性回归、LSTM 及可信度增强比较；
- 污染比例鲁棒性、物理约束、风险预警和区块链完整性。

## Pending external data / 等待外部数据

- 实验室真实数据接入；
- 企业传感器或化验数据清洗；
- 加速老化实验数据；
- 真实工业验证和外部有效性评价。

以上不阻塞 synthetic 算法开发，但不得用当前结果替代真实工业结论。

## 最新验证结果

### M0 数据与 M1 预处理

- M0 run：`milestone0_20260719T153039079827Z_0abad9ce`；
- M1 run：`preprocess_20260728T101115651491Z_40442611`；
- 30 条序列 × 160 步 × 4 特征；train/val/test = 18/6/6；
- 96 个 missing 全部插补，标准化后非有限值为 0。

### M2 正常参考与协方差

- run：`trustref_20260813T074437348987Z_34d5fa5c`；
- 参考序列：5 条完整 normal train 序列；
- 参考矩阵：800 行 × 4 特征；
- Ledoit–Wolf 收缩系数：`0.0036545028372638863`；
- 协方差特征值：`0.0051141175843415695`、`0.028090600353115384`、`0.041727836195232425`、`1.3954002296960009`，均为正；
- 条件数：`272.85259024321965`；
- 协方差对称误差：`0.0`；
- `covariance @ precision` 单位阵最大绝对误差：`2.4868995751603507e-14`；
- NPZ SHA-256：`f22e4f7b90558a363b2a527ea84633f6fa64df165d2ebae524252f6015caccd2`；
- 参数 JSON SHA-256：`00d3a9718c77f67157437b94339299d283c9f7e99a5bbad5709589e8748e109b`；
- 独立重跑：全部 NPZ 数组和参数 JSON 逐值一致；
- 最新全量测试：`12 passed in 2.92s`；
- 语法编译：通过。

## 结果路径

- M0：`data/processed/milestone0_20260719T153039079827Z_0abad9ce/`；
- M1：`data/processed/preprocess_20260728T101115651491Z_40442611/`；
- M2：`data/processed/trustref_20260813T074437348987Z_34d5fa5c/`；
- M2 参数：`data/processed/trustref_20260813T074437348987Z_34d5fa5c/trust_reference_params.json`；
- M2 参考数组：`data/processed/trustref_20260813T074437348987Z_34d5fa5c/reference_set.npz`；
- 项目日志：`docs/project_log.md`；
- 执行计划：`docs/plans/current_execplan.md`。

## 当前限制与风险

- 全部实验为仿真，不代表真实装置、产品或实验室结论；
- clean oracle 参考不等于真实部署数据；
- normal train 只有 5 条序列，虽然共有 800 个时间点，但时序相关性意味着有效独立样本数小于 800；
- 当前结果只说明协方差为正定且数值可逆，不能说明异常检测已经有效；
- standardization 来自全部 clean train，非 normal train 会间接影响参考坐标尺度；
- 当前 `.venv` 为 Python 3.14，scikit-learn 1.9.0，尚未在规范建议的 Python 3.11 复验；
- 仓库仍无 Git 元数据，代码版本为 `unavailable_no_git_repository`。

## 下一步唯一优先任务

更新 ExecPlan，使用已保存的中心和精度矩阵计算正常参考集及全部标准化观测的平方马氏距离，并添加公式数值测试、shape/有限性测试和 train-only 输入身份检查。本小步先不选择 q90/q99 阈值，也不计算 Precision/Recall/F1。
