# STATUS

## 当前阶段

Milestone 0 的可复现仿真与五类异常、Milestone 1 的无泄漏插补与标准化、Milestone 2 的正常参考集合与 Ledoit–Wolf 收缩协方差、Milestone 3 的平方马氏距离均已完成。项目处于 E1 可信度识别阶段；尚未计算 q90/q99、可信组、连续可信度或异常识别指标，也未进入预测模型、物理约束、区块链或界面。

## 已实际完成

- 四变量、多工况、离散质量递推的 synthetic time series；
- train/val/test 完整序列划分和五类异常注入；
- clean、observed、quality、condition、split、mask 和异常事件保存；
- clean-train feature median 插补与 mean/sample-std 标准化；
- train + normal 完整序列正常参考集合；
- 使用 Milestone 1 参数标准化 clean 参考值；
- Ledoit–Wolf 中心、收缩协方差、精度矩阵和数值诊断；
- 使用冻结 M2 参数计算全部标准化观测和正常参考集的平方马氏距离；
- M1/M2 run ID、SHA-256、特征/序列/split 和参考参数跨文件身份核验；
- JSON/NPZ/YAML、输入/输出哈希和唯一 run 持久化；
- 当前全套 16 项自动测试通过；
- M2 正式 run 在独立临时目录重跑，全部参考数组和参数逐值一致。
- M3 正式 run 在独立临时目录重跑，全部8个距离与身份数组逐值一致。

## 已创建框架但尚未实现

- `data/raw/real/`：只有真实数据接口和字段说明；
- `results/tables/`、`results/models/`：尚无指标表或模型；
- `configs/baseline.yaml`：含后续可信度和模型规划，相应算法未全部实现。

## 已计划但尚未开始

- q90/q99 分组和连续可信度；
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

### M3 平方马氏距离

- run：`trustscore_20260814T140937863216Z_09a4bd41`；
- code version：`e1748bf0f9c2096ad700ec7d1e61212122f4a609`；
- 观测距离：`(30,160)`，4,800个，范围`0.023012334111355618`至`13004.834691635755`；
- 正常参考距离：`(800,)`，范围`0.12766020043397946`至`24.026885127638717`；
- 两组距离的非有限值数和负值数均为0；
- NPZ SHA-256：`587892147cb6a818213c85fe69a0180deb20e99069829affdc532b36755dfd53`；
- 独立重跑：NPZ中8个数组逐值一致；
- 最新专项测试：`4 passed in 2.13s`；
- 最新全量测试：`16 passed in 3.63s`；
- 语法编译：通过。

## 结果路径

- M0：`data/processed/milestone0_20260719T153039079827Z_0abad9ce/`；
- M1：`data/processed/preprocess_20260728T101115651491Z_40442611/`；
- M2：`data/processed/trustref_20260813T074437348987Z_34d5fa5c/`；
- M2 参数：`data/processed/trustref_20260813T074437348987Z_34d5fa5c/trust_reference_params.json`；
- M2 参考数组：`data/processed/trustref_20260813T074437348987Z_34d5fa5c/reference_set.npz`；
- M3：`data/processed/trustscore_20260814T140937863216Z_09a4bd41/`；
- M3 距离数组：`data/processed/trustscore_20260814T140937863216Z_09a4bd41/mahalanobis_distances.npz`；
- 项目日志：`docs/project_log.md`；
- 执行计划：`docs/plans/current_execplan.md`。

## 当前限制与风险

- 全部实验为仿真，不代表真实装置、产品或实验室结论；
- clean oracle 参考不等于真实部署数据；
- normal train 只有 5 条序列，虽然共有 800 个时间点，但时序相关性意味着有效独立样本数小于 800；
- M2 结果只说明协方差为正定且数值可逆，不能单独说明异常检测有效；
- 当前距离尚无训练参考阈值、可信组或标签评价，不能说明异常检测已经有效；
- standardization 来自全部 clean train，非 normal train 会间接影响参考坐标尺度；
- 当前 `.venv` 为 Python 3.14，scikit-learn 1.9.0，尚未在规范建议的 Python 3.11 复验；
- 仓库已初始化 Git 并推送 GitHub；历史 M0–M2 产物仍保留其生成时的 `unavailable_no_git_repository`，M3 已记录真实 commit。

## 下一步唯一优先任务

只使用已保存的800个正常训练参考距离计算 q90/q99 基线阈值，并对全部距离划分 `high/uncertain/low` 可信组；补充分位数公式、边界归组、train-only 阈值来源、shape 和复现测试。本小步先不计算连续可信度、PCA 或 Precision/Recall/F1/PR-AUC。
