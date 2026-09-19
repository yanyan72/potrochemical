# ARCHITECTURE

## 当前储运路径

1. `configs/logistics_e1b.yaml`：示意分布、状态、种子、split规模、污染参数、校准分位数。
2. `src/logistics_simulator.py`：生成均衡的独立储运状态序列；复用污染注入，保留clean/observed/mask/events。
3. `src/sensor_reliability.py`：train_fit标准化和LedoitWolf参考；train_cal校准；观测+工况输出逐通道偏离、分数、可用性和告警。
4. `src/run_logistics_e1b.py`：5种子运行、共同完整行支持集评价、缺失与牵连误报、参数和数据保存、版本与哈希审计。
5. `results/logistics_e1b/<run_id>/`：唯一运行目录。

评分API不接收隐藏污染标签；未知状态下分工况评分不可用。边际方法只需本通道值，条件残差需要同一时刻的全部输入。缺失不能由隐藏真值填补。

## E1c压力实验路径

`configs/logistics_e1c.yaml -> logistics_robustness.py -> run_logistics_robustness.py`。训练参考复用E1b算法；场景数据包含逐时刻true_regimes/recorded_regimes和transition标记，评分仅接收记录状态。新数据/事件/评分、逐种子分层指标、配对差和图位于`results/logistics_e1c/<run_id>/`。这一路径不修改E1b生成器或评分公式。

## 历史路径（继续可复现）

`simulator.py -> corruption.py -> run_preprocessing.py -> run_trust_reference.py -> run_trust_scoring.py -> run_e1_evaluation.py`。
这是四变量全局评分路径，历史数据/结果不覆盖，不能代替新通道评分。

## 后续预测路径

实际数据字典/化验时间 -> 序列级划分 -> 因果窗口与标签对齐 -> 相同预测器的观测、观测+工况、观测+工况+评分三组对照。
历史质量只有已经出具才可作为输入。无质量标签时不生成伪连续真值。LSTM、选择性修复和真实接入待后续计划。

生产推广、物理约束、区块链仅为远期扩展，不属于当前已实现架构。
