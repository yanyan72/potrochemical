# ARCHITECTURE

## 总体分层

```text
数据层
  ├─ 仿真数据（当前）/真实传感器数据（Pending external data）
  ├─ 清洁真值
  ├─ 污染观测
  └─ 区块链完整性标记
        ↓
可信度层
  ├─ 标准化
  ├─ 可信集合估计
  ├─ 马氏距离/密度
  ├─ 物理残差
  └─ 综合可信度
        ↓
数据处理层
  ├─ 保留
  ├─ 加权
  ├─ 局部稳健修正
  └─ 缺失插补
        ↓
预测层
  ├─ 持久性
  ├─ 线性回归
  ├─ LSTM
  └─ 物理约束 LSTM
        ↓
风险层
  ├─ 质量状态 Q
  ├─ 劣化速率 r
  ├─ 累积风险 R
  └─ 阈值越界时间
        ↓
审计层
  ├─ 数据摘要
  ├─ 参数
  ├─ 预测
  ├─ 预警
  └─ 哈希链
```

> 真实数据接入、清洗和外部有效性验证均为 `Pending external data / 等待外部数据`。当前架构只实现仿真路径，保留真实数据接口但不实现依赖真实数据的代码。

## 第一阶段模块

### `src/simulator.py`

- 生成正常工况时序；
- 生成质量真值；
- 保存生成参数；
- 固定随机种子；
- 输出统一数据结构。

### `src/corruption.py`

- spike；
- bias；
- drift；
- missing；
- random replacement；
- 返回污染数据、异常掩码和异常元数据。

### `src/preprocess.py`

- 已实现：clean train 逐特征中位数插补参数；
- 已实现：clean train 均值与样本标准差；
- 已实现：train/val/test 仅使用已拟合参数 transform；
- 已实现：保留完整序列、时间索引和 split；
- 待 Phase 3：滑动窗口构造。

### `src/trust_score.py`

- 已由 `src/trust_reference.py` 实现：clean train normal 参考集合；
- 已由 `src/trust_reference.py` 实现：Ledoit–Wolf 收缩协方差和精度矩阵；
- 已由 `src/trust_score.py` 实现：使用冻结 M2 参数的平方马氏距离；
- 阈值；
- 连续可信度；
- 高/中/低可信集合标签。

### `src/repair.py`

- 局部高可信中位数；
- 可信度融合；
- 边界处理；
- 无可信邻居时回退。

### `src/models.py`

- 持久性；
- 线性模型；
- LSTM；
- 后续物理约束模型。

### `src/train.py`

- 配置读取；
- 种子控制；
- 训练循环；
- 检查点；
- 指标日志；
- early stopping（后续）。

### `src/evaluate.py`

- 质量预测指标；
- 异常识别指标；
- 分工况指标；
- 多种污染比例；
- 图表和表格保存。

## 第二阶段模块

- `src/physics.py`
- `src/risk.py`
- `src/blockchain.py`

## 数据流原则

1. 清洁真值永远保留；
2. 污染只作用于观测特征，不覆盖真值；
3. 可信度模型只用训练数据拟合；
4. 修正后的数据作为独立版本保存；
5. 每个实验记录数据版本和配置哈希。
