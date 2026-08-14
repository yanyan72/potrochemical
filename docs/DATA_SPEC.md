# DATA_SPEC

## 数据来源状态

当前只有 synthetic/simulated data。实验室和企业真实数据接入、清洗及工业验证为 `Pending external data / 等待外部数据`；以下真实数据字段规划保留，但待获得数据后完成，不得用仿真数据替代真实工业结论。

## 基本单位

一条序列代表一个独立批次、储运任务或连续工况片段。每个序列有固定时间步。

## 长表字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `sequence_id` | int/str | 完整序列 ID |
| `time_index` | int | 时间步 |
| `timestamp` | optional datetime | 真实数据阶段使用；当前等待外部数据 |
| `temperature_clean` | float | 干净温度 |
| `pressure_clean` | float | 干净压力 |
| `vibration_clean` | float | 干净振动 |
| `concentration_clean` | float | 干净浓度 |
| `temperature_obs` | float | 污染后温度 |
| `pressure_obs` | float | 污染后压力 |
| `vibration_obs` | float | 污染后振动 |
| `concentration_obs` | float | 污染后浓度 |
| `quality` | float | 质量真值，范围 `[0,1]` |
| `condition` | category | 正常/温度异常/振动异常/复合异常 |
| `is_corrupted` | bool | 是否为人工污染点 |
| `corruption_type` | category | spike/bias/drift/missing/replacement |
| `corruption_feature` | category | 被污染变量 |
| `corruption_magnitude` | nullable float | 异常幅度；missing 时为空并由掩码区分 |
| `trust_score` | float | 后续计算，范围 `[0,1]` |
| `trust_group` | category | high/uncertain/low |
| `split` | category | train/val/test |

## 数据约束

- `0 <= quality <= 1`
- 每个 `sequence_id` 只能属于一个 split
- 干净列不可被异常注入覆盖
- 异常掩码必须与注入位置一致
- 非缺失型污染后值必须为有限数
- 缺失型异常必须可由掩码区分
- 所有特征的单位和范围写入元数据
- 标准化器只在训练集拟合

## 建议保存格式

初期：

- Parquet：主数据；
- YAML/JSON：生成参数和元数据；
- NPZ：训练窗口（可选）；
- CSV：仅用于人工检查的小样本。

## 数据划分

按完整序列划分：

- 训练 70%
- 验证 15%
- 测试 15%

最终比例可配置，但不得按窗口随机切分导致序列泄漏。

## 版本标识

每个生成数据集应包含：

- 生成时间；
- 随机种子；
- 配置文件副本；
- 数据哈希；
- 代码版本；
- 数据集 ID。
