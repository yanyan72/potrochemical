# PROJECT LOG

本文件按日期记录已经真实完成并验证的科研工程工作。所有结果均应区分“代码已实现”“测试已通过”和“实验已经证明有效”。当前数据来源全部为 synthetic/simulated；真实数据状态为 `Pending external data / 等待外部数据`。

## 2026-08-13：Milestone 2 正常参考集合与收缩协方差

### 本次目标

在不使用验证集、测试集和人工异常标签的前提下，从训练集正常工况建立多维统计参考集合，并得到可逆、数值稳定的协方差和精度矩阵，为下一步马氏距离计算提供参数。

### 完成内容

- 安装 requirements 已声明但当前 `.venv` 缺失的 scikit-learn 1.9.0；
- 新增 `src/trust_reference.py`，封装 Ledoit–Wolf 参考估计和数值诊断；
- 新增 `src/run_trust_reference.py`，核对 M0/M1 输入 run、SHA-256、序列顺序、split 和特征顺序；
- 新增 `configs/milestone2_trust_reference.yaml`；
- 从 M0 选择5条完整 `train + normal` 序列；
- 沿用 M1 clean-train 参数，将 clean 参考值标准化为800×4矩阵；
- 保存中心、收缩协方差、精度矩阵、收缩系数、序列 ID、配置、元数据和哈希；
- 新增 `tests/test_trust_reference.py` 和 `tests/test_trust_reference_pipeline.py`。

### 方法说明

Ledoit–Wolf 收缩协方差可以简单理解为：经验协方差能够描述温度、压力、振动和浓度如何共同变化，但变量相关性较强时直接求逆可能不稳定；收缩估计会把经验协方差适度拉向更规则的矩阵，从而得到更稳定的精度矩阵。

本次参考集合直接使用 synthetic clean 真值，是为了先建立可审计的 oracle baseline。它不是现实部署假设，不能描述成真实传感器系统能够知道隐藏正确值。

### 实际命令

```bash
.venv/bin/pip install 'scikit-learn>=1.4,<2'
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/petrochemical_mplconfig \
  .venv/bin/python -m pytest -q -p no:cacheprovider
PYTHONPYCACHEPREFIX=/tmp/petrochemical_pycache \
  .venv/bin/python -m compileall -q src tests
.venv/bin/python -m src.run_trust_reference \
  --config configs/milestone2_trust_reference.yaml
```

### 测试记录

- 首次全套测试：`1 failed, 11 passed in 16.55s`；
- 失败原因：NPZ中的参考序列ID为object dtype，默认安全加载不允许pickle；
- 修复：保存前强制转为Unicode数组，并新增dtype断言；
- 修复后专项测试：`3 passed in 2.14s`；
- 最终全套测试：`12 passed in 2.92s`；
- `compileall`：通过；
- 独立 `/tmp` 重跑：全部NPZ数组和参数JSON逐值一致。

### 正式实验结果

- run ID：`trustref_20260813T074437348987Z_34d5fa5c`；
- 数据来源：synthetic；
- 参考序列ID：`train_0004`、`train_0008`、`train_0010`、`train_0015`、`train_0017`；
- 参考矩阵：800×4；
- 收缩系数：`0.0036545028372638863`；
- 协方差特征值：`0.0051141175843415695`、`0.028090600353115384`、`0.041727836195232425`、`1.3954002296960009`；
- 协方差条件数：`272.85259024321965`；
- 协方差对称误差：`0.0`；
- 精度矩阵对称误差：`3.552713678800501e-15`；
- `covariance @ precision` 最大单位阵误差：`2.4868995751603507e-14`；
- NPZ SHA-256：`f22e4f7b90558a363b2a527ea84633f6fa64df165d2ebae524252f6015caccd2`；
- 参数JSON SHA-256：`00d3a9718c77f67157437b94339299d283c9f7e99a5bbad5709589e8748e109b`。

这些结果说明当前矩阵有限、对称、正定、可逆且可复现；它们不能说明异常检测已经有效。

### 结果路径

- `data/processed/trustref_20260813T074437348987Z_34d5fa5c/reference_set.npz`；
- `data/processed/trustref_20260813T074437348987Z_34d5fa5c/trust_reference_params.json`；
- `data/processed/trustref_20260813T074437348987Z_34d5fa5c/metadata.json`；
- `data/processed/trustref_20260813T074437348987Z_34d5fa5c/config_snapshot.yaml`。

### 已知限制

- 只有5条normal train序列，800个时间点存在序列内相关性；
- M1标准化参数来自全部clean train，非normal train会间接影响尺度；
- 当前没有真实实验室或企业数据；
- 当前Python 3.14尚未在规范建议的Python 3.11复验；
- 仓库无Git元数据。

### 下一步唯一计划

使用已保存中心和精度矩阵计算参考集与全部标准化观测的平方马氏距离，并添加公式数值测试、shape/有限性检查、输入身份核验和复现测试。本步先不选择q90/q99阈值，不计算Precision、Recall、F1或PR-AUC。
