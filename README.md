# 区块链 + AI 石化质量可信预警研究

GitHub 仓库：<https://github.com/yanyan72/potrochemical>

## 1. 项目简介

本项目研究石化产品在生产、仓储、运输等过程中的质量状态预测、风险累积与可信预警。项目不把“区块链 + AI”作为简单技术拼接，而是尝试建立以下算法闭环：

1. 多源传感器数据形成多维时序向量；
2. 从正常训练数据构建参考可信集合；
3. 利用距离、密度、物理残差和链上完整性计算数据可信度；
4. 依据可信度修正或加权污染数据；
5. 使用 LSTM 预测质量状态；
6. 用动力学约束限制不合理预测；
7. 计算累积质量风险和阈值越界时间；
8. 将原始数据、参数、预测与预警摘要形成可验证记录。

## 2. 当前范围

当前第一阶段聚焦：

> 多维传感器数据可信度识别与可信度增强的质量时序预测。

暂不优先实现：

- 完整联盟链平台；
- 智能合约业务系统；
- Transformer/多模态视觉大模型；
- 全生命周期 UI；
- 复杂动态参数 PINN。

## 3. 第一阶段路线

```text
正常仿真时序
    ↓
异常注入与真值掩码
    ↓
训练集标准化
    ↓
可信集合建模
    ↓
马氏距离 / 连续可信度
    ↓
可信度修正
    ↓
普通 LSTM 与可信度 LSTM 对比
    ↓
污染比例鲁棒性实验
```

## 4. 计划目录

```text
.
├── AGENTS.md
├── README.md
├── MASTER_PROMPT_FOR_CODEX_WORK.md
├── requirements.txt
├── configs/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   ├── PROJECT_BRIEF.md
│   ├── ARCHITECTURE.md
│   ├── DATA_SPEC.md
│   ├── EXPERIMENT_PLAN.md
│   ├── TASKS.md
│   ├── DECISIONS.md
│   ├── STATUS.md
│   ├── SESSION_HANDOFF.md
│   └── plans/
├── prompts/
├── results/
│   ├── figures/
│   ├── tables/
│   └── models/
├── src/
└── tests/
```

## 5. 建议环境

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 6. 预期命令

以下命令由实现阶段逐步补齐：

```bash
python -m src.simulator --config configs/baseline.yaml
python -m src.trust_score --config configs/baseline.yaml
python -m src.train --config configs/baseline.yaml
python -m src.evaluate --config configs/baseline.yaml
pytest -q
```

### Milestone 0 已可运行

当前已实现配置驱动的干净时序仿真，以及 `spike`、`bias`、`drift`、`missing`、`random_replacement` 五类异常注入、逐特征掩码、事件元数据、测试和示例图。异常幅度和随机替换参考值只使用 clean train 序列。运行：

```bash
source .venv/bin/activate
MPLCONFIGDIR=/tmp/petrochemical_mplconfig python -m pytest -q -p no:cacheprovider
MPLCONFIGDIR=/tmp/petrochemical_mplconfig python -m src.generate_phase1 \
  --config configs/milestone0.yaml
```

每次生成都会创建唯一 run 目录，不覆盖历史结果。主数据位于 `data/processed/<run_id>/`，示例图位于 `results/figures/<run_id>/`。当前时间步为抽象采样单位，数据全部为仿真，不代表真实工业性能。

### Milestone 1 已可运行

当前已实现无泄漏的预处理基线：只用 clean train 的逐特征中位数拟合 missing 插补值，并只用同一 clean train 的均值和样本标准差拟合标准化参数；train/val/test 污染观测均只调用 `transform`。运行：

```bash
source .venv/bin/activate
python -m src.run_preprocessing \
  --config configs/milestone1_preprocess.yaml
```

每次运行会在 `data/processed/preprocess_<timestamp>_<config_hash>/` 创建唯一目录，保存插补后数组、标准化数组、原缺失掩码、序列/split 标识、参数 JSON、元数据和配置快照。该结果仍然全部来自仿真数据；clean-train 拟合是受控算法基线，不表示真实部署能够访问隐藏真值。

### Milestone 2 已可运行

当前已建立第一版正常统计参考集合：只选择 `train + normal` 的完整 synthetic clean 序列，沿用 Milestone 1 的 clean-train 标准化参数，再使用 Ledoit–Wolf 方法估计收缩协方差和精度矩阵。运行：

```bash
source .venv/bin/activate
python -m src.run_trust_reference \
  --config configs/milestone2_trust_reference.yaml
```

结果保存在 `data/processed/trustref_<timestamp>_<config_hash>/`，包括标准化正常参考矩阵、参考序列 ID、中心、协方差、精度矩阵、收缩系数、数值诊断、输入/输出哈希和配置快照。当前正式 run 使用 5 条 normal train 序列、800 个四维参考点；协方差最小特征值大于 0，逆矩阵残差约为 `2.49e-14`。Milestone 2 本身只证明统计参考估计可复现且数值可用；马氏距离由下面的 Milestone 3 独立实现，阈值、可信度和异常识别指标仍未计算。

### Milestone 3 已可运行

当前已使用 Milestone 2 冻结的中心和精度矩阵，对 Milestone 1 的全部标准化 synthetic 观测以及正常参考集计算平方马氏距离：

```bash
source .venv/bin/activate
python -m src.run_trust_scoring \
  --config configs/milestone3_mahalanobis.yaml
```

结果保存在 `data/processed/trustscore_<timestamp>_<config_hash>/`。正式 run `trustscore_20260814T140937863216Z_09a4bd41` 保存了 `(30,160)` 的全部观测距离和 `(800,)` 的参考距离；4,800 个观测距离与 800 个参考距离均有限、非负，独立重跑的 8 个 NPZ 数组逐值一致。该结果仅证明平方二次型计算、输入身份核验和持久化可复现；尚未选择 q90/q99、划分可信组、计算连续可信度或评价异常识别性能。

### 真实数据状态

`Pending external data / 等待外部数据`。当前没有实验室或企业真实数据，因此不实施真实数据清洗和工业验证，也不会编造企业、设备、批次或检测记录。未来接口与最小字段说明见 `data/raw/real/README.md`；在外部数据到位前，项目继续使用明确标记为 synthetic/simulated 的数据验证算法正确性和鲁棒性。

## 7. 当前状态

请阅读 `docs/STATUS.md` 和 `docs/project_log.md`。每次结束工作前必须更新状态、项目日志和 `docs/SESSION_HANDOFF.md`，确保下一次 Codex 会话只依赖仓库即可恢复上下文。

## 8. 关键原则

- 算法创新优先于场景包装。
- 先基线，后增强。
- 先可信度，后物理约束。
- 先轻量哈希链，后平台。
- 不伪造数据或结果。
- 所有结论必须由可复现实验支持。
