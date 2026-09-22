# 石化产品运输与仓储：工况感知传感器级可信度研究

用户于2026-09-18确认：当前聚焦运输、仓储，生产环节仅作未来推广；先推进模拟算法，真实数据后续提供。研究目标是一篇包含真实过程验证的论文。

## 当前可运行内容

- 历史M0–M4：四变量模拟、污染注入、预处理、单一normal工况全局可信度及E1评价。
- 新储运E1b：独立的温度/相对湿度/振动示意基准，仓储/平稳运输/颠簸运输三种固定状态，完整序列均衡划分。
- 储运E1c：10个成对压力场景，弱偏置/污染比例/双通道/工况延迟，5种子分层与配对结果、完整故障集合评价。
- 储运E1d：复用E1c数据，逐点与因果EWMA对照、独立校准、事件新告警/延迟/拖尾，5种子消融。
- 储运E1e：完整9步滑动均值对照，共同支持/全时间轴双口径与启动期覆盖；结果未支持替换EWMA，负面证据完整保留。
- 储运E1f：新开发验证集上独立控制事件时长/数量/起点，5种子×12场景及经典CUSUM；固定CUSUM的24组F1均低于EWMA，保留失败证据。
- 储运E1g：同数据同阈值EWMA重置消融；carry改善跨切换检出，但延迟记录恢复后误报严重，保留reset工程默认。
- 五种逐传感器评分对照：单一参考边际、合并参考边际、分工况边际、合并参考条件残差、分工况条件残差。
- 可靠训练序列拟合，另一组训练序列校准q99；5个种子，仅评价validation，test保留。
- 缺失单独报告，不计作数值异常检测成功；记录污染对其他干净通道的牵连误报。

这不是经过产品标定的储运机理模型。状态合法不等于产品安全；一致性分数不是测量正确概率或质量指标。尚无质量预测收益和真实工业性能结论。

## 运行

建议Python 3.11。当前评分阶段不需要PyTorch：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-core.txt
python -m pytest -q
python -m src.run_logistics_e1b --config configs/logistics_e1b.yaml
python -m src.run_logistics_robustness --config configs/logistics_e1c.yaml
python -m src.run_temporal_e1d --config configs/logistics_e1d.yaml
python -m src.run_finite_memory_e1e --config configs/logistics_e1e.yaml
python -m src.run_controlled_e1f --config configs/logistics_e1f.yaml
python -m src.run_transition_e1g --config configs/logistics_e1g.yaml
```

结果在唯一的 `results/logistics_e1b/<run_id>/` 中，包含配置、数据、事件、参数、validation分数、逐种子/工况/污染类型指标、汇总和哈希。不能覆盖同名运行。

E1c结果位于 `results/logistics_e1c/<run_id>/`，另含配对差、双视图逐通道分数、PNG/PDF图。

历史命令仍可使用：

```bash
python -m src.generate_phase1 --config configs/milestone0.yaml
python -m src.run_preprocessing --config configs/milestone1_preprocess.yaml
python -m src.run_trust_reference --config configs/milestone2_trust_reference.yaml
python -m src.run_trust_scoring --config configs/milestone3_mahalanobis.yaml
python -m src.run_e1_evaluation --config configs/milestone4_e1_trust.yaml
```

## 阅读顺序与科学范围

- [状态与结果](docs/STATUS.md)
- [项目分阶段验收（与论文分开）](docs/PROJECT_ACCEPTANCE.md)
- [E1g重置消融结果与工程取舍](docs/RESULTS_LOGISTICS_E1G.md)
- [E1g固定阈值配对协议](docs/METHOD_LOGISTICS_E1G.md)
- [E1f受控事件结果与下一步](docs/RESULTS_LOGISTICS_E1F.md)
- [E1f事件配对与CUSUM协议](docs/METHOD_LOGISTICS_E1F.md)
- [E1e有限窗口结果、失败分析与论文对应](docs/RESULTS_LOGISTICS_E1E.md)
- [E1e公平支持与覆盖率协议](docs/METHOD_LOGISTICS_E1E.md)
- [E1d时序消融结果、论文对应与下一步](docs/RESULTS_LOGISTICS_E1D.md)
- [E1d因果累积和事件评价协议](docs/METHOD_LOGISTICS_E1D.md)
- [E1c鲁棒性结果、论文对应与下一步](docs/RESULTS_LOGISTICS_E1C.md)
- [E1c压力实验协议](docs/METHOD_LOGISTICS_E1C.md)
- [储运E1b五种子结果与负面发现](docs/RESULTS_LOGISTICS_E1B.md)
- [研究范围](docs/PROJECT_BRIEF.md)
- [方法定义](docs/METHOD_LOGISTICS_E1B.md)
- [实验协议](docs/EXPERIMENT_PLAN.md)
- [论文主张与证据](docs/PROJECT_EVIDENCE_MATRIX.md)
- [数据规范](docs/DATA_SPEC.md)
- [当前执行计划](docs/plans/current_execplan.md)
- [后续任务](docs/TASKS.md)

RQ1减少工况误报，RQ2定位污染，RQ3检验对质量预测的增益。三个问题分别验证，不用检测结果代替预测结果。真实数据未到位不阻塞RQ1/RQ2原型；质量指标、产品和化验时间尚未确认，不擅自生成新的产品质量真值。物理约束、区块链、生产过程推广均不属于当前实现范围。

主要结果至少5个种子；训练、校准、验证、测试按完整序列互斥；旧数据、旧结果和归档计划保留。
