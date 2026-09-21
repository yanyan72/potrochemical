# E1f：事件时长、数量与起点分离，CUSUM基线

## Purpose
修正E1c比例预算同时改变事件时长、起点落在12步网格的问题，在新模拟validation上比较逐点/EWMA/CUSUM。交付可复现代码、5种子结果、失败分析和论文对应，并上传GitHub。

## Current state
2026-09-21本地/远端705b173一致。E1e否决完整9步SMA默认增强。55测试已有记录；质量目标尚未定义。读取E1c训练参考及E1d逐点/EWMA模型，不读取旧test性能。

## Scientific assumptions
运输仓储示意分布与E1c相同，但validation使用独立随机命名空间20260921，属于新的开发验证集而非最终盲测。序列180步、每初始状态4条、种子42–46。段偏置一次一个通道，幅度来自旧train_fit尺度。
CUSUM双侧C+=max(0,C+prev+u-k)、C-=max(0,C-prev-u-k)，统计max(C+,C-)，预设k=.5；依据NIST经典公式，阈值采用独立train_cal经验q99。不在告警时重置，便于连续评分与恢复评价；这不是经典ARL设计/停机策略，也不是新算法。记录状态改变/缺失/序列起点重置。

## Scope
12场景×5种子×2残差×3统计。duration=[4,12,24]，count=[1,2,3]，magnitude=[1,2,4]，基准12步/2事件/1倍/固定状态。随机候选事件以最大时长24和至少12步间隔约束，预留12步尾窗；数量对照使用同一候选的嵌套子集，时长对照起点/符号/通道不变。占用比例是数量×时长/180的派生量，不能要求三者同时固定。
切换对照沿用60步边界；包含随机起点、记录延迟12步，以及相对60/120边界的-6/0/+6步定点事件。定点组改变起点，跨边界与工况分布影响不能完全分离。SMA不继续搜索，质量预测/真实数据仍未运行。

## Milestones
1. 冻结本计划、配置、NIST依据和METHOD_LOGISTICS_E1F。
2. 新增controlled_events.py、cusum_reliability.py、run_controlled_e1f.py和测试。验收因果公式/状态重置、事件精确长度/嵌套数量/随机相位、训练参考与验证隔离、端到端复现。
3. 全回归和diff/编译检查；实现先提交，运行：.venv/bin/python -m src.run_controlled_e1f --config configs/logistics_e1f.yaml --run-id logistics_e1f_20260921_v1。保存新数据、事件、模型/分数、分工况/窗口/事件指标、配对差、表图和hash。
4. 更新状态/任务/决策/论文证据与交接，结果提交后按用户设备授权方式上传，核对远端。

## Data leakage controls
源E1c/E1d产物与相互引用审核；旧train_fit参考、旧train_cal阈值校准，validation标签只评价。新随机流不生成或评价test。评分函数只接受observed和recorded_regimes。输入manifest、校准ID与新validation ID落盘。不能跨E1c/E1f数据直接归因。

## Reproducibility
W无搜索，alpha=.2与k=.5固定；相同名义q99并报告实际clean FPR。先按seed聚合，再均值/样本标准差。新目录不覆盖；原始模型、分数、事件与配置均保存。新环境按requirements-core和Python3.11重建，metadata记录版本。

## Validation
手算双侧CUSUM、前缀因果、缺失/未知/记录变化重置、校准缩放/隔离；事件mask、duration/count精确、同因子配对、强制覆盖非网格相位、定点边界、独立随机流；完整回归、两次小运行一致、hash和test未评价。正式图人工检查。

## Progress
- [x] 2026-09-21核对仓库和规则，冻结方案。
- [x] 实现和科学测试：65项全回归通过（20.38秒）；compileall、diff检查通过，实施提交后运行正式实验。
- [ ] 正式五种子实验与结果解读。
- [ ] 文档、提交与远程交付。

## Decision log
一次比较一种时间统计变化；生成器修正对所有方法同时使用。新增CUSUM是更强候选对照，不承诺效果；持续评分无告警重置会有恢复代价，必须同时报告。污染事件受防重叠/预留尾窗约束，不称全时刻均匀独立发生或真实故障模型。

## Surprises and discoveries
旧.venv入口失效，按既有依赖重建。10项新增测试首次运行通过，完整回归65项通过；未新增依赖。

## Outcomes and retrospective
待运行，不预设CUSUM胜出；质量预测任务字段仍是主线闭环所需信息。
