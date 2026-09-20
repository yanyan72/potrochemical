# E1d：有符号残差的因果时序累积

## Purpose
用户2026-09-20要求继续推进并上传。检验单一EWMA模块是否改善持续弱偏置，完整记录检出/误报/告警延迟与拖尾，并形成论文证据对应。

## Current state
远程与本地e775e8f一致。E1c五种子10场景数据、训练参考和事件已保存，39测试通过。弱偏置难检出、工况延迟导致误报。真实质量任务尚未定义。

## Scientific assumptions
复用E1c全部synthetic validation，不改变数据、标签或参考。对分状态边际/条件的有符号标准残差分别计算h_t=alpha*r_t+(1-alpha)*h_(t-1)。alpha预先固定0.2，alpha=1为逐点消融。新序列、记录状态改变、不可用通道之后从零初始化；不读取未来或隐藏状态。EWMA本身为经典统计模块，不能当成新算法首创。

## Scope
2种残差×2个alpha×5种子×10场景。每组合在原train_cal完整序列独立校准逐状态逐通道q99；参考参数保持train_fit版本。名义校准误报预算一致，不保证验证误报相同；不调整validation阈值。事件评价包含命中、首次新告警、检测到事件的延迟、漏检计入时长的截断延迟、事件后干净通道拖尾。

## Milestones
1. 输入E1c与既有公式；新增temporal_reliability.py、run_temporal_e1d.py、配置和方法文档；验收alpha=1与E1c逐点得分/告警一致。
2. 测试有符号公式、因果前缀、序列/状态/缺失重置、校准隔离、事件漏检与提前告警；pytest全回归和最小烟雾运行。
3. 实现提交后运行`.venv/bin/python -m src.run_temporal_e1d --config configs/logistics_e1d.yaml --run-id logistics_e1d_20260920_v1`。保存参考、校准审核、评分、逐点/事件表、配对差和图，source/output hash核对。
4. 更新STATUS/TASKS/DECISIONS/SESSION_HANDOFF/论文证据；提交结果并按用户选择的设备授权码流程推送，核对远程。

## Data leakage controls
仅读取既有train_cal作EWMA阈值校准，不重新拟合reference；原train_fit模型从references.json读入。源运行hash验证后读取指定val数据和事件。评分不接收mask/true_regimes/events；仅observed和recorded_regimes。test不评分，标签只用于指标。

## Reproducibility
种子42–46，alpha=[1.0,0.2]提前固定，无调参扫描。源E1c路径及元数据hash记录；不重复复制大数据，保存输入manifest。各方法逐seed配对差，均值/样本标准差，禁止以时间点当独立实验。唯一目录，不覆盖历史产物。

## Validation
手算EWMA与条件残差；alpha=1退化一致性；修改未来输入不影响历史；重复序列不共享状态；记录切换重置与缺失后重置；校准只影响阈值不变reference；手算事件首次新告警/漏检惩罚/拖尾支持集；端到端重跑与hash验证。

## Progress
- [x] 2026-09-20 核对远程与本地e775e8f，冻结方案并归档E1c计划。
- [x] 2026-09-20 完成EWMA、事件评价、模型/评分保存和alpha=1复现审核；47测试通过（9.85秒），compileall及diff检查通过。
- [x] 2026-09-20 正式5种子×10场景×2残差×2alpha完成；源177/输出67hash通过，alpha=1评分复现通过，test未评分。
- [x] 2026-09-20 文档、实现7ca922a和结果ec519b5已提交并推送；设备授权成功，git ls-remote确认远程main为ec519b53065170d1b9b99ba753c312c42e792216。

## Decision log
alpha=0.2为单一候选，不根据已有val选择更优参数。校准使用模拟可靠train_cal的完整轨迹，以涵盖时间相关性；不直接套独立高斯稳态阈值。报告实际val误报，不能把相同q99称为严格匹配部署误报。拖尾可能造成收益抵消，保留负面结果。

## Surprises and discoveries
旧.venv Python入口再次失效，按既有Python3.11/requirements-core重建。首次新增测试7通过1失败：把校准FPR上限写死为1.1%，但720点q99可产生8/720=1.111%；改为按1/n分辨率检查，算法和阈值未调整。

## Outcomes and retrospective
交付E1d算法、事件指标、原始评分、五种子表图、论文对应与失败分析。1倍偏置条件EWMA局部收益；强偏置拖尾与响应延迟限制应用。下一步有限记忆对照；真实产品/质量标签、真实数据、更强异常检测基线和状态不确定性仍为后续。GitHub交付已完成并核对；首次无凭据推送失败，用户完成分步设备授权后成功，未改变科研结论。
