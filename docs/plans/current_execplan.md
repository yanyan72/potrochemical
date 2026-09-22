# E1g：可见工况切换的EWMA重置消融

## Purpose
同数据同阈值检验E1f跨切换漏检中重置的作用，交付策略开关、测试、五种子证据与GitHub更新。项目验收与论文写作分开，不修改外部论文。

## Current state
2026-09-22本地与远端f7cf25d一致。E1f五种子×12场景完成，65测试通过；条件EWMA切换前6步起始事件Recall15.90%，后6步38.47%，差异尚不能单独归因于重置。CUSUM和SMA不升级默认方法。质量目标未定义。

## Scientific assumptions
全部synthetic、抽象时间步、clean oracle训练、瞬时切换。同一通道的工况内标准化残差可跨工况递推是待检验假设，条件关系改变时尤其需谨慎。不声称正确概率、质量或工业有效性。

## Scope
固定E1f全部12场景、种子42–46、alpha=.2、原train_cal q99与参考，只将EWMA reset_on_record_change从true改为false。缺失/未知工况和序列起点仍重置；不拒绝可评分观测。旧六种评分通过manifest引用，复算EWMA核验；新增边际/条件carry。
旧train_cal全为固定状态，开关前后校准分数须完全相同，故复用旧阈值，不从validation校准。固定状态验证也须不变。新增可见记录变化后12步窗口，与真切换12步窗口分别报告，窗口可重合不能相加。

## Milestones
1. 输入既有代码/文档，冻结本计划、METHOD_LOGISTICS_E1G、configs/logistics_e1g.yaml。
2. temporal_reliability.py加入默认保持旧行为的开关；新增run_transition_e1g.py及测试。输入E1f存档/E1c训练，输出新模型/分数/配对差。验收公式、因果、缺失/未知、固定状态不变、旧分数重放和复现。
3. 全回归、compileall、diff通过后先提交实现，再运行：
   .venv/bin/python -m src.run_transition_e1g --config configs/logistics_e1g.yaml --run-id logistics_e1g_20260922_v1
   验收5种子×12场景、产物hash、图和失败分析。
4. 更新项目状态/任务/决策/交接及证据矩阵，提交结果，按已有授权上传并核对远端。凭据失效时在本地完成后获取设备授权。

## Data leakage controls
审核E1f209与E1c177产物hash及manifest关联，只读train_cal和已评价开发validation。评分输入只有观测/可见工况/训练模型；隐藏状态与mask仅评价。test不评价。沿用开发集是配对消融，不是新盲测。

## Reproducibility
不新增随机过程/依赖，不搜索alpha/阈值。新分数与开关模型落盘，旧数据manifest引用。每seed聚合再均值/样本SD，保存全时间轴/共同支持、逐工况、真/记录切换窗口及事件指标。代码先提交再运行，保存版本/环境/hash，不覆盖目录。

## Validation
手算carry/reset、前缀因果、序列隔离、缺失/未知清零、alpha1不变、固定状态校准/验证不变、旧EWMA完整重放、共同支持一致、独立重跑、hash与图检查。

## Progress
- [x] 2026-09-22核对本地/远端及约束，冻结方案。
- [x] 实现与测试完成：新增7项通过，全回归72项通过（17.60秒）；编译/diff检查通过，实现先提交后正式运行。
- [ ] 正式五种子运行与结果审查。
- [ ] 文档、结果提交及远端交付。

## Decision log
先比较全保留与全重置两个端点，不加识别器、混合参考或门控。carry保留标准化残差历史，当前残差/阈值仍用当前记录工况。风险是错误工况记录产生的偏离在记录恢复后继续拖尾。

## Surprises and discoveries
旧虚拟环境Python入口失效，按原requirements-core与Python3.11.16重建；未新增依赖。新增7项测试及全回归首次运行通过。

## Outcomes and retrospective
待结果，不预设carry胜出。不以此代替质量预测或真实数据任务。
