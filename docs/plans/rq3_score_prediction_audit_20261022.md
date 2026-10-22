# RQ3配对证据：评分有效性与预测增量分开核查

## Purpose
按用户和导师反馈优先检查配对结果，分别回答评分的观测一致性诊断能力与预测增量价值，交付可审核诊断报告和下一轮有限实验依据。正面结论不预设，旧结果不覆盖。

## Current state
main=9b01fe883d8bcd2af8fc6f108e1d143f419db247。旧10000批重建数据、两架构开发预测、已用final测试及1250批新正常噪声敏感性都已保存。final主端点未支持评分增益；正常噪声实验没有人工故障，不能评价故障识别。本机Python为../lstm-venv/bin/python。

## Scientific assumptions
Trust Score是参考工况下可见观测的因果EWMA一致性分数，不是真实质量概率或测量正确概率。故障mask为人工注入标签，只检验设定异常下的识别表现。评分由已有观测和训练参考计算，研究有限预测器是否获得表示/先验帮助，不能直接声称新增独立质量信息。当前arrival_proxy依赖温度潜在过程，湿度和振动不直接进入标签公式，但可能通过共同因子相关；因此分开通道诊断，不把污染识别等同质量误差减少。

## Scope
本轮只读既有数据和参数，重算旧val评分，并读取已经保存的val预测做配对诊断；不重新训练或预测旧test。五seed各250个开发批次、十配对视图。旧final和最新正常噪声结果仅引用已发布汇总，不能因诊断再调整原test结论。结果为事后开发分析，不作为新的独立确认。

## Milestones
1. 提交本计划和configs/rq3_score_prediction_audit_20261022.yaml；冻结诊断定义后实现。
2. 新增src/rq3_score_prediction_audit.py、tests/test_rq3_score_prediction_audit.py；运行pytest验证事件窗口、配对差和指标口径。源码及测试先推送。
3. 输入旧data/ridge/lstm存档；命令python -m src.rq3_score_prediction_audit --config configs/rq3_score_prediction_audit_20261022.yaml --output results/rq3_prediction/rq3_score_prediction_audit_20261022_v1 --implementation-commit <sha>，以既有venv执行。输出各seed配对数组及指标、汇总、来源校验；新目录拒绝覆盖。验收1250批各视图ID及目标匹配、源文件未改变、指标有限或明示不可用。
4. 按五seed和汇总提交GitHub，核验远端blob，形成导师MD和更新STATUS/DECISIONS；依据实际瓶颈明确下一轮实验，禁止把评分检测表现冒充RQ3正面结果。

## Diagnostic definitions
第一层：六个数值故障视图的逐点AP（异常度1−score，记录阳性比例）、固定阈值逐点召回、同事件窗口未污染通道告警率、原始同位置告警率、事件窗口内任意告警率和新告警率（与预先活跃告警区别）；配对的原始与污染事件通道平均score差（正数表示下降）。每seed先汇总再给五seed均值SD；相关时间点不当独立样本。missing单独报告事件单元可评分率，不把不可评分当作高可信或检测成功；delay仅说明条件记录错配，不能当数值故障定位。

第二层：从保存预测计算逐批|B3-y|−|B2-y|及相对原始条件的误差差变化；统计均值、样本SD、改善批次比例，并按实际故障通道分层。主要目标arrival及负对照null均保留。检查六数值故障下“新告警/未新告警”批次的配对差，属于事后条件化描述，不作因果归因或显著性选优。保存批次对齐的数组使后续能独立重算聚合表。

## Data leakage controls
评分只接受可见观测和记录工况，标签和事件仅用于事后评价。评分参考/阈值/模型全部冻结；不读取隐藏过程帮助打分。新开发分析使用val，val已参与模型选择，不称无偏测试。旧test仅文件hash校验和引用已有结论。

## Reproducibility
固定配置、旧manifest和源码身份、seed42—46、版本ID。输出numpy数组/CSV/JSON清单，拒绝覆盖；记录实现提交，分阶段推送。发生器与既有模型实现不改。

## Validation
手工构造新告警与延续告警事件，检查受影响/牵连通道；预测配对差手算；异常掩码/ID/标签/输入hash一致；从落盘batch配对数组独立复算第二层表。评分AP使用sklearn标准average_precision_score，不把它称概率校准或工业准确率。

## Progress
- [x] 核对main和旧文件，确认最近噪声样本无注入故障。
- [ ] 计划配置提交。
- [ ] 实现测试提交。
- [ ] 配对诊断、独立重算、上传。
- [ ] 导师报告与项目状态同步。

## Decision log
优先定位“评分识别→预测利用”之间的断点，暂不盲目扩大样本或同时改多个机制。是否采用公平的故障训练增强实验，由本次诊断和明确的新协议决定，不能预先声称其有效。

## Surprises and discoveries
执行后填写。

## Outcomes and retrospective
执行后填写实际诊断、限制和后续实验范围。
