# Purpose
完成21个固定预算RQ3开发预测模型，解释评分与融合的价值并交付写作图表。

# Current state
main起点7135b46789dd644d27f3739c08a77300d30aef5b。评分候选已验收；数据为原重建版，评分完整结果为shared_mode_fault_development_20261029_v2。原预测管线提供单层LSTM、共享故障训练范式；本轮新增18维公共输入实现，旧结果保持不变。

# Scientific assumptions
温度暴露质量代理、3工况、单传感器12步故障；候选12状态工作模型沿用冻结参数。实测SBS独立，本轮不检验真实产品质量。

# Scope
三数据种子、七组、40轮；复用1000train/250val。比较旧新拼接、单纯门控、参考回填、随机与无故障假设消融，不重跑评分筛查。

# Milestones
1. 输入旧协议/结果，输出新协议config；先推送。检验配置覆盖七组和21模型。
2. 新增src/rq3_joint_prediction_dev.py与tests/test_rq3_joint_prediction_dev.py；通过门控恒等、随机排除、公共特征与微型训练测试。命令：PYTHONPATH=src python -m unittest discover -s tests -p test_rq3_joint_prediction_dev.py。
3. 用脚本run入口逐seed保存模型/日志；输出新results/rq3_joint_prediction_dev_20261029_v1；各模型320更新、共同初始权重与日程一致。命令见脚本--help和发布复现说明。
4. 读取存档预测与模型，独立重算和重放；自动生成表/图/报告；更新状态和决策并推送。验收：21模型完整，主/分层结果均保存，远端文件一致。

# Data leakage controls
独立train/val序列，标准化与随机轨迹池train-only；正常参考既有fit/cal。目标只进入损失与评价，不进入评分/融合。无test入口。

# Reproducibility
配置rq3_joint_prediction_dev_20261029.json给出所有种子。输入哈希和实现commit入库；模型逐个保存，输出目录拒绝覆盖。

# Validation
数学边界值、共享输入/预算、训练排除自身随机评分、存档重放、独立指标重算及论文图人工查看。

# Progress
- [x] 2026-10-29 用户验收评分机制；七组预测协议确定。
- [ ] 实现与软件测试。
- [ ] 21模型与机制分析。
- [ ] 图表报告及远端校验。

# Decision log
按用户新要求，将NR替换ZG以直接区分门控与工况参考回填，保留RG；不增加模型数。旧数值门槛报告但不作为唯一科学价值标准。

# Surprises and discoveries
待预测结果产生后填写。

# Outcomes and retrospective
待实验交付。
