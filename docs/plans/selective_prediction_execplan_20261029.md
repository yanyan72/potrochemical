# Purpose
检验异常触发修正是否保留强故障预测收益并减少正常信息损失；交付9模型、完整配对表、论文公式/图和英文材料。

# Current state
起点d8cfcd5；21模型及结果已验收。复用rq3_joint_prediction_dev_20261029_v1的事件、随机索引、视图日程、标准化器及N/NG/RG/AG等全部模型预测。用户同意继续，保持原评分、数据和模型。

# Scientific assumptions
仍为原温度暴露质量代理和三工况模拟研究。S=exp(−r²/2)，既有告警边界r=1对应S0=exp(−1/2)。只改变融合：lambda=max(0,1−S/S0)，x_tilde=x+lambda*(c(q)−x)。S≥S0保持观测原值；低评分逐渐靠近参考。阈值沿用评分校准，不依据质量结果优化。

# Scope
SG真实评分选择性修正、SR训练池随机评分同类修正、SA去故障假设评分及参考同类修正。所有组公共输入q仍为完整候选。三种子42/43/44，每种子1000train/250val、六视图、hidden32/18维/6689参数、40轮320更新，完全复用上一轮模型/打乱种子。共9模型，不重训旧对照，不新增评分筛查。

# Milestones
1. 配置与本计划先提交；实现src/rq3_selective_prediction_dev.py及边界/等价测试。
2. 运行脚本run入口，固定来源data/a0-root/scorer-root/base-root/config/output/commit/seed；各模型即时保存。验收为共同初始化、顺序、预算与旧组一致，checkpoint重放一致。
3. summarize入口合并旧预测与新预测；逐seed、六场景、通道/切换相位/末端工况给出MAE差；独立重算、生成PNG/PDF和Methods/Results/Discussion材料后推送。

# Data leakage controls
只读原train/val；评分与参考冻结，原训练池随机索引和公共特征不变。标签不进入触发器。故障mask仅用于结果分层。测试集不读取。

# Reproducibility
配置rq3_selective_prediction_dev_20261029.json包含来源清单哈希。新输出拒绝覆盖，旧文件只读。完整保存checkpoint、840轮旧史的来源及360轮新史、输入身份和预测。

# Validation
触发边界、lambda范围、未触发时原值不变、公共字段一致；新旧初始化/日程/容量匹配；重放和独立指标核对。评估效果大小、跨种子趋势和消融，不另设复杂门槛。

# Progress
- [x] 用户授权及冻结单一融合变化。
- [x] 实现及4项触发测试通过。
- [x] 9模型完整完成，三个种子总体优于N/SR/NG。
- [x] 论文材料与图表完成，随本阶段归档及回执核验。

# Decision log
采用上轮提出的无新增调参触发式，不同时搜索其他门控。SG−SR评价针对性，SG−NG评价选择性，SG−SA评价联合故障假设的增量。报告与旧RG的比较但不以不同触发方式的随机组代替SR。

# Surprises and discoveries
SG总体MAE1.59096，相对N降低0.82%；触发机制贡献主要收益，联合故障假设仅见温度子组小幅一致作用。

# Outcomes and retrospective
交付后给出一次完整验证建议及论文同步安排，保持11月28日目标。
