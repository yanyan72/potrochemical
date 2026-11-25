# 论文中英术语表与翻译约定

更新日期：2026-11-25

用于石化产品储运传感器级可信度估计论文的中英翻译。本文档记录项目用词约定，不代表 TIMC 官方强制术语，也不构成新的方法或实验结论。

## 1. 使用规则

- 首次出现时定义核心概念，后文使用同一英文名称；标题、摘要、正文、图表和补充材料保持一致。
- 可信度表示观测与当前工况预期行为的一致程度；不得译成测量正确概率、产品质量分数或设备寿命可靠性。
- 翻译保持原文的研究范围、方法定义、数字、结论强度和适用条件。模拟证据保留 synthetic 或 simulation 限定。
- 介绍已有文献时尊重原研究的术语与定义，不将文献中的 reliability 一律替换成 trustworthiness。
- 表中的“不建议”指本文同一概念的用词选择，并非断言这些表达在其他研究中错误。
- 改动术语时同步检查全文、缩写及符号解释，保留代码和冻结产物中的既有字段名。

## 2. 核心概念

| 中文 | 推荐英文 | 使用边界与不建议混用的表达 |
| --- | --- | --- |
| 石化产品储运 | petrochemical product storage and transportation | 首次明确 product，避免与生产过程混同 |
| 传感器级可信度估计 | sensor-level trustworthiness estimation | 方法与任务名称；不交替使用 credibility estimation 或 reliability estimation |
| 可信度分数 | trust score | 数值输出；不轮换使用 credibility score、reliability score 或 quality score |
| 观测一致性 | observation consistency | 可信度的操作性含义；不是另一种独立输出 |
| 可靠性 | reliability | 文献或确有可靠性含义时使用；不作为本文可信度的同义词 |
| 工况 | operating regime；后文简称 regime | 首次结合仓储、平稳运输、颠簸运输解释；不为同一潜在状态轮换使用 mode、condition、state |
| 工况感知 | regime-aware | 总体方法特征；本身不承诺自动识别未知工况 |
| 工况条件化 | regime-conditioned | 指按工况建立、选择或融合参考，或按工况校准；具体动作须说明 |
| 工况适配 | regime adaptation | 仅用于确有迁移、域适配或在线适应的机制 |
| 合法工况变化 | legitimate regime changes | 不意味着产品质量合格、环境安全或无风险 |
| 工况记录 | regime record | 可见记录，与实际工况区分 |
| 工况记录滞后 | delay in regime records；delayed regime records | 说明滞后步数；与状态空间大小区分 |
| 工况推断 | regime inference | 联合统计模型内的推断，不等于可靠识别物理故障原因 |
| 共享工况参考 | shared regime reference | 同一工况推断结果供多个通道评价使用 |

## 3. 观测、参考与评分

| 中文 | 推荐英文 | 使用边界与不建议混用的表达 |
| --- | --- | --- |
| 传感器观测／读数 | sensor observation / sensor reading | 讲建模输入用 observation，讲具体数值可用 reading |
| 测量污染 | measurement contamination | 与产品污染区分 |
| 干净观测／未受污染观测 | uncontaminated observation | 可在定义后简称 clean observation；仍可能包含正常测量噪声 |
| 潜在干净过程量 | latent uncontaminated process value | 与包含正常测量噪声的干净观测区分 |
| 未受污染通道 | unaffected channel / uncontaminated channel | 事件中未被注入污染的通道；不暗示其永远无故障 |
| 工况条件化参考 | regime-conditioned reference | 与合并参考保持相同工况覆盖时，才讨论条件化设计差异 |
| 单一工况参考 | single-regime reference | 只由一个合法工况拟合 |
| 合并参考 | pooled reference | 全部已覆盖工况合并拟合的一组参考 |
| 参考覆盖 | reference coverage | 区分覆盖不足与条件化作用 |
| 边际残差 | marginal residual | 相对单通道边际参考的偏离 |
| 条件残差 | conditional residual | 使用其他通道的条件信息；与 regime-conditioned 区分 |
| 混合参考残差 | mixture-reference residual | 工况权重依赖当前观测，不据此宣称标准正态性 |
| 因果时间累积 | causal temporal accumulation | causal 指仅使用当前及过去信息，不表示因果效应识别 |
| 指数加权移动平均 | exponentially weighted moving average (EWMA) | 首次给出全称 |
| 分数校准 | score calibration | 经验尺度和阈值校准；不是测量正确概率的概率校准 |
| 全局／分工况校准 | global / regime-specific calibration | 全局校准仍逐通道拟合阈值 |
| 接受区域／接受边界 | acceptance region / acceptance boundary | 与评分阈值及修正触发规则保持一致 |

## 4. 故障解释、定位与反馈风险

| 中文 | 推荐英文 | 使用边界与不建议混用的表达 |
| --- | --- | --- |
| 单通道／多通道污染 | single-channel / multi-channel contamination | 描述受污染观测，不直接等同于已确认的物理故障 |
| 并发传感器故障 | concurrent sensor faults | 强调多个通道同时发生故障 |
| 双故障 | dual-sensor faults | 明确是两个传感器，而非一个传感器的两种故障类型 |
| 故障组合 | fault combinations | 三个通道的故障掩码组合 |
| 可组合的故障解释 | modeling fault combinations / allowing concurrent fault hypotheses | 按句子选择具体动作；不机械使用 combinatorial fault interpretation |
| 工况与故障组合联合推断 | joint inference over operating regimes and fault combinations | 说明模型推断对象；不声称由此消除参考污染 |
| 故障方差膨胀 | covariance inflation under a fault hypothesis | 解释作用于故障通道的方差；按实际协方差公式表述 |
| 故障定位 | fault localization | 指通道定位，不等于已诊断物理根因 |
| 正常通道牵连误报 | false alarms on unaffected channels | 优先于含义较隐晦的 collateral false alarms；报告率时注明事件期间的分母 |
| 工况权重 | regime weights | 对应 q_t(k)；确需后验表述时注明是在工作模型内的 filtered posterior |
| 内生反馈 | endogenous feedback | 当前观测影响评价自身的参考 |
| 异常自我掩盖 | anomaly self-masking | 定义为污染改变工况权重、残差缩小并可能导致漏检；不暗示该风险已被排除 |
| 原12状态分支 | original 12-state branch | 三工况 × 无故障或三种单通道故障；用于既有 RQ3 预测确认 |
| 完整24状态分支 | full 24-state branch | 三工况 × 八种故障掩码；状态可表达的组合不等于全部经过验证 |
| 限制组合12状态消融 | restricted 12-state ablation of the combinatorial model | 从24状态模型裁剪并重新归一化；不等同于原12状态分支 |
| 记录滞后12步 | a 12-step delay in regime records | step 为采样步；与 12-state、24-state 区分 |

## 5. 修正、预测与比较

| 中文 | 推荐英文 | 使用边界与不建议混用的表达 |
| --- | --- | --- |
| 选择性观测修正 | selective observation correction | 接受区域内保留原值，越界后按评分修正 |
| 评分指导修正 | score-guided correction | 表明评分如何被使用，与附加评分特征区分 |
| 连续修正 | continuous correction | 本文 NG 对照，强度为 1-S；此处 continuous 不指连续时间 |
| 参考中心 | reference center | 明确是由工况权重融合的中心 |
| 向参考收缩 | shrinkage toward the reference | 可能减少污染误差，也可能引入参考偏差 |
| 附加评分特征 | trust scores as additional features | 不与选择性修正合并为同一验证结果 |
| 无评分修正对照（N） | no-correction baseline | 正式修正比较中保留原始观测，仍可获得共同工况信息 |
| 随机评分修正对照（SR） | randomized-score correction baseline | 随机评分替代真实轨迹，采用相同选择性规则 |
| 去故障假设消融 | ablation without fault hypotheses | 按实际实现说明方差膨胀被移除及阈值独立校准 |
| 未来温度暴露代理预测 | prediction of a future temperature-exposure proxy | 保留未来目标与 proxy 限定，不译成已验证真实产品质量预测 |
| 当前质量软测量 | current-quality soft sensing | 目标时刻等于预测时刻；与未来预测区分 |
| 质量化验结果 | laboratory quality measurement / assay result | 同时区分取样时刻和结果可用时刻 |
| 匹配预测器比较 | comparison using matched predictors | 同类结构、信息与预算匹配；各组分别训练，不意味着使用完全相同参数 |
| 配对模拟序列 | paired synthetic sequences | 共享基础过程的多个版本，不作为额外独立样本 |
| 独立模拟测试 | independent simulation test | 按实际协议使用；不升级为真实工业验证 |
| 平均绝对误差 | mean absolute error (MAE) | 相对下降百分比与绝对差值分开 |
| 召回率 | recall | 与事件内至少一次检出的事件指标区分 |
| 误报率 | false-positive rate (FPR) | 明确干净副本、事件内正常通道或其他统计分母 |

## 6. 结论强度与表达

| 要表达的含义 | 建议英文措辞 | 应避免的扩大 |
| --- | --- | --- |
| 减轻污染对参考的干扰 | mitigate the influence of contamination on the shared regime reference | eliminate / prevent interference，除非有相应证据 |
| 特定场景下提高检出 | improve detection under the evaluated conditions | universal / comprehensive improvement |
| 模拟结果支持 | simulation results support / show | proves industrial effectiveness |
| 允许并发故障解释 | allow concurrent fault hypotheses | guarantees correct fault identification |
| 未排除自我掩盖 | anomaly self-masking cannot be ruled out | self-masking has been resolved |
| 统计显著性 | statistically significant | 仅在相应检验支持时使用；不得只因均值较低而添加 |
| 24状态预测验证边界 | the 24-state branch has not been evaluated for downstream prediction in RQ3 | 将 RQ2 检测改善直接归为24状态预测收益 |

短句优先说明主语、动作和对象。However、therefore 等连接词只用于真实逻辑关系，不逐句添加。可使用 We propose、The method combines、Simulation results show 等自然表达，不统一改成被动语态。

## 7. 来源与维护

依据项目《写作skill.md》既定术语，以及2026-11-25本轮翻译讨论整理。这里只保存翻译约定，不含论文正文。TIMC 的投稿格式要求应另查其[官方作者指南](https://journals.sagepub.com/author-instructions/TIM)，不能把本术语表的措辞偏好当成期刊硬性规定。

后续若用户或导师确认新译法，更新相应条目并同步检查中英文稿；引用文献的专有术语和冻结代码字段保留原名。
