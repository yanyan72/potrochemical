# 实验团队给论文负责人的第一周证据交付清单

截止日：2026-11-28。本文件是证据及任务接口，不是修改论文主稿。原题目和 RQ1—RQ3 结构保留。章节是否已经写入由论文负责人确认；当前未取得该确认。

## 可直接复用的结果及不能越过的边界

| 对应章节/RQ | 可用事实 | 证据位置（原始项目） | 允许主张/尚缺事项 |
|---|---|---|---|
| Methods：原评分 | 工况参考、边际/条件残差、EWMA、阈值校准及重置已实现 | src/sensor_reliability.py、src/temporal_reliability.py；原方法与实验协议 | 经典残差与 EWMA 不宣称首创；参考为合成干净参考，不称现场标定 |
| Results：RQ1 | 开发 E1b 单一/pooled/regime clean FPR 43.25%/1.25%/1.15%；pooled→regime Recall 32.60%→64.16% | docs/RESULTS_LOGISTICS_E1B.md；docs/PAPER_STAGE_ACCEPTANCE_DETAIL_20261023.md | 覆盖解释主要误报改善；条件化额外收益主要在召回。完整三参考确认对照列入本月必要补充 |
| Results：RQ2 | E1b 边际 Top-1 85.32%；条件残差干净通道牵连 29.22%，边际 1.04% | 同上及 E1c 报告 | Top-1 条件在已知污染行上，不能叫实际检出率。保留牵连和故障强度依赖 |
| Results：切换边界 | E1g 条件 EWMA 跨切换 Recall 15.90%→34.17%，记录恢复窗口 FPR 0.72%→41.18% | docs/RESULTS_LOGISTICS_E1G.md | 保留记忆带来检出与错误记忆传播权衡，不能主张一律取消重置 |
| Results：已完成 RQ3 | 共享故障 100 模型；T−N +0.00850 [−0.00888,+0.02588]，T−R +0.00198 [−0.00832,+0.01227] | docs/RQ3_SHARED_FAULT_TEACHER_REVIEW_20261023.md；results/rq3_prediction/rq3_shared_fault_20261023_v1 | 五数据/训练联合种子的近似 t 区间；稳定增益未获支持，不是“证明无效” |
| Discussion：目标适用性 | arrival_proxy 是完整温度超阈值累计暴露加噪声；观测只有前 118 步 | configs/rq3_paired_rebuild_20261019_v1.yaml；src/rq3_paired_data.py | 非真实质量标签；湿度/振动仅间接相关，不能外推其工业作用 |
| Results：A0 | 延迟 clean FPR：匹配硬组 1.924%，全通道软组 0.763%，留通道 0.879%；延迟＋强偏置正常通道 FPR：1.227%/1.366%/11.875% | docs/RQ3_UNCERTAINTY_A0_REPORT_20261029.md；对应 metrics_by_seed.csv | 三种子开发均值；两候选均未通过冻结门槛，不是 RQ3 预测收益；校准风险分配须说明 |
| Methods：本月候选 | 12 状态共享工况—故障后验，保留原正常参考与评分骨架 | WEEK1_MECHANISM_AND_PROTOCOL_20261029.md、候选配置与后续代码 | 当前是候选数学规格；性能、新颖性与 RQ3 均待验证 |
| Application/SBS | 缺少经核实的过程—样品—终点配对，最小实测尚未实施 | docs/SBS_FEASIBILITY_GATE_20261029.md | 不作为真实预测验证；符合条件再决定是否入正文/附录 |

上述历史数值来自阶段验收与原结果报告，本轮不重复重训。写作引用时以对应 commit 的原表为准，不能把不同开发协议的数字串成持续提升曲线。

## 可复用的英文事实句（证据边界用语，非论文全文）

- **RQ1:** Broader reference coverage accounted for most of the reduction in false alarms, whereas regime conditioning provided additional recall gains in the evaluated development setting.
- **RQ3:** Under the fixed synthetic target and training budget, the registered comparisons did not provide evidence of a stable predictive gain from the trust scores.
- **A0:** Soft regime inference reduced false alarms caused by delayed regime records, while channel-wise leave-one-out inference could propagate contaminated evidence to the references used for healthy channels.
- **Candidate:** We consider a shared posterior over operating regimes and single-channel fault hypotheses; its predictive value remains to be evaluated under matched-information controls.
- **Scope:** All predictive experiments reported so far use a synthetic exposure proxy rather than independently measured industrial product quality.

## 每周交接及验收责任

| 最晚日期 | 本助手交付 | 论文负责人同步任务 | 当前状态 |
|---|---|---|---|
| 11/04 | 历史证据、A0 补充诊断、候选公式/假设/最近邻差异 | 修订前半部分及已完成结果，保留原结构 | 首批材料已准备；正文状态未核实 |
| 11/11 | 最终方法规格、全部实现参数、数据划分及计算预算 | 完成 Methods/Experimental Setup | 待实验阶段完成 |
| 11/18 | 冻结表格、图数据、统计及全量正负结果 | 完成 Results/Discussion | 待执行 |
| 11/25 | 数值/术语核验清单、复现附录、限制 | 完成英文表达、图表与引用整合 | 待执行 |
| 11/28 | 实验最终包和未决事项 | 提交导师可审阅的完整英文初稿 | 目标，非已完成承诺 |

## 必须保留的统计与信息边界

旧 RQ3 的五种子才是外层重复；同批视图不独立，数据与优化方差未分开。新确认设计将采用五数据种子×两优化种子，外层仍为五；两个优化种子只帮助描述训练波动。开发中使用过的验证样本不能叫新盲测。共享参考、已知模拟机制与多轮适应性研究均需披露。

评分是输入与冻结参考的派生表示，不能称新增独立物理信息。真实企业检验结果、假设模拟、真实过程加人工故障必须分别标记。方法公式有依据不等于有新颖性；软件检查通过不等于 RQ3 成立。

## 当前需要导师/写作负责人决定的事项

如果 11/18 仍没有稳定预测收益，需决定是否接受机制与失效边界的贡献定位，及论文题目/摘要哪些预测导向措辞需要限定。这个决定不阻塞本周实验。本助手不会代替导师承诺创新性充分，也不会为保留正向措辞修改实验结果。
