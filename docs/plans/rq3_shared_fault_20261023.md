# ExecPlan：共享故障训练与有限归因

## Purpose
执行用户批准的公平对照，回答评分增量、通道差异、弱漂移及归因边界。

## Current state
旧配对诊断完成，main=a38f49913d9d62f53fc3bff7dc86e9bea3e71917；旧测试已使用。本轮尚未拟合或生成新val/test。

## Scientific assumptions
全部合成，目标为温度累计暴露代理；确定性评分的增量是有限模型下的表示收益。理想故障指示仅用于机制诊断。

## Scope
严格执行RQ3_SHARED_FAULT_PROTOCOL_20261023.md和configs/rq3_shared_fault_20261023.yaml；100模型上限，不搜索、不重复旧数据。

## Milestones
1. 保存协议、配置并上传。
2. 新模块src/rq3_shared_fault.py，公平性/门控/保存重放/禁用测试门测试；小样本验证通过，保存日志并上传实现。
3. 复用train，生成新val，按seed训练20模型，保存故障日程/固定预算记录/模型/checkpoint验证并逐seed上传。
4. 全部模型锁定后生成新test并一次评价，逐seed保存与上传；独立重算指标、主区间和公平性记录。
5. 汇总teacher Markdown、状态、决定与复现说明；上传全部结果，核对旧成果哈希。

## Data leakage controls
train-only标准化/随机轨迹池；旧参考冻结；新val选择模型，新test只在100模型锁定后生成。主模型只收允许的观测特征；理想mask仅O/OG。保存split与ID，标签不入X，配对标签不变。

## Reproducibility
五固定seed、namespace、输入与源文件SHA256，CPU单线程确定性，固定模型及shuffle种子、40轮视图日程；文件存在拒绝覆盖，完成seed可恢复未开始seed。

## Validation
手算门控、共享公共输入、无自身随机评分、同初始化/日程/预算、train-only scaler、test阶段门、两次小样本确定性、模型落盘重放、独立评价算术、远端blob校验。

## Progress
2026-10-23：协议制定，正式实验尚未开始。

## Decision log
主比较N/T/R，理想指示和单一门控为事前固定诊断；两项主差均区间上界<0且全seed负才支持稳定增益。

## Surprises and discoveries
暂无本轮结果。旧正常训练随机评分池不适合作为故障训练的唯一随机基线，本轮按各条件训练分布取池并排除自身。

## Outcomes and retrospective
待执行。完成固定预算后停止扩展。

## 阶段执行与验收细则

- 协议阶段：输入旧诊断与用户授权；输出冻结MD/YAML；GitHub逐blob核对。已提交42db3d131f41dea07c5f27c9b30f509b7861daa0。
- 实现阶段：输入冻结配置；运行 `python -m pytest -q tests/test_rq3_shared_fault.py`；输出测试日志；要求共享输入/种子/预算、数学公式、测试门和两次小训练逐元素可复现全部通过。
- 训练阶段：`python -m src.rq3_shared_fault train --seed 42 --commit IMPLEMENTATION_SHA`（42至46）；输入旧train/参考，输出每seed新val、20模型、800行轮次日志与manifest；要求320步/模型及重放差0。
- 评价阶段：`python -m src.rq3_shared_fault evaluate --seed 42 --commit IMPLEMENTATION_SHA`（42至46），全部训练manifest锁定后执行；输出新test与全组预测/指标；要求ID不相交与输入逐组一致。
- 汇总阶段：`python -m src.rq3_shared_fault summary --commit IMPLEMENTATION_SHA`；运行独立核验脚本；输出主结论/全指标/导师稿/远端证据；要求五seed全收录、主准则不变、所有旧文件不变。

## 2026-10-23 实现进度

- [x] 协议提交42db3d13，先于研究训练。
- [x] 固定算法模块、独立算术核验脚本与9项专项测试已编写。首次收集因一个括号错误失败，已修复并保留日志；小样本缺少通道时跳过空分层，避免NaN指标，未改研究协议。
- [x] 两次相同微型训练参数/预测逐元素一致；共享输入、oracle隔离、随机自身排除、预算和模型选择、训练锁与拒绝覆盖测试通过。隔离小样本完整20模型训练/新测试流水线已通过。
- [ ] 正式五seed训练、测试与独立指标核验。

微型fixture仅软件验证，非RQ3研究结果；不作为100个正式模型以外的择优候选。保留所有测试尝试日志。

## 测试前隔离纠正（已完成100模型，未查看研究test指标）

烟雾测试此前触及namespace20261023中seed42的前8个test样本（其中4个进入微型预测流水线），虽然没有用于选模/数据调整，也未向研究者显示指标，仍不应称该命名空间完全未打开。最终test统一改为2026102301，冻结独立评价模块；不重训、不改变数据规律/故障分布/指标/主准则。原训练源与产物不变。此前软件测试原始日志保留，fixture以后用20990101。

- [x] 五seed100模型完成；每模型6305参数、320优化步，重放差0，逐seed已上传。
- [ ] 使用新入口 `python -m src.rq3_shared_fault_fresh_test --seed 42 --commit dd23e11d9f31bffe16f70737e5f41357b2bf8de5 --evaluation-commit EVALUATOR_SHA`（42至46）。
- [ ] 检查新test ID中的namespace为2026102301；旧训练数据/模型和val保持原哈希。
