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
3. 复用train，生成新val，按seed训练20模型，保存故障日程/100轮序列预算记录/模型/checkpoint验证并逐seed上传。（每模型实际40轮，100为总模型数；不增加轮数。）
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
