# RQ3 岭回归开发验证执行计划

## Purpose
用户要求按审查建议继续拟合、上传原始项目，并交付供老师审核的Markdown。本阶段交付真实执行的五种子岭回归及消融开发验证结果、代码和审计，严格标注合成代理与验证集结果。

## Current state
已有10,000批重建版，GitHub与本机84文件一致。生成器c3aa06e，数据清单提交2b2049b；此前没有本版预测模型。20项专项测试与结构审计已通过。待修正幅度抽样描述；自动审计尚缺负对照与元数据交叉核验。现有评分为边际EWMA .2/q99/reset。

## Scientific assumptions
沿用抽象时间、环境示意通道、未物理标定温度暴露代理、理想无故障参考。保持原始训练/故障验证的分布差异；分数不是正确概率。B2可用性在当前设定与缺失标识冗余。数据与科学范围详见同次冻结的RQ3_RIDGE_PILOT_PROTOCOL_20261019.md和配置。

## Scope
本轮做岭回归、训练均值、非线性平方项和随机分数对照；两个目标、五种子、十个val视图。普通LSTM、新test最终评价及真实数据验证是后续里程碑，不宣称本轮完成整个预测研究。

## Milestones
1. 输入已有数据和审查；创建协议、配置、本计划，修正文档并提交GitHub。验收：尚未运行正式模型结果，所有实验选择预先记录。
2. 新增src/rq3_release_audit.py、src/rq3_prediction_pilot.py及tests/test_rq3_prediction_pilot.py。命令：pytest -q tests/test_rq3_paired_data.py tests/test_rq3_prediction_pilot.py。测试语义拒绝、手算/库解对照、无测试读取、因果、重复模型和落盘重放。用既有160批夹具烟雾运行，验收后提交实现。
3. 命令：python -m src.rq3_prediction_pilot --data data/processed/rq3_paired_20261019_rebuild_v1 --config configs/rq3_ridge_pilot_20261019.yaml --output results/rq3_prediction/rq3_ridge_pilot_20261019_v1 --implementation-commit <已上传实现提交>。本机数据路径可不同，其内容哈希必须相同。只读审计、五种子实际拟合、模型回读重放及输出manifest通过后上传结果。
4. 输入实测指标，编写供老师审核的Markdown和项目状态/决策，列出正负结果、限制、待审核事项和提交记录；上传并逐文件核验。

## Data leakage controls
模型入口只加载reference_fit/reference_cal/train/val；不提供test预测接口。参考和校准分开，填补/特征标准化/平方项尺度只取train。val原始MAE按同一预算选alpha，故标记开发验证。事件仅用于模拟配对与事后分层，隐藏过程不进入评分/预测。严格审计单独进行，可能读取test做结构核验。

## Reproducibility
数据种子42–46，随机对照901；新目录拒绝覆盖。记录源数据固定清单、生成器外部预期哈希、当前实现提交及实际源码哈希、环境与配置、模型和逐批预测。复用本机Python3.14.6环境，无新增依赖，明确与仓库Python3.11约定差异。

## Validation
相关专项测试、小夹具烟雾运行、严格审计、拟合后重载模型重放每个val预测、每文件SHA-256与远端Git blob核对。预设B1/B2等价检查。检查原数据manifest未变化。

## Progress
- [x] 读取仓库规则与当前源码，完成实验范围和协议设计。
- [x] 协议/配置/文档纠正已推送，提交fb29f501a7eae967600b531f8bb44f6dd795fad3。
- [x] 代码、32项相关测试及临时小样本端到端烟雾验收通过（2.94秒）；本提交保存实现，随后运行正式数据。
- [ ] 五种子拟合、验证及结果持久化。
- [ ] 老师审核文档与项目状态同步。

## Decision log
先以可解释的岭回归开发验证供老师审核；正式保留test在最终协议确定后评价。使用平方项及随机分数对照，检查分数增益的替代解释。相同搜索预算不等于模型容量相同。沿用原评分实现，不根据验证结果调评分参数。

## Surprises and discoveries
边际评分当前可用性与非缺失标识完全重复，提前登记该限制。现有环境没有PyTorch，本阶段不把LSTM写成已经完成。

## Outcomes and retrospective
待执行后据实际结果更新。
