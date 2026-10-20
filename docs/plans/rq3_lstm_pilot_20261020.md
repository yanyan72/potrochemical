# RQ3 LSTM 开发验证执行计划

## Purpose
用户要求今天继续模拟研究、上传原始仓库并提供导师汇报。交付复用既有重建数据的普通LSTM开发验证，保留正负结果；不把合成结果说成真实质量验证。

## Current state
main=19599b170156d0ab0ce24d39364908c5c6e0bdd0；10,000批/84文件已发布，岭回归五种子结果已归档，未见评分稳定增益。生成器不改。当前Python3.14环境缺PyTorch，拟建立独立3.11环境遵循仓库约定。

## Scientific assumptions
沿用180步温度暴露代理/独立null、118步观测、理想正常参考、离散工况和未标定参数。用户确认本轮仅依赖模拟；外部有效性不因此获得证明。详细配置和证据限制见同次RQ3_LSTM_PILOT_PROTOCOL_20261020.md。

## Scope
五数据种子、六个LSTM特征组、两个独立目标、十个val视图、均值与已保存岭回归参照。新过程模拟、训练故障增强、最终test、真实数据均不在今日范围。

## Milestones
1. 输入现有代码/数据/日志；新增协议/配置/计划，更新当前执行指针；推送。验收：正式训练前冻结科学选择。
2. 新增src/rq3_lstm_pilot.py、测试及锁定依赖。运行 python -m pytest -q tests/test_rq3_paired_data.py tests/test_rq3_prediction_pilot.py tests/test_rq3_lstm_pilot.py；小夹具端到端验证。验收：无测试读取、可重复、模型回读与数学测试通过；推送实现。
3. 输入固定manifest与已发布实现；逐seed执行 python -m src.rq3_lstm_pilot --data <data> --ridge <ridge_run> --config configs/rq3_lstm_pilot_20261020.yaml --output results/rq3_prediction/rq3_lstm_pilot_20261020_v1 --implementation-commit <commit> --seed <42..46>。每seed拒绝覆盖已有目录；保存参数/预测/曲线/清单，逐seed推送。完成后 --summarize 汇总。验收：所有5seed模型重放通过，原数据不变。
4. 独立指标复算、远端文件核验；由实测表生成导师审核稿，更新STATUS/DECISIONS等；推送。验收：结果、局限、提交可追溯，文件可交付。

## Data leakage controls
预测仅reference/train/val；test无预测入口。预处理/标签标准化只train；评分参考/校准分开；val原始MAE选checkpoint，压力视图不参与选择；事件只分层，潜在真值不进特征。

## Reproducibility
保留原数据和岭回归清单；源码提交、配置、环境及SHA256入档；单线程CPU确定性PyTorch；模型/打乱种子公式在协议中；每seed完成即落盘上传，已有目录拒绝覆盖。

## Validation
手算LSTM单步、可复现训练和更新、常量通道屏蔽/训练统计、测试入口拒绝、小夹具端到端、模型重放、指标独立复算、逐文件哈希和远端blob核验。

## Progress
- [x] 2026-10-20读取规则和现有证据，确认远端main未变化。
- [x] 科学协议和配置已拟定；正式训练尚未开始。
- [x] 独立Python3.11.17/PyTorch2.14.1环境；37项专项测试通过（23.18秒），实现随本阶段提交后再训练。
- [ ] 五种子拟合、分批持久化及独立复核。
- [ ] 导师文档、项目状态同步与最终验收。

## Decision log
真实数据不作为本轮继续条件。沿用已有样本，避免把重复生成当作研究进展。使用普通LSTM均值读出和等预算组间对比；训练恒定特征在验证时仍置零，避免未训练权重任意响应。平方项和随机分数对照保留。

## Surprises and discoveries
先前环境无PyTorch；创建独立3.11环境，不修改已有3.14运行环境。首次测试因缺已有包导入链依赖pandas而收集失败，补齐后37项通过。正式结果未产生。

## Outcomes and retrospective
待执行；不预先声明成功或增益。
