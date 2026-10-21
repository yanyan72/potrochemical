# RQ3 冻结模型保留测试执行计划

## Purpose
按用户授权取得一版可检验RQ3的正式模拟测试结果，上传potrochemical并交付导师Markdown。评价已存模型，无新增基础模拟批次或重新训练。

## Current state
main=370ff5906f7cc0f5bca6495b1c51a03f1f81e753。五种子岭回归与60个LSTM模型已存，数据10,000批不变；开发结果无总体评分增益。RQ3 test只有结构审计，未作预测评价。已有Python3.11.17/PyTorch2.14.1环境可复用。

## Scientific assumptions
沿用抽象储运过程、118步输入、180步温度暴露代理和独立null；无真实质量数据。原始train无人工故障、test配对故障。主终点为LSTM B3−B2十视图等权MAE差，只有五个独立seed，95%t区间假设及次要比较限制详见RQ3_FINAL_TEST_PROTOCOL_20261021.md。

## Scope
两个冻结模型家族×7组×2目标×5seed×10视图，2500基础test批次；不重新拟合、修复、调参或生成正式数据。机制敏感性后续另立范围。

## Milestones
1. 输入已发布目录/哈希；新增协议、配置、计划及当前指针；提交后验收test尚未预测。
2. 新增src/rq3_final_test.py和tests/test_rq3_final_test.py；运行相关pytest、数学/隔离夹具、python -m src.rq3_final_test --validate-only ...；输出val重放报告，验收不读test值、无fit、重放差<=1e-6，提交实现。
3. 输入冻结实现/协议commit；运行python -m src.rq3_final_test --execute-frozen-test --seed <42..46> ...；先保存test使用标记，随后只读推断；各seed新目录拒绝覆盖并分批推送。验收全部预测有限、源数据/模型不变、ID对齐。
4. --summarize汇总，独立脚本复算，输入输出远端hash核对；生成导师报告、更新STATUS/DECISIONS/test已使用，推送。验收五seed齐全、无结果遗漏、结论受主终点及科学范围限制。

## Data leakage controls
预测函数不接收标签；test标签/事件仅在预测存档后读入。参考/校准/标准化/超参数/checkpoint只加载既有训练/开发存档，不调用fit。val重放是实现校验，不重新选择。原开发guard保持拒绝test。

## Reproducibility
外部固定三个manifest；记录源码、环境、协议/实现commit、配置、逐seed完成标记与产物清单；CPU单线程确定性LSTM；随机轨迹对照与原规则相同。遇到中断保留已使用状态，已有产物不覆盖。

## Validation
手算配对套件及t区间、显式test授权/标签对齐guard、隔离小夹具端到端推断、禁止fit、正式val存档重放、独立预测算术复核、原模型/数据不变、远端逐blob确认。

## Progress
- [x] 读取当前状态与源代码，确认main及RQ3 test未预测记录。
- [x] 协议配置已推送82d3fff44bba01628a250af31f765b0f8c2cdfdb。
- [x] 42项专项测试通过；新增独立复核集成再次执行5项测试；正式val五seed两架构重放差全部0，实现随本阶段推送后再执行test。
- [ ] 五seed test推断与上传。
- [ ] 汇总、复核、导师稿与状态同步。

## Decision log
用户授权前述正式检验安排，直接推进当前冻结版本test；不将更广的机制敏感性作为本版结论的虚构前提。唯一主终点为预定十视图等权套件，原始/压力/子组作为次要描述；不能把等权模拟压力解释为现场频率。

## Surprises and discoveries
待执行记录。此前负面val结果已知；不因此更换目标或挑选有利test条件。

## Outcomes and retrospective
待执行，完成后更新实际结果和未完成范围。
