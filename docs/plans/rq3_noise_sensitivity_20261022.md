# RQ3测量噪声敏感性执行计划

## Purpose
用户继续模拟、分阶段上传并取得导师MD。交付五seed新配对样本、冻结模型敏感性结果和完整审计，不追求预定正向增益。

## Current state
main=c7f325242ca23d5a6f1d4281cb6dab1400299170，旧10000批、岭回归与LSTM存档、2500批保留测试已发布。旧test已使用，LSTM主端点+0.062576，95%区间[0.005464,0.119688]。本机work/rq3-rebuild-check为源码/产物子集；Python运行时../lstm-venv/bin/python（3.11.17）。

## Scientific assumptions
全部为合成过程及累计温度暴露代理，非真实质量。正常测量噪声统一乘数0.5/1/2是唯一因素；新namespace20261022、五seed各250，共1250基础批次，三档共用潜在过程与标签。模型冻结；当前小噪声范围不能代表强噪声或全机制稳健。完整边界见RQ3_NOISE_SENSITIVITY_PROTOCOL_20261022.md。

## Scope
新正常观测、不注入故障/延迟；两架构七组两目标。无重新拟合、无旧test再评价、无新训练策略，结果为探索性描述。

## Milestones
1. 输入旧存档和已知负面结果；新增配置/协议/本计划并更新当前指针，提交GitHub；验收协议先于新正式数据。
2. 新增src/rq3_noise_sensitivity.py、tests/test_rq3_noise_sensitivity.py、scripts/verify_rq3_noise_sensitivity.py。运行../lstm-venv/bin/python -m pytest -q tests/test_rq3_noise_sensitivity.py及相关旧测试；小样本双次复现、单因素/标签/ID/拒覆盖/数学/禁止fit检查；--replay验证旧val一致。提交实现和验证记录。
3. 运行python -m src.rq3_noise_sensitivity --config configs/rq3_noise_sensitivity_20261022.yaml --output results/rq3_prediction/rq3_noise_sensitivity_20261022_v1 --protocol-commit <id> --implementation-commit <id> --seed <42..46>，每个seed新目录拒覆盖；按seed提交并核验源数据和新产物。命令默认数据/模型路径与现有目录一致，真实运行用上述venv Python。
4. 同入口--summarize汇总，独立verify脚本重算、逐远端blob核对、自动生成导师MD并更新项目文档，提交；验收五seed齐全、源不变、全部结果和限制保留。

## Data leakage controls
预测只接受INPUT_KEYS；标签/truth分文件保存，模型推断不接收。评分/填补/标准化来自旧训练存档，随机银行仅旧train。新三档使用同批ID配对；旧test仅哈希核对，禁止在本模块预测。

## Reproducibility
固定namespace/seed和噪声配置；记录协议、实现commit、依赖版本、源码和来源manifest。CPU单线程确定性。各seed保存立即上传，断线按完成标记恢复；不覆盖既有目录或重新生成完成seed。

## Validation
手算误差和差中差；两次小样本逐数组相同；档间truth/标签/ID一致及噪声比例；新旧ID不相交；val重放匹配；禁止fit；独立复算分层/配对/汇总和原标签公式；源/输出hash与远端blob核验。

## Progress
- [x] 核查main和旧模型/数据/结果路径；读相关实现与项目规则。
- [x] 协议与配置已推送65a53207b7dbf82d9695e93afe20be2d76c5e061。
- [x] 9项测试通过，五seed两架构原始val重放差0；实现与验收记录随本阶段提交。
- [x] 五seed1250批生成评价并逐seed上传，提交08a6f51e、33ae4ea4、4e58a596、d643ff02、8b4b69fd。
- [x] 独立复核及汇总已推送7d45a902；导师稿和项目状态随本交付阶段提交。

## Decision log
用户授权按旧计划推进有限机制敏感性。先选正常测量噪声单因素局部范围，固定模型可隔离推断分布改变；不能把它描述成每个噪声环境重新训练的比较。真实数据对接不阻塞已授权模拟。

## Surprises and discoveries
初次配置准备使用系统Python缺少PyYAML，未生成任何正式文件，改回既有虚拟环境后完成。三档LSTM评分差均值均为正，但描述性区间跨0；噪声增大时绝对误差略降，保留结果而不解释为噪声有益。正式生成后仅补充测试的完整仓库数据路径兼容，冻结预测/生成实现未修改。

## Outcomes and retrospective
新增正常测量噪声局部敏感性实验，namespace=20261022，五seed各250批，共1250基础批次及3750配对观测版本；噪声标准差乘数0.5/1/2，潜在过程、工况、标签和模型参数不变。本轮无人工故障、无记录延迟、无重新训练，旧RQ3 test未再次预测，仍保持已使用状态。

LSTM三档B3−B2平均MAE差为+0.013681、+0.013353、+0.012740，描述性95%区间均跨0；2倍相对1倍的评分增量变化−0.000613，区间[−0.002754,0.001527]。当前小范围内变化较小，没有明确评分收益或方向变化证据；不能解释为等效、强噪声稳健或工业有效。部分绝对误差略降不表示增加测量噪声有益。上一轮正式test负面结论不变。

协议65a53207b7dbf82d9695e93afe20be2d76c5e061和实现b5e6e467843153e015467ddc18cf5e69ff707dee均先于本轮正式生成。9项测试通过，五seed两架构原始val重放差0；独立重建测量噪声/标签随机流并复算1260行指标、540行配对差、360行汇总通过。汇总提交7d45a902d8e1c3d51390b9cb736d35d8e88ae24f，源文件保持一致。详见RQ3_NOISE_SENSITIVITY_TEACHER_REVIEW_20261022.md与RQ3_NOISE_SENSITIVITY_REPRODUCE_20261022.md。

真实企业数据仍在沟通阶段，尚未接入，不作为本轮模拟来源。后续时间相关性等其他机制、强噪声、训练分布匹配、故障增强及新评分利用方式尚未完成。以下为各日期历史记录。
