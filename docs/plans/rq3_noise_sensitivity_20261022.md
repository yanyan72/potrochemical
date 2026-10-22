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
- [ ] 五seed生成评价与分批上传。
- [ ] 独立复核、汇总与导师稿推送。

## Decision log
用户授权按旧计划推进有限机制敏感性。先选正常测量噪声单因素局部范围，固定模型可隔离推断分布改变；不能把它描述成每个噪声环境重新训练的比较。真实数据对接不阻塞已授权模拟。

## Surprises and discoveries
待执行后记录，失败及负面结果必须保留。

## Outcomes and retrospective
待执行完成后写入实际结果、提交和剩余范围。
