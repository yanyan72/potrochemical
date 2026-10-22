# 测量噪声敏感性：正式生成前验证

2026-10-22；协议提交65a53207b7dbf82d9695e93afe20be2d76c5e061。正式新1250批尚未生成，本页为实现阶段验收。

## 已执行

- `../lstm-venv/bin/python -m pytest -q tests/test_rq3_noise_sensitivity.py`：9项通过，5.32秒。包括两种子各8批的两次配对生成；新namespace2026102201；三档truth/targets/IDs一致、噪声线性比例容差1e-12。
- 实际归档模型小规模集成在临时目录使用另一namespace2026102202、两种子各8批，无正式样本混用。禁止原fit函数，验证只载入旧train构造随机评分银行，完成三档推断、存盘、汇总和独立复核；拒绝覆盖已完成seed。夹具不计入正式研究样本量。
- 差中差手算：B3−B2为[1,2,4]时，相对1倍档变化[-1,0,2]；五seed汇总检验重复seed拒绝和已知t区间。
- `../lstm-venv/bin/python -m src.rq3_noise_sensitivity --config configs/rq3_noise_sensitivity_20261022.yaml --output results/rq3_prediction/rq3_noise_sensitivity_20261022_val_replay.json --replay`：五种子两架构、所有七组两目标原始val预测与旧存档最大差均0，无重新拟合或旧test预测。
- 对新增模块、独立复核器和测试编译检查通过；审查新增代码未编辑旧生成器或模型模块。

测试与重放输出分别保存于results/rq3_prediction/rq3_noise_sensitivity_20261022_tests.log、rq3_noise_sensitivity_20261022_val_replay.json及rq3_noise_sensitivity_20261022_val_replay.log。

独立复核器不导入本轮预测/汇总代码，通过固定随机流重建测量噪声、两项标签噪声及null目标，从落盘预测直接重算所有指标、配对变化和描述性区间。此验证不证明真实物理规律成立。

初次配置准备误用了不带PyYAML的系统Python，未创建配置即报ModuleNotFoundError；随即改用既有Python3.11虚拟环境完成。没有因此修改依赖、基础数据或实验设计。

后续仅在本实现提交后生成五seed正式敏感性样本，结果无论方向均保留。
