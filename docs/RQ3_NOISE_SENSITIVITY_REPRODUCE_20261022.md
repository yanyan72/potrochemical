# RQ3噪声敏感性产物复核与复现

## 读取已发布结果

正式产物已保存于results/rq3_prediction/rq3_noise_sensitivity_20261022_v1/；共1250基础批次、三档观测。默认优先复核落盘数据，避免重复生成已完成数据。

使用已记录的Python3.11依赖环境（configs/rq3_lstm_environment_20261020.txt）。在完整仓库根目录执行：

```sh
python scripts/verify_rq3_noise_sensitivity.py --run results/rq3_prediction/rq3_noise_sensitivity_20261022_v1 --data data/processed/rq3_paired_20261019_rebuild_v1
python -m pytest -q tests/test_rq3_noise_sensitivity.py
```

本机源代码子集将旧数据保存在根目录rq3_paired_20261019_rebuild_v1，因此此前执行日志使用默认短路径；完整GitHub仓库采用data/processed路径，命令应明确传--data。其余岭回归、LSTM和旧final路径相同。集成测试兼容这两种数据目录；缺少已发布存档时跳过归档模型集成测试，不代表全部验收完成。

## 确有必要时复现

不要向已有结果目录重新运行。选择不存在的新输出目录，保持原配置与原冻结源码；通过--seed逐个运行42—46，协议和实现ID分别为65a53207b7dbf82d9695e93afe20be2d76c5e061及b5e6e467843153e015467ddc18cf5e69ff707dee：

```sh
python -m src.rq3_noise_sensitivity --config configs/rq3_noise_sensitivity_20261022.yaml --output results/rq3_prediction/rq3_noise_sensitivity_reproduction --data data/processed/rq3_paired_20261019_rebuild_v1 --protocol-commit 65a53207b7dbf82d9695e93afe20be2d76c5e061 --implementation-commit b5e6e467843153e015467ddc18cf5e69ff707dee --seed 42
```

其余四seed齐全后，使用相同--config、--output、--data并改为--summarize。拒绝覆盖属于正常保护行为；不要删除失败或完成的产物绕过保护。元数据时间会不同，复现不等于所有文件字节一致；应检查数组、指标及来源。要改机制或样本规模，必须另立协议，不能借用本版本身份。

## 文件含义

- inputs_level_0/1/2.npz：0.5/1/2倍标准差的可见输入，含观测、记录工况、批次ID和可用时间。
- evaluation_only.npz：共享的完整潜在过程、真工况、两项标签噪声、ID和两个目标，只用于评价/审计。
- predictions.npz：轴为架构×噪声档×组×批次×目标；轴标签同时保存。
- metrics_by_seed.csv：MAE、RMSE及相对1倍MAE变化，包含总体和运输工况分层。
- paired_by_seed.csv：B3−B2、B3Q−B2Q、B2R−B2以及相对1倍的差中差。
- 根summary：五seed均值、样本SD、描述性t区间；档位不是独立重复。
- manifest和identity：文件校验、冻结源码、协议/实现提交与运行环境。

正式结果生成后，仅为完整仓库兼容补充测试中的数据路径选择，预测/生成模块未修改。正式实现提交、原测试日志、补充测试日志及所有运行产物均保留。教师报告生成器使用存档上传收据；报告已有时拒绝覆盖，需先指定独立副本或手动审阅更新，不能把重新渲染误当新实验。
