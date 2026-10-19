# RQ3 配对模拟数据重建版发布记录（2026-10-19）

## 发布身份与保存位置

本次发布的数据集为 **`rq3_paired_20261019_rebuild_v1`**，位置为 [`data/processed/rq3_paired_20261019_rebuild_v1/`](../data/processed/rq3_paired_20261019_rebuild_v1/) 。这是原件未找回后，经用户确认，用仓库已保存的生成器、配置和固定种子**重新生成**的版本。上一轮中断前所报的原始 10,000 批文件未取得，不能声称本版与其逐字节相同。小规模通路样本 `rq3_rebuild_smoke_20261019_v1` 与本正式版分别保留。

本目录共 84 个文件、68,732,575 字节：82 个由 manifest 记录哈希的数据与来源文件，加 `manifest.json` 和 `audit.json`。五个种子 42–46 各 2,000 个独立基础批次；每种子分别为 reference_fit 150、reference_cal 100、train 1,000、val 250、test 500。合计 10,000 批、1,180,000 行预测截止前可见输入、1,800,000 行完整潜在过程。原始/污染副本共享基础批次，不计为额外独立样本。

## 来源与重建环境

- 生成器来自原始项目提交 `c3aa06e3bdcf73dc8fb8188aff18ef8653a5aa52`；`src/rq3_paired_data.py` 的 Git blob SHA 为 `eaa75524234f438c4e8c90280b72d4f60bc39ebf`，SHA-256 为 `c63312c1236bb703c841203bd5300a8c4712f38231ef21741b10b94ede76571b`。未修改算法。
- 重建配置为 [`configs/rq3_paired_rebuild_20261019_v1.yaml`](../configs/rq3_paired_rebuild_20261019_v1.yaml)，Git blob SHA `17cfda1c02303bca6301ff1ff6fc33421ed2899d`。相对原配置仅改 `dataset_id`；数据目录内 `config.yaml` 是运行时保存的配置快照。
- 本机运行环境：Python 3.14.6、NumPy 2.5.3、PyYAML 6.0.3。它与旧运行报告中的 Python 3.11.17 不同。因为以逐 blob 核对的源码副本运行，而非完整 Git checkout，自动生成的 `metadata.json.source_commit` 如实为 `unavailable`；上面的来源提交与源码哈希是单独核实的追溯证据，不能把元数据字段改写为自动记录成功。
- 重建前先完成 20 项 RQ3 专项测试及两次 160 批小规模复现，详见 [小规模验证记录](RQ3_REBUILD_SMALL_VALIDATION_20261019.md)。本正式目录只生成一次，随后对同一目录进行只读审计，没有覆盖或重新生成旧文件。

## 审计与远端核对

独立执行 `python -m src.rq3_paired_data --audit data/processed/rq3_paired_20261019_rebuild_v1`，`audit.json.status=pass`，复核 82 个 manifest 哈希条目、上述批次和时间行数，以及 100 组配对视图。审计范围是结构、身份、标签公式、因果可见时间、配对掩码与完整性；`test_predictive_evaluation=false`。这不是模型预测精度或工业真实性的验收。

按种子提交并推送到原始项目 `main`，每次均核对本机与远端 Git blob SHA：

| 阶段 | 文件数 | Commit |
| --- | ---: | --- |
| seed 42 | 16 | `4aec9fb096aecb31be09ebd795c5804f057d1799` |
| seed 43 | 16 | `8c8b25f7e09d9ce51f1ba813de9a52806d3ee3dd` |
| seed 44 | 16 | `cca36b4e05d61f5cd7f7205453b190e90ca29d11` |
| seed 45 | 16 | `87727795bf2d468cfc05fbda545359ccd7705164` |
| seed 46 | 16 | `302991e2f1f8533bdeae247c31d1210113f9e807` |
| 配置、元数据、总清单、审计 | 4 | `2b2049bafb39f1b6c795c603de1c2e934e1ce2c2` |

发布总目录在 GitHub 上恰有 84 个文件；对这 84 个路径逐项比较远端 Git blob SHA 与本机原件，均一致，且无多余文件。四个根文件的 SHA-256：`config.yaml` `4e714ea570fd35b440257a8b068cdba3cc98f4a78d8d781da71666f384613fea`；`metadata.json` `3f192388ecf488623f02cdaadd7c5d95225826190f2f5bc31cb29c8094ed1633`；`manifest.json` `a2224ee148d8b5d61a5cb24fd225d5d56817fc17ebd37753545c8494d7ca47ed`；`audit.json` `270f36a510961662e55b704c9b11df02b56dd66ee5187ca8ae1de01cf74144cd`。本机重建目录继续保留。

## 论文数据说明与边界

可在论文数据准备部分写明：本研究使用按冻结协议重建的**合成储运环境序列**，每个批次提供预测时点以前 118 个抽象时间步的可见测量，终点标签为人为定义的累计温度暴露代理，单位为任意代理单位；五个固定数据种子共形成 10,000 个独立基础批次。训练组无人工注入；验证/测试组提供同一批次的原始与传感器偏置、漂移、缺失配对视图；完整潜在过程、真工况和注入事件与预测输入分开保存。标签是模拟器构造的代理，不是某种石化产品的化验质量，也不是企业真实测量。原件未找回，因此本版以 `rebuild_v1` 单独标识。

这次只完成数据生成、专项测试、只读结构审计与 GitHub 持久化。**尚未训练 RQ3 质量预测模型，尚未评价预测准确率、B0–B3 预测比较或该新测试集，也没有真实数据验证。** 已公开保存的 test 为流程保留集，不能称为保密盲测。未来预测实验须另行冻结模型、评分和评估协议，并记录模型种子与数据种子；不得从本次审计推出“预测准确率提升”。旧评分组件的既有结果也不等于本 RQ3 预测结果。
