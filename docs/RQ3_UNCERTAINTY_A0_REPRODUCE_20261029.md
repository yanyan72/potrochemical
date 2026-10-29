# A0 复现与存档说明

冻结版本：实现 4806543b32337a8ae681befec3117fef221425f5；正式结果 eeb5d454f50127bce5459dfa49985e6f5ba3dcb1。运行需要 Python、NumPy、SciPy，正式精确版本见结果 identity.json。无需训练深度模型，不需要企业数据。

在原始项目仓库根目录运行（输出必须是不存在的新目录）：

```bash
python -m unittest discover -s tests -p 'test_rq3_uncertainty_a0.py' -v
python src/rq3_uncertainty_a0.py --data data/processed/rq3_paired_20261019_rebuild_v1 --legacy-root . --config configs/rq3_uncertainty_a0_20261029.json --source-git configs/a0_source_git_blobs.json --output results/a0_local_replay_NEW --commit 4806543b32337a8ae681befec3117fef221425f5
python src/verify_rq3_uncertainty_a0.py --root results/rq3_prediction/rq3_uncertainty_a0_20261029_v1 --output results/a0_verification_NEW
```

--commit 标记所用实现版本，若修改实现必须另用对应提交并明确不同于本次冻结运行。默认开发样本数量、故障参数和门槛由已提交配置固定；不读取旧 test。输入逐文件与来源 Git blob/旧数据清单核对。新结果目录拒绝覆盖。

原始评分及输入仍从已存档位置读取，未重新生成潜在过程。sources 清单中的本机绝对路径只记录原运行来源，复现入口不要求同一绝对路径。浮点数组 NPZ 的压缩容器字节可能随环境变化，复现比较以数组/告警和指标为准，不以新压缩文件与旧容器字节相同作为必要条件。

verification 目录另存事后模式/正常通道牵连诊断；它不参与模型调整。结果发布回执见 results/rq3_prediction/rq3_uncertainty_a0_20261029_remote_verification.json。主结果 64 项清单、自身清单及其他材料由发布清单与 Git blob 核验共同覆盖。
