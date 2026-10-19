# RQ3 重建小规模验证与保存方案（2026-10-19）

## 原件定位结论

旧对话可读取记录未提供正式数据附件或产物路径；本机 Documents/Codex、Desktop、Downloads 与临时目录未找到 `rq3_paired_20261019_v1`、其 `manifest.json`、`audit.json` 或压缩包。GitHub `main` 在本阶段前也没有该目录。故旧对话所报 10,000 批与 82 文件审计仍是未取得原件的执行报告；下述测试是**新执行的小规模重建验证**。

## 源码与测试

从原始项目提交 `c3aa06e3bdcf73dc8fb8188aff18ef8653a5aa52` 取回 `src/rq3_paired_data.py`、`src/simulator.py`、`src/__init__.py`、配置与 `tests/test_rq3_paired_data.py`。本地 `git hash-object` 与 GitHub 对应 blob SHA 逐项一致；生成器为 `eaa75524234f438c4e8c90280b72d4f60bc39ebf`，原配置为 `3aa5e0e7c1c51933f818fc7f0310a3a67d092754`，测试为 `23c311326f72eddaeb7460509382c3dd25272890`。没有修改生成器算法。

现有本机环境为 Python 3.14.6、NumPy 2.5.3、PyYAML 6.0.3、pytest 8.4.2，与旧提交记录的 Python 3.11.17 不同。运行单独 RQ3 测试模块，**20 项通过（1.27 秒）**。这证明本机环境中的代码行为通过这些检查，不证明旧运行的字节身份；未运行全部项目回归。

## 两次落盘复现

保留原协议其他设置，仅将种子缩为 42、43，五个 split 各缩为 16 批。两次独立运行均通过原生成器 `write_dataset` 及后续只读 `audit_dataset`：各 160 批、18,880 行可见输入、28,800 行潜在过程，manifest 含 34 个数据与来源文件，完成 40 组配对视图检查。审计覆盖逐文件哈希、批次 ID、标签公式、可见时间、原始/污染配对掩码、记录延迟及预测输入与评价真值隔离；`test_predictive_evaluation=false`。两次 manifest 的 SHA-256 均为 `4defd003c8d8583ddcddb9fba4ed64a28136e406203df834022446beca151b1c`，审计报告 SHA-256 均为 `d4e87560b8b7535257fbb636971b536096c610f2f4f2b8f9f7eb44347dc737e5`。另直接逐数组对比 seed 42/train 与 seed 43/test 的重新生成结果，完全一致。

新增 `configs/rq3_paired_rebuild_20261019_v1.yaml`，仅将原配置 `dataset_id` 改为 `rq3_paired_20261019_rebuild_v1`，原始配置保持不变。对两个种子/split 的数组、目标及事件进行对比，标记只改变发布身份，不改变采样内容。按此标记另生成并审核了一份 160 批样本，存于 `data/processed/rq3_rebuild_smoke_20261019_v1/`；其 34 个 manifest 条目中，只有 `config.yaml` 与 `metadata.json` 相比未标记的小样本变化。该目录含 36 个实际文件（34 个清单条目加 `manifest.json`、`audit.json`），均已作为本阶段提交保存。上传前对每个 Git blob SHA 与本地 `git hash-object` 比对，远端提交后应再次确认完整路径集合与 blob SHA。

这份 GitHub 样本仅用于结构和发布通路验收，**不是正式 10,000 批数据**，不用于拟合或评价预测器。由于小样本运行目录不是完整 Git checkout，其自动 `metadata.json` 中 `source_commit` 为 `unavailable`；本文件通过源码 blob、配置 blob 和运行环境记录实际来源，不能把该字段解释成已由生成器自动记录提交身份。

## 正式重建前的保存规则

正式重建需用户确认后才启动。使用新 `dataset_id` 和全量原种子/计数，在本机 Documents/Codex 下的**全新目录**落盘，保留原始目录与本阶段样本，绝不覆盖。生成结束后先只读重审所有文件并复核 82 个 manifest 条目、100 组视图检查、10,000 个独立批次及时间行数；这些是验收目标，不是当前正式结果。

上传按五个种子分批提交到原始项目，每批核对远端路径和 Git blob SHA；最后提交配置、元数据、manifest、audit 和发布说明，再核对完整远端树。上传全部验收通过前，GitHub 目录明确标为部分发布；本机原件保留到远端复核完成之后。若任一文件上传或远端哈希不一致，停止发布结论并保留失败状态。全程不训练预测器、不读取测试集计算预测指标、不把合成代理或重建版本称为真实数据或上一轮原件。
