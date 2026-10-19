# RQ3配对数据执行计划

## Purpose
按用户2026-10-19批准的方案，交付与论文B0–B3及原始训练/配对污染评价对应的数据；每个验收阶段上传GitHub。

## Current state
基于434b502评分v0.1。论文已完整写出方法和检测证据，3.6/4.2预测仍为方案。10月18日S1/S2仅作历史原型记录，本轮不静默沿用其训练协议。原目录有两处未知NPZ改动，使用独立worktree保留。

## Scientific assumptions
详见RQ3_PAIRED_DATA_PROTOCOL及configs/rq3_paired_data.yaml。所有参数人为预设，抽象时间、环境测点、无故障参考、温度暴露标签、单事件单通道均为限制。没有真实数据或已证明的预测收益。

## Scope
本轮生成/验证数据和加载器；不增加检测器，不训练预测模型，不打开test计算性能。最后同步论文仓库的数据准备说明，正文数值保持现有证据。

## Milestones
1. 输入论文/既有约束；输出协议、配置、本计划及状态；审核时间、标签与B0–B3对应后提交上传。
2. 实现src/rq3_paired_data.py及tests/test_rq3_paired_data.py；小夹具验证随机独立、因果、配对、标签、只读加载；完整pytest通过后提交上传。
3. 执行python -m src.rq3_paired_data --config configs/rq3_paired_data.yaml --output data/processed/rq3_paired_20261019_v1；输出五种子数据、结构报告、manifest；哈希/数组审计通过后提交上传。论文库增加可追溯数据说明及修订记录，不制造预测结果。

## Data leakage controls
完整批次划分；参考fit/cal互斥；只以reference_fit可见观测计算故障尺度；训练不注入；val/test配对标签不变；可用时间<=119；新test只结构检查，既有最终test不使用。

## Reproducibility
命名空间20261019，seed42–46，split/批次/机制独立随机流；精确配置及源码hash；NPZ压缩原始输入和隐藏真值、CSV目标和稀疏事件。跨次比数组/文本，不把NPZ容器时间戳当数值不一致。

## Validation
配置拒绝非法时间和单位；手算标签/漂移；因果前缀；split/seed独立；原始与副本仅事件范围不同；污染不改变标签；reference/train无注入；导出及重载；全回归；五种子结构/哈希审计。

## Progress
- [x] 方案与配置冻结。
- [x] 生成器与测试：20项新增、113项完整回归通过（22.70秒）；完成后立即上传。
- [x] 用户确认后，以新身份 `rq3_paired_20261019_rebuild_v1` 重新生成正式数据、只读审计，并按种子上传与逐文件核对；见 `../RQ3_REBUILD_RELEASE_20261019.md`。旧原件未找回。
- [ ] 论文库同步：依用户最新指示，本轮仅维护原始项目；论文仓库不在本轮修改范围。原始项目已补齐可引用的数据说明与交接。

## Decision log
固定180步研究终点，不把125–155步物理到达与取样混淆。参考也保留测量噪声；主训练无注入。以稀疏事件重建配对副本节省重复存储。普通Git push未配置认证，使用已连接GitHub API并带expected_sha更新分支。

## Surprises and discoveries
Python3.11.17安装成功。完整回归113项通过。重建副本的掩码与实际修改逐元素一致；小数据两次落盘的全部manifest内容一致。测试集加载默认拒绝，但其公开可读性质不改变，不能称保密盲测。

## Outcomes and retrospective
重建版 10,000 批及 84 个实际文件已在原始项目发布并逐 blob 核对；82 个 manifest 条目和 100 组配对视图审计通过。旧原件仍未找回。本轮无 RQ3 预测或真实数据结果。
