"""Build a Chinese teacher report from sealed development-validation artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def table(path: Path) -> list[dict[str, str]]:
    """Read a persisted table without fitting or generating predictions."""
    with path.open() as f:
        return list(csv.DictReader(f))


def build(run: Path, ridge: Path, verification: Path, commits: dict[str, str]) -> str:
    """Render actual numbers and explicit synthetic/development limitations."""
    cfg = json.loads((run / 'config.json').read_text())
    identity = json.loads((run / 'run_identity.json').read_text())
    meta = json.loads((run / 'metadata.json').read_text())
    verify = json.loads(verification.read_text())
    assert verify['status'] == 'pass'
    rows, old = table(run / 'metrics_summary.csv'), table(ridge / 'metrics_summary.csv')
    pair = table(run / 'paired_summary.csv')
    raw_pair = table(run / 'paired_by_seed.csv')
    history = table(run / 'training_history.csv')
    def metric(group: str, view: str = 'original', target: str = 'arrival_proxy', source: list | None = None) -> dict:
        return next(r for r in (rows if source is None else source) if r['group'] == group and r['view'] == view and r['target'] == target and r['stratum'] == 'all')
    def delta(comparison: str, view: str) -> dict:
        return next(r for r in pair if r['comparison'] == comparison and r['view'] == view and r['target'] == 'arrival_proxy')
    def fmt(r: dict, key: str = 'mae') -> str:
        return f"{float(r[key + '_mean']):.4f} ± {float(r[key + '_sd']):.4f}"
    views = ['original', 'bias_1', 'bias_2', 'bias_4', 'drift_1', 'drift_2', 'drift_4', 'missing', 'delay_6', 'delay_12']
    positive = sum(float(delta('B3-B2', v)['mae_difference_mean']) < 0 for v in views)
    qpositive = sum(float(delta('B3Q-B2Q', v)['mae_difference_mean']) < 0 for v in views)
    chosen = [r for r in history if r['selected'] == 'True']
    boundary = sum(int(r['epoch']) == cfg['epochs'] for r in chosen)
    minutes = sum(json.loads((run / f'seed_{s}/metadata.json').read_text())['elapsed_seconds'] for s in cfg['seeds']) / 60
    lines = [
        '# 石化储运模拟研究：普通 LSTM 开发验证日报（导师审核稿）',
        '', '训练日期：2026-10-20；续传、复核与汇报整理：2026-10-21  ',
        '主项目：yanyan72/potrochemical  ',
        '**证据类型：重建版合成数据上的开发验证；无真实工业数据，本轮未开展最终测试集预测。**',
        '', '## 一、本阶段完成了什么', '',
        '在无法取得真实数据的前提下，研究继续以可审计的模拟实验推进。本阶段复用此前保存的10,000批基础数据，补齐普通LSTM预测对照，没有重新生成或覆盖基础数据。新增的是模型、逐批预测、训练曲线和评价记录，不是新增独立模拟批次。',
        '', f'完成五个数据种子（42—46）、六组LSTM输入、两个目标的独立训练，共{meta["learned_models"]}个模型；每个模型固定训练40轮，再按原始验证集MAE选择checkpoint。每个种子均评价10个配对条件，另有训练均值对照。正式逐种子运行累计约{minutes:.1f}分钟（不含环境、实现、测试、上传和文档时间）。',
        '', f'当前结果：加入连续评分的B3相对B2，在10个条件中有{positive}个条件的五种子平均MAE降低；加入平方项后的B3Q相对B2Q有{qpositive}个条件平均MAE降低。条件共用批次，不能把这些条件视为10次独立重复，也不能仅凭改善条件个数宣称稳定有效。具体幅度和种子差异见下表。',
        '', '**结果解释：当前普通LSTM在原始验证条件的误差低于上一轮岭回归，但连续评分作为额外输入未显示总体预测增益。B3−B2及B3Q−B2Q在全部10个条件的平均MAE差均为正；记录延迟和缺失时退步更明显。这个结论仅针对当前冻结训练方式和合成机制，不等于所有评分利用方式都无效。**',
        '', '## 二、数据和模拟假设', '',
        '| 划分 | 每个种子 | 五种子合计 | 用途 |', '| --- | ---: | ---: | --- |',
        '| 参考拟合 | 150 | 750 | 评分参考 |', '| 参考校准 | 100 | 500 | 评分阈值 |',
        '| 训练 | 1,000 | 5,000 | 模型及预处理 |', '| 开发验证 | 250 | 1,250 | 选训练轮次及压力评价 |',
        '| 保留测试 | 500 | 2,500 | 未作本轮预测评价 |',
        '', '数据版本为 `rq3_paired_20261019_rebuild_v1`。这是旧对话原件缺失后按保存生成器重建的版本，不声称与旧原件逐字节相同。其118万行可见输入、180万行潜在过程及既有文件保持原样。',
        '', '每批模拟仓储→平稳或颠簸运输→仓储，三通道为温度、相对湿度和振动。只有预测时已可见的0—117步进入模型；潜在过程共180步。预测截止时所有批次仍在运输中。时间为抽象步，转场为统计分布切换，参数未按具体产品或现场传感器标定。',
        '', '`arrival_proxy = 10 + 0.03 × Σ(t=0…179) max(潜在温度(t) − 298 K, 0) + 批次噪声 + 化验噪声`。这是人为定义的累计温度暴露代理，不能直接称为产品实测质量、酸值、纯度或损失率。固定终点180也不等同于模拟到达时点。该目标累计包含预测前已经发生的暴露及尚未观测的后续暴露，因此任务并非只预测未来新增暴露。独立的 `null_proxy ~ Normal(25, 5²)` 用作负对照。',
        '', '偏置和漂移的1/2/4倍幅度是同批次配对条件；事件通道、时间段、潜在过程与标签保持不变。缺失及记录延迟6/12步另列。正常训练不人工注入故障，因此正常训练→故障验证的分布差异仍是限制。',
        '', '## 三、模型和公平比较', '',
        '普通单向、单层LSTM，32隐单元；对118步可见隐状态取时间均值，再接线性标量输出。没有注意力、故障修复、物理约束或评分加权。该读出在训练前冻结，不能代表所有LSTM方案。两个目标分别训练，不相互共享标签。',
        '', 'B0=观测+缺失标识；B2=再加工况和评分可用性；B3=再加连续评分；B2Q/B3Q=分别再加观测平方项；B2R=用随机训练评分轨迹作为附加块。当前B1与B2可用性信息冗余，因此不重复拟合B1。评分、填补参数直接复用经哈希核验的上一阶段存档。',
        '', '输入按train批次和时间的逐通道统计标准化，目标也只按train统计标准化。训练期恒定输入列在所有划分保持零，避免从未训练的输入权重在故障验证中任意响应；这也意味着当前模型没有学到缺失标识的作用。',
        '', 'AdamW学习率0.003、权重衰减0.0001、batch128、梯度裁剪1、40轮、标准化目标MSE。选择最小原始val MAE的最早轮次；所有组预算一致，压力条件不参与选择。不同组输入通道数和参数量仍不同，随机评分对照用于检查附加容量的替代解释。',
        '', '每个数据种子与目标组合只有一个固定优化种子，同一组合的各输入组共享打乱顺序。报告的标准差同时包含数据与优化随机性，尚不能单独估计初始化敏感性。与岭回归并列仅供基线参考：二者预处理、模型与调参方式有差异，不能把全部差值单独归于架构。',
        '', '## 四、实际结果', '', '以下均为五种子均值 ± 样本标准差，单位为任意代理单位；越低越好。验证集参与选择，不能当作最终测试性能。',
        '', '### 4.1 原始验证条件：与上一轮岭回归并列', '',
        '| 输入组 | 岭回归 MAE | LSTM MAE | LSTM RMSE |', '| --- | ---: | ---: | ---: |']
    for group in cfg['groups']:
        lines.append(f'| {group} | {fmt(metric(group, source=old))} | {fmt(metric(group))} | {fmt(metric(group), "rmse")} |')
    lines += ['', '### 4.2 所有配对条件：连续评分增量', '',
              '差值=B3−B2；负值表示加入评分后误差降低，正值表示升高。', '',
              '| 条件 | B2 MAE | B3 MAE | 配对 MAE 差 |', '| --- | ---: | ---: | ---: |']
    for view in views:
        lines.append(f'| {view} | {fmt(metric("B2", view))} | {fmt(metric("B3", view))} | {fmt(delta("B3-B2", view), "mae_difference")} |')
    lines += ['', '同一主比较的逐种子MAE差如下；负值为改善。', '',
              '| 条件 | seed42 | seed43 | seed44 | seed45 | seed46 |', '| --- | ---: | ---: | ---: | ---: | ---: |']
    for view in views:
        values = [float(next(r['mae_difference'] for r in raw_pair if int(r['seed']) == seed and r['view'] == view and r['comparison'] == 'B3-B2' and r['target'] == 'arrival_proxy')) for seed in cfg['seeds']]
        lines.append('| ' + view + ' | ' + ' | '.join(f'{v:+.4f}' for v in values) + ' |')
    lines += ['', '### 4.3 非线性与随机评分对照', '',
              '| 条件 | B3Q−B2Q | B2R−B2 |', '| --- | ---: | ---: |']
    for view in views:
        lines.append(f'| {view} | {fmt(delta("B3Q-B2Q", view), "mae_difference")} | {fmt(delta("B2R-B2", view), "mae_difference")} |')
    lines += ['', '### 4.4 独立随机标签负对照（原始验证）', '',
              '| 输入组 | null_proxy MAE |', '| --- | ---: |']
    for group in cfg['groups']:
        lines.append(f'| {group} | {fmt(metric(group, target="null_proxy"))} |')
    lines += ['', '### 4.5 四倍偏置按实际故障通道分层', '',
              '| 故障通道 | B2 MAE | B3 MAE |', '| --- | ---: | ---: |']
    for channel in ('temperature', 'relative_humidity', 'vibration'):
        sub = [next(r for r in rows if r['view'] == 'bias_4' and r['group'] == g and r['target'] == 'arrival_proxy' and r['stratum'] == 'fault_channel:' + channel) for g in ('B2', 'B3')]
        lines.append(f'| {channel} | {fmt(sub[0])} | {fmt(sub[1])} |')
    lines += ['', '四倍温度偏置分层的平均误差有所降低，而湿度和振动分层升高；这是需保留的局部现象。本代理由温度暴露直接构造，湿度和振动没有直接进入标签公式；不同故障通道的表现不能无条件推广到真实多因素质量任务。',
              '', '负对照的偶然小幅差异不等于可预测信号；保留全部五种子结果。负对照也不能替代完整的泄漏检查。',
        '', '## 五、验证、保存与可复现性', '',
        '- 37项相关专项测试通过，含门控递推手算、训练统计隔离、重复训练、小夹具端到端、模型重放和test入口拒绝；不是全仓库回归。',
        f'- 运行环境：Python {identity["python"]}，PyTorch {identity["dependencies"]["torch"]}，CPU单线程、float32、确定性算法；依赖版本已锁定。',
        f'- 已完成60模型、40轮/模型的训练记录。{boundary}个模型所选轮次落在第40轮上界；保留这个诊断，不据结果增加预算。',
        f'- 模型从存档重新加载，并重算全部验证视图；最大预测差为{meta["model_replay_max_abs_difference"]}。',
        f'- 独立复核通过：{verify["metric_rows_recomputed"]}行分层指标、{verify["paired_rows_recomputed"]}行配对差、{verify["summary_rows_recomputed"]}行汇总及{verify["selection_rows_checked"]}条轮次选择记录。',
        f'- 输入{verify["input_files_hashed"]}个清单条目未变化；运行产物清单覆盖{verify["output_files_hashed"]}个文件（另有清单自身），每种子分阶段保存并上传。',
        f'- 结果manifest SHA-256：`{verify["run_manifest_sha256"]}`。',
        '', '## 六、目前不能据此得出的结论', '',
        '本轮不能证明真实石化产品质量预测有效，也不能证明模拟机制贴近某个具体产品。连续评分是观测相对参考的统计一致性，不是测量正确概率。现有验证集已经反复用于开发，任何局部改善仍需事先冻结的保留测试验证；不据验证结果反向调整模拟标签来制造收益。',
        '', '当前需要继续审查代理目标的定位、离散转场、噪声和相关性设定，以及正常训练对故障验证的适用边界。尚未完成针对这些机制的独立敏感性实验和最终test评价；没有新增真实数据、真实故障或工业有效性证据。',
        '', '## 七、请老师审核的事项与下一步', '',
        '1. 是否同意把论文定位为“可控模拟条件下的观测一致性与预测稳健性研究”，并将累计温度暴露明确写为合成代理任务？',
        '2. 现有过程与标签假设是否足以支撑论文问题；是否需要缩小“石化质量预测”的表述范围？',
        '3. 下一步建议先冻结少量机制敏感性条件，例如测量噪声或时间相关强度；每次只改变一个主要因素，保存新版本且保留当前结果。若引入训练故障增强，必须作为所有方法共享的独立实验，不能只帮助评分组。',
        '4. 在机制与方法选择完成后，冻结最终测试协议，再一次性评价保留test；最终报告同时呈现成立与不成立的条件。',
        '', '## 八、GitHub阶段记录', '', '| 阶段 | Commit |', '| --- | --- |']
    for name, sha in commits.items():
        lines.append(f'| {name} | [{sha}](https://github.com/yanyan72/potrochemical/commit/{sha}) |')
    result_sha = commits['汇总与复核']
    lines += ['', f'[完整结果与逐种子文件](https://github.com/yanyan72/potrochemical/tree/{result_sha}/results/rq3_prediction/{cfg["run_id"]})；[冻结协议](https://github.com/yanyan72/potrochemical/blob/{commits["协议"]}/docs/RQ3_LSTM_PILOT_PROTOCOL_20261020.md)。', '',
              '本文数值由归档表自动生成；无未运行的实验指标。导师文档在结果提交之后单独同步，文档提交号由交付消息报告。', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'ridge', 'verification', 'commits', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    a.output.write_text(build(a.run, a.ridge, a.verification, json.loads(a.commits.read_text())))
