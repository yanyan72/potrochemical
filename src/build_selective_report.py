"""Build concise research report and complete seed/scenario tables from artifacts."""
import csv,json,statistics
from pathlib import Path

root=Path('results/rq3_selective_prediction_dev_20261029_v1')
read=lambda p:list(csv.DictReader(p.open()))
metrics=read(root/'metrics_by_seed.csv');pairs=read(root/'paired_by_seed.csv');suite=read(root/'suite_by_seed.csv')
views=['correct_clean','correct_drift1','correct_bias4','delay12_clean','delay12_drift1','delay12_bias4'];names=['正确记录·干净','正确记录·弱漂移','正确记录·强偏置','延迟记录·干净','延迟记录·弱漂移','延迟记录·强偏置']
get=lambda seed,arm,view:float(next(r['mae'] for r in metrics if r['seed']==str(seed) and r['arm']==arm and r['view']==view and r['stratum']=='all'))
mean=lambda arm,view:statistics.mean(get(s,arm,view) for s in [42,43,44])
text='''# RQ3：9模型选择性修正实验与论文材料交付

2026-10-29。结论：**选择性修正实现了本轮希望验证的权衡改善——保留强故障收益，大幅减少正常观测损失，并在三个数据种子上取得总体预测改善。建议冻结当前候选，进入一次完整验证。**

## 结果概览

复用原数据、标签、Trust Score、共同工况后验和LSTM架构，只将全时段修正改为越过既有评分边界后才介入。新训练SG（真实评分）、SR（随机评分）、SA（去故障假设）各三个种子，共9模型。N、NG、RG、AG及原拼接模型直接复用上一轮存档。

| 对照 | 六场景等权MAE均值 | SG相对该组的MAE差 | 三个种子总体改善数 |
|---|---:|---:|---:|
'''
sg=statistics.mean(float(r['mae']) for r in suite if r['arm']=='SG')
for arm,label in [('N','无评分 N'),('SR','随机评分选择性修正 SR'),('RG','随机评分全时段修正 RG'),('NG','真实评分全时段修正 NG'),('SA','去故障假设选择性修正 SA')]:
    v=statistics.mean(float(r['mae']) for r in suite if r['arm']==arm);delta=[float(next(r['mae_difference'] for r in pairs if r['seed']==str(s) and r['comparison']==f'SG-{arm}' and r['view']=='suite')) for s in [42,43,44]]
    text+=f'| {label} | {v:.5f} | {sg-v:+.5f} | {sum(d<0 for d in delta)}/3 |\n'
text+='''
SG总体MAE为 **1.59096**，相对N降低 **0.82%**，相对SR降低 **0.69%**，相对原NG降低 **1.28%**。主要价值是改善方向在三个数据种子重复，且有与设计目标一致的分场景解释；目前属于开发支持，正式留出确认尚待完成。

## 六场景比较

差值为SG减对照，负数表示SG更好；数值为三个数据种子均值，所有条件完整保留。

| 场景 | N | SG | SG−N | SG−SR | SG−RG | SG−NG | SG−SA |
|---|---:|---:|---:|---:|---:|---:|---:|
'''
for v,n in zip(views,names):text+='| '+n+' | '+f'{mean("N",v):.5f} | {mean("SG",v):.5f} | '+' | '.join(f'{mean("SG",v)-mean(a,v):+.5f}' for a in ['N','SR','RG','NG','SA'])+' |\n'
text+='''
强偏置场景SG相对N降低 **2.66% / 2.99%**（正确/延迟记录），相对N和SR在三个种子均改善。干净场景损失从原NG的 **2.01% / 1.69%** 缩小至 **0.06% / 0.32%**；弱漂移损失也由 **2.26% / 1.83%** 缩小至 **0.24% / 0.39%**。这说明无需提高弱漂移检出率，仅改变修正策略就能显著改善预测的收益—代价关系。

## 各数据种子：总体与完整场景

| 种子 | N | SG | SR | NG | SA | SG−N | SG−SR | SG−NG |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
'''
for s in [42,43,44]:
    vals={a:float(next(r['mae'] for r in suite if r['seed']==str(s) and r['arm']==a)) for a in ['N','SG','SR','NG','SA']}
    text+=f'| {s} | '+' | '.join(f'{vals[a]:.5f}' for a in ['N','SG','SR','NG','SA'])+' | '+' | '.join(f'{vals["SG"]-vals[a]:+.5f}' for a in ['N','SR','NG'])+' |\n'
text+='\n| 种子 | 场景 | SG−N | SG−SR | SG−RG | SG−NG | SG−SA |\n|---|---|---:|---:|---:|---:|---:|\n'
for s in [42,43,44]:
    for v,n in zip(views,names):text+=f'| {s} | {n} | '+' | '.join(f'{get(s,"SG",v)-get(s,a,v):+.5f}' for a in ['N','SR','RG','NG','SA'])+' |\n'
text+='''
## 机制发现与创新解释

**1. 主要增量来自减少不必要干预。** 正确记录的干净温度观测，仅约0.80%的点触发修正，平均修正强度从原全时段的6.25%降至0.081%。强温度偏置窗口仍有66.67% / 82.21%的点触发，平均强度37.00% / 48.58%。算法把干预集中到了异常证据更强的区域，与预测结果中正常代价下降、强故障收益保留一致。

**2. 温度仍是主要预测收益来源，湿度损失基本消退。** SG−N在强温度故障下为−0.15877 / −0.16562，三个种子均改善；湿度为−0.00104 / −0.00614，从原NG的明显退步降至接近零或小幅改善；振动仍为+0.02817 / +0.02412，虽低于原NG代价，仍值得作为任务相关性案例分析。原目标是温度暴露代理，不修改标签制造其他通道收益。

**3. 联合推断与触发机制应分层归因。** SG−SA总体仅+0.00036，没有联合故障假设带来的总体额外收益；但强温度故障下为−0.00173 / −0.00266，三个种子均有小幅改善。可以保留这条局部机制线索，正式验证中继续设置SA，不把整个0.82%的总体改善归给联合推断。共同后验q已提供给所有组，因此这里比较的是评分与修正分支的增量。

**4. 可深化的贡献是一个连贯链条。** RQ1确定工况条件化参考；不确定工况下的联合推断增强参考选择；RQ2评价传感器一致性和正常通道牵连；RQ3通过选择性修正把可靠性证据转化为有条件的预测增益。核心框架无需重写，新增的是原分数与预测器之间的可解释使用规则。

## Methods核心公式

沿用原绝对平滑残差比r及Trust Score：

`S_tj = exp(−r_tj²/2),  S0 = exp(−1/2)`。

共同工况参考 `c_tj = Σ_z q_t(z) μ_zj`，其中均值换回原传感器单位。选择性修正为：

`λ_tj = max(0, 1−S_tj/S0)`；

`x̃_tj = x_tj + λ_tj(c_tj−x_tj)`。

当r≤1，λ=0、观测严格保持；当r>1，修正连续增强，0≤λ≤1，输出是观测与参考的凸组合。触发边界由已有评分校准给定，没有通过质量标签重新调阈值。整个映射沿用因果评分和工况过滤，不读取未来观测。

相对配对无注入观测x⁰，令e=x−x⁰、b=c−x⁰，则：

`(x̃−x⁰)²=(1−λ)²e²+λ²b²+2λ(1−λ)eb`。

该分解解释污染削减与参考替代偏差的权衡；它不是任意过程下必然改善的定理。

## 直接供论文整合的英文材料

独立文件 `docs/RQ3_SELECTIVE_PAPER_MATERIALS_20261029.md` 提供Methods、Results、Discussion段落及三张英文图题；主稿、题目和原RQ1—RQ3不改写。图表为PNG预览及矢量PDF，可直接用于排版。

## 完整验证与写作同步建议

**现在冻结选择性修正，不继续搜门控形式或阈值。** 建议一次正式主实验：5个数据种子×2个独立优化种子×N/NG/SG/SR/SA五组，共50模型；原train和val复用，最终冻结后一次生成新命名空间的5×500独立测试批次。六场景等权作为主评价，预定delay6干净/强偏置作次要压力评价。另用原null_proxy做5种子×N/SG/SR三组，15模型负对照，总65个新增模型；连同前21+本9，累计95个研究拟合，仍低于原106预算。

主比较SG−N、SG−SR；SG−NG检验选择性，SG−SA检验联合故障假设。先在每数据种子内平均优化重复，再以5个数据种子计算配对均值、SD及df=4的近似t区间，保留各优化重复与剔除单种子敏感性。既有1%幅度作为参考量报告，不再把它当唯一科研价值门槛；真实效果及不确定性一起解释。不要把2500批次或六场景当独立训练重复。

本轮开发证据足以支持进入这次确认，但不替代它。继续保留弱漂移和振动子组，以检验收益是否只集中在温度强故障。确认后不再依据测试挑参数；若联合模块增量仍弱，论文分别表述其评分价值与选择性修正的预测价值。

时间建议：尽快冻结上述正式协议和候选；11月11日前完成训练与实验设置交付，11月18日前完成新留出、必要RQ1对照及统计冻结，11月19—25日复现和论文图文核对，11月26—28日提交完整证据包。当前立刻可写Methods及开发Results；正式结果只需替换/补充确认表，不等全部实验结束再写。

RQ1复用single/pooled/regime框架并补齐原计划必要确认性对照；RQ2复用A0及联合推断诊断，不再全面重做评分搜索。SBS独立按既有最小实测方案推进，当前没有新的已确认过程—检验配对，继续不阻塞主线。

## 执行与复现

9模型共360轮、2880次更新，训练合计约85.9秒；保留全部checkpoint和逐批预测。4项触发专项测试通过，独立重算1440行分层指标、1029行配对差、343行配对汇总，旧预测和源文件哈希一致。共同初始化、打乱顺序、视图日程、容量和预算与上一轮严格匹配。当前结果为3个开发数据种子、1个优化种子的模拟研究，尚无新留出或企业真实预测结论。

协议：`89642020260f12f985c5e7cee12700db83d1bde8`；实现：`1a0995b68ca5b3777c1cfdc494d9f9fbb7d40ccd`。结果目录 `results/rq3_prediction/rq3_selective_prediction_dev_20261029_v1/`；图目录 `results/rq3_prediction/rq3_selective_prediction_figures_20261029_v1/`。本轮只完成已授权9模型，上述65模型为下一轮完整验证建议，尚未运行。
'''
Path('docs/RQ3_SELECTIVE_FINDINGS_20261029.md').write_text(text)
Path('../../RQ3选择性修正9模型_结果与论文推进建议_20261029.md').write_text(text)
