"""Render a teacher-facing report from verified noise sensitivity artifacts."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/rq3_prediction/rq3_noise_sensitivity_20261022_v1'
M=pd.read_csv(RUN/'metrics_summary.csv')
P=pd.read_csv(RUN/'paired_summary.csv')
S=pd.read_csv(RUN/'paired_by_seed.csv')
VERIFY=json.loads((RUN.parent/'rq3_noise_sensitivity_20261022_verification.json').read_text())
assert VERIFY['status']=='pass'
RECEIPTS=[json.loads(x) for x in (ROOT/'results/rq3_prediction/rq3_noise_sensitivity_20261022_upload_receipts.jsonl').read_text().splitlines()]
URL='https://github.com/yanyan72/potrochemical'


def table(headers:list[str],values:list[list]) -> str:
    """Format a Markdown table."""
    return '\n'+'\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in values])+'\n'


def select(frame:pd.DataFrame,**filters) -> pd.Series:
    """Return exactly one filtered output record."""
    q=frame
    for key,value in filters.items():q=q[q[key]==value]
    assert len(q)==1
    return q.iloc[0]


def metric(architecture:str,group:str,factor:float,target:str='arrival_proxy',stratum:str='all') -> pd.Series:
    """Select a seed-aggregated metric."""
    return select(M,architecture=architecture,group=group,noise_factor=factor,target=target,stratum=stratum)


def pair(architecture:str,factor:float,comparison:str='B3-B2',target:str='arrival_proxy',stratum:str='all') -> pd.Series:
    """Select a paired seed-aggregated contrast."""
    return select(P,architecture=architecture,noise_factor=factor,comparison=comparison,target=target,stratum=stratum)


def ms(r:pd.Series,key:str='mae',digits:int=5) -> str:
    """Format mean and sample SD without treating views as replicates."""
    return f'{r[key+"_mean"]:.{digits}f} ± {r[key+"_sd"]:.{digits}f}'


def interval(r:pd.Series,key:str='mae_difference') -> str:
    """Format a descriptive interval."""
    return f'[{r[key+"_ci_low"]:.6f}, {r[key+"_ci_high"]:.6f}]'


factors=[.5,1.,2.]
positive=sum(pair('lstm',f).mae_difference_mean>0 for f in factors)
interaction=pair('lstm',2.).interaction_vs_1x_mean
change_b2=metric('lstm','B2',2.).mae_change_vs_1x_mean
change_b3=metric('lstm','B3',2.).mae_change_vs_1x_mean
text=[f'''# 石化储运研究：测量噪声局部敏感性实验汇报

日期：2026年10月22日  
仓库：[yanyan72/potrochemical]({URL})  
版本：`rq3_noise_sensitivity_20261022_v1`  
数据身份：**新生成的人工模拟敏感性数据；无真实企业或质检数据。**

## 一、今日完成与结果摘要

沿用已有生成器和已训练的岭回归、LSTM模型，新增五个种子、每组250批，共**1,250个基础批次**。同一批次产生三档正常测量噪声观测，共3,750个相关观测版本。只改变测量噪声标准差乘数0.5、1、2；不重新训练，也未再次评价旧保留test。

LSTM在三档中的B3−B2平均MAE差有**{positive}/3档大于0**，正数表示加入评分后误差更高。2倍相对1倍噪声时，B2 MAE变化{change_b2:+.6f}，B3 MAE变化{change_b3:+.6f}，评分增量差的变化为{interaction:+.6f}。具体种子波动和描述性区间见下表，不能仅凭小数正负作成功判定。

三档LSTM的B3−B2描述性95%区间均跨0；相对1倍档的评分增量变化区间也跨0。本轮没有给出明确评分收益或噪声改变评分增量方向的证据，但这不等于证明方法等效。部分误差随噪声增大略微降低，可能包含有限样本及固定模型响应的影响，不能据此宣称增加测量噪声有益。

本轮在已知旧结果后预先固定方案，属于探索性局部敏感性分析。**它没有替代上一轮正式测试，也不能单独证明工业有效性或所有噪声环境下的稳健性。**

## 二、本次模拟了什么

| 项目 | 设置 |
| --- | --- |
| 新随机流 | namespace=20261022；seed=42、43、44、45、46 |
| 样本 | 每seed250个基础批次，三档严格配对 |
| 潜在过程 | 每批180步；仓储→平稳/颠簸运输→仓储 |
| 预测信息 | 每批118步可见温度、相对湿度、振动及工况记录 |
| 唯一变化 | 正常测量噪声标准差统一乘数0.5、1、2 |
| 保持不变 | 同批潜在过程、工况、标签、随机噪声创新、模型参数、评分参考和预处理 |
| 本轮未叠加 | 偏置、漂移、缺失、记录延迟 |

| 噪声档位 | 温度标准差/K | 相对湿度标准差/%RH | 振动标准差/(mm/s) |
| --- | --- | --- | --- |
| 0.5倍 | 0.05 | 0.10 | 0.0025 |
| 1倍 | 0.10 | 0.20 | 0.0050 |
| 2倍 | 0.20 | 0.40 | 0.0100 |

这些噪声相对原潜在过程波动较小；此次是有限范围的局部检查，不能外推至强噪声或故障污染。随机发生器沿用旧代码，通过新命名空间避免与旧10,000批ID重合。三档总计442,500行可见观测，只有147,500个不同的批次—时间位置；共享潜在过程225,000行。不得把相关副本算成独立样本扩大量。

目标仍为`arrival_proxy = 10 + 0.03 × Σ max(潜在温度−298 K,0) + 两项模拟噪声`，累计180步，包含已发生及未来暴露；固定终点不等同于模拟到达时刻。它是任意单位代理，不是实测产品质量。`null_proxy ~ N(25,5²)`是独立负对照。所有参数未按具体产品或现场仪表标定，时间为抽象步。

## 三、比较与统计口径

B0=观测+缺失标识；B2再加工况和评分可用性；B3再加连续评分。B2Q/B3Q分别增加观测平方项；B2R增加来自训练集的随机评分轨迹；mean是训练标签均值。两架构均保留全部七组、两个目标。

沿用原训练所得标准化、填补、参考校准和模型参数。岭回归正则系数、LSTM所选轮次未改变。LSTM仍为单层32隐单元、可见隐状态均值读出；恒定训练特征屏蔽等旧限制不变。三档同一批次使用相同随机评分轨迹。隐藏过程及标签单独落盘，预测函数只接受可见输入；生成标签不等于把标签传入模型。

核心配对为`Δ(f)=MAE_B3(f)−MAE_B2(f)`；噪声对评分增量的影响描述为`Δ(f)−Δ(1)`。另列各组`MAE(f)−MAE(1)`。正的Δ表示评分组误差更高；负的差中差只表示相对1倍档变化更有利，不能直接写成评分组优于基础组。

先在每个seed内计算批次平均，再按五个模型—新数据组合汇总均值、样本SD和df=4的95%t区间。三档不是三次独立重复；区间全部为描述性，依赖小样本近似正态等假设，未作多重检验校正。不能将档位/子组挑选为新主检验。

## 四、实际结果

### 4.1 基础组与评分组：全部档位
''']
for arch,label in [('lstm','LSTM'),('ridge','岭回归')]:
    text += [f'\n\n#### {label}\n',table(['噪声标准差乘数','B2 MAE均值±SD','B3 MAE均值±SD','B3−B2均值±SD','差的描述性95%区间'],[[f,ms(metric(arch,'B2',f)),ms(metric(arch,'B3',f)),ms(pair(arch,f),'mae_difference'),interval(pair(arch,f))] for f in factors])]
text += ['''

### 4.2 相对1倍档的变化

以下全是同seed、同批次的配对变化；1倍档变化为0，不作为额外证据。
''',table(['架构','档位','B2 MAE变化','B3 MAE变化','评分增量变化Δ(f)−Δ(1)','变化的描述性95%区间'],[[a,f,f"{metric(a,'B2',f).mae_change_vs_1x_mean:+.6f}",f"{metric(a,'B3',f).mae_change_vs_1x_mean:+.6f}",ms(pair(a,f),'interaction_vs_1x',6),interval(pair(a,f),'interaction_vs_1x')] for a in ['lstm','ridge'] for f in [.5,2.]]),'''

### 4.3 LSTM主配对的逐种子结果
''',table(['种子','0.5倍B3−B2','1倍B3−B2','2倍B3−B2'],[[seed]+[f"{select(S,seed=seed,architecture='lstm',noise_factor=f,comparison='B3-B2',target='arrival_proxy',stratum='all').mae_difference:+.6f}" for f in factors] for seed in [42,43,44,45,46]]),'''

### 4.4 其他输入组与非线性/随机对照

下表为全部组的MAE均值；各组SD、RMSE及全部描述性区间随CSV保存。
''',table(['架构','输入组','0.5倍MAE','1倍MAE','2倍MAE'],[[a,g]+[f'{metric(a,g,f).mae_mean:.5f}' for f in factors] for a in ['lstm','ridge'] for g in ['mean','B0','B2','B3','B2Q','B3Q','B2R']]),table(['架构','配对比较','0.5倍MAE差','1倍MAE差','2倍MAE差'],[[a,c]+[f'{pair(a,f,c).mae_difference_mean:+.6f}' for f in factors] for a in ['lstm','ridge'] for c in ['B3Q-B2Q','B2R-B2']]),'''

### 4.5 截止运输工况与负对照
''',table(['架构','截止工况','0.5倍B3−B2','1倍B3−B2','2倍B3−B2'],[[a,s]+[f'{pair(a,f,stratum=s).mae_difference_mean:+.6f}' for f in factors] for a in ['lstm','ridge'] for s in ['transit_smooth','transit_rough']]),'''

所有批次的预测截止均处于运输中，因此没有伪造仓储截止子组。无注入故障，也没有故障定位性能可报告。

负对照null的MAE均值：
''',table(['架构','组','0.5倍','1倍','2倍'],[[a,g]+[f'{metric(a,g,f,target="null_proxy").mae_mean:.5f}' for f in factors] for a in ['lstm','ridge'] for g in ['mean','B2','B3']]),'''

负对照按设计独立于输入，有限样本差异不应解释为发现可预测质量信号，也不能单凭这张表证明无泄漏。

## 五、校验、保存和提交

实现前先推送协议；9项专项测试通过，其中临时小样本集成使用独立namespace，不计入正式1250批。新入口对五seed、两架构的全部原始val预测重放最大差为0。正式数据和预测按seed生成、落盘并上传，无重新拟合。独立复核不调用本轮预测或汇总代码：从随机流重建测量噪声、两项标签误差及null标签，复算全部结果。
''',table(['检查','实际结果'],[['状态',VERIFY['status']],['独立基础批次',VERIFY['unique_batches']],['配对观测版本',VERIFY['paired_versions']],['输出清单条目',VERIFY['output_files_hashed']],['指标逐行重算',VERIFY['metric_rows_recomputed']],['配对差逐行重算',VERIFY['paired_rows_recomputed']],['汇总逐行重算',VERIFY['summary_rows_recomputed']],['噪声重建最大残差',f"{VERIFY['maximum_independent_noise_residual']:.3e}"],['标签公式及随机流',VERIFY['label_formula_and_random_streams']]]),f'''

完整产物目录：`results/rq3_prediction/rq3_noise_sensitivity_20261022_v1/`。每seed含三档inputs、evaluation_only隐藏过程与标签、predictions、指标、配对差、metadata及manifest。根目录含配置、生成器基础配置、源码/提交身份、五seed汇总和清单。新数据是敏感性版本，不覆盖旧重建数据或旧正式测试。

manifest SHA-256：`{VERIFY['manifest_sha256']}`。

| 阶段 | Commit |
| --- | --- |
''']
text[-1]=text[-1].rstrip()
text += [f"| {r['message']} | [{r['commit']}]({URL}/commit/{r['commit']}) |" for r in RECEIPTS]
text += ['''

本报告文档提交ID见本轮最终交付消息；逐文件远端核验另存记录。

## 六、与上一轮及论文的关系

上一轮已发布的冻结LSTM十条件测试主端点为B3−B2=+0.062576，95%区间[0.005464,0.119688]，当前方案总体预测增益不获支持。这个正式结果保持不变。本轮仅为正常观测、局部测量噪声变化，样本和比较范围不同，不能把两轮指标直接相减当作改进，也不能用本轮较有利子组替换上一轮结论。

论文目前可以报告：完成了一个明确标记为合成的配对基准、冻结预测比较，以及正常测量噪声范围内的局部敏感性分析；具体正负效果应以上表为准。不能写成真实产品质量提高、全部故障下稳健或已有企业验证。

真实数据目前仍在沟通阶段，没有已接入数据。后续可按预先有限预算研究时间相关性等另一机制因素；强噪声、训练分布匹配、故障增强和评分门控/加权/修复仍未完成，需要另立协议，不能将本轮描述性分析包装成全面验证。

请老师重点审核：代理目标与业务目标的距离、当前负面结果应如何限定贡献、下一轮机制因素的优先级。研究推进不以出现正面增益作为结束条件。
''']
report='\n'.join(text).strip()+'\n'
path=ROOT/'docs/RQ3_NOISE_SENSITIVITY_TEACHER_REVIEW_20261022.md'
dest=ROOT.parents[1]/'outputs/石化储运RQ3_测量噪声敏感性与导师汇报_20261022.md'
assert not path.exists() and not dest.exists(), 'Do not overwrite a prior report.'
path.write_text(report);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(report)
print(json.dumps({'report':str(path),'deliverable':str(dest)},ensure_ascii=False))
