# SESSION_HANDOFF

更新2026-09-22。读取AGENTS、PROJECT_BRIEF、ARCHITECTURE、DATA_SPEC、EXPERIMENT_PLAN、TASKS、DECISIONS、STATUS、当前ExecPlan。
用户确认运输仓储主场景、生产Future Work、真实数据后续，并再次授权继续推进和上传GitHub；不需重复征求开发/推送许可。

## 最新里程碑

E1g已交付：实现88a7c1d，72测试通过，5种子×12场景×8组合，正式目录results/logistics_e1g/logistics_e1g_20260922_v1。旧EWMA重放240次、固定状态不变140次，源209/177和输出79哈希通过。跨切换条件Recall15.90%→34.17%，延迟记录更新后clean FPR0.72%→41.18%；保留reset工程默认，carry实验选项。test/质量未评价，详见RESULTS_LOGISTICS_E1G。2026-09-22用户设备授权后，实现和结果c63b408已上传，git ls-remote确认远端c63b408a2f85f5edc268eaad25f0737cb59e724c；随后提交交付记录并在最终核对后清理临时认证文件。

用户已明确项目与论文分开；新增PROJECT_ACCEPTANCE。下一步评分组件统一批量入口、模型/字段校验及使用示例，不再把不断增加检测器当完成标准。质量任务仍需产品/指标/时间定义。

## E1f里程碑

E1f已交付：实现1376b49，65测试通过，5种子×12场景×6组合，正式目录results/logistics_e1f/logistics_e1f_20260921_v1。源177/67及输出209哈希通过，未评价test或质量。独立控制时长/数量/起点，随机起点模12覆盖全部相位。CUSUM24组F1均低于EWMA，拖尾均上升；不升级。下一轮仅改变可见工况切换处理，检验跨切换召回损失，报告覆盖与实际误报。详见RESULTS_LOGISTICS_E1F。2026-09-21用户设备授权后，实现和结果30f0b34已上传，git ls-remote确认远端30f0b348d23773a6e571d8a3b1a78431fff37124；随后提交交付记录并在最终核对后清理临时认证文件。

## E1e里程碑

E1e完整W=9 SMA已实现，55测试通过（11.53秒），源码2b68a4f，运行results/logistics_e1e/logistics_e1e_20260920_v1。审核源177/基线67/本轮69产物hash，test未评价。与EWMA相比，20组全时间轴F1/Recall均退步，19组共同支持F1下降；切换覆盖86.67%。不升级SMA为主方法，不做无边界窗口搜索。详见RESULTS_LOGISTICS_E1E。E1e实现2b68a4f、结果a36004c已推送并核对远程main；本次发布完成，后续从上述下一步继续，临时认证文件在最终核对后清理。

## E1d里程碑

E1d实现有符号EWMA（alpha=1/0.2），47测试通过；源码7ca922a，正式运行results/logistics_e1d/logistics_e1d_20260920_v1，输出67hash通过，源177hash通过，alpha=1复现E1c，test未评分。1倍偏置条件F1 0.0956→0.3031，新告警事件检出45.83%→54.17%，但已检出延迟3.85→4.58步，强偏置拖尾严重。不能宣称普遍收益或工况延迟已解决。详见RESULTS_LOGISTICS_E1D。

## 上一里程碑

E1c已完成5种子×10场景×5方法，全部synthetic validation。新增`logistics_robustness.py`、`run_logistics_robustness.py`、`configs/logistics_e1c.yaml`及7项测试。完整39测试通过（Python3.11.16）；源码提交ec46936；运行`results/logistics_e1c/logistics_e1c_20260919_v1/`，177个产物hash核验。test保存未评价。

关键修正：条件残差不是始终更差，中等偏置有检出收益；强偏置会牵连干净通道。弱偏置召回仍低。延迟12步造成分状态边际整体clean FPR由0.98%增至7.92%，多通道高Top-1不能代替完整故障集合检出。详情见RESULTS_LOGISTICS_E1C和METHOD_LOGISTICS_E1C。不得直接用E1b/E1c跨协议指标差解释算法进步。

## 下一步

E1g重置消融已完成，下一轮按PROJECT_ACCEPTANCE收尾可复用评分组件。保留边际/条件及逐点/EWMA/CUSUM/carry实验对照；默认reset，carry没有可靠的自动启用条件。未知/估计工况另设，评分不接收隐藏真状态。名义q99不等于实际等误报；复用开发集不等于最终盲测。
质量任务需要具体产品、指标、取样与报告可用时间；无质量预测/真实数据结果。外部PDF未改；结果文档已提供论文逐项对应和用户补充信息。

## 复现

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m src.run_temporal_e1d --config configs/logistics_e1d.yaml
.venv/bin/python -m src.run_finite_memory_e1e --config configs/logistics_e1e.yaml
.venv/bin/python -m src.run_controlled_e1f --config configs/logistics_e1f.yaml
.venv/bin/python -m src.run_transition_e1g --config configs/logistics_e1g.yaml
```

新run_id默认唯一；不要覆盖正式目录。源代码修改先提交，再跑正式实验；结果作为后续提交。若.venv失效，按Python3.11和requirements-core重建；不要绕过科研测试。
E1b和更早结果保留。2026-09-19已核对远程main包含E1c实现ec46936、结果a85cfef和交接631fa1a；不在仓库或聊天中保存访问令牌。

2026-09-19发布完成：首次push因无登录凭据失败，用户完成分步设备登录后推送成功。临时认证文件于交付后清理；本轮无需重做实验。

2026-09-20：用户选择继续使用设备授权码上传，不安装持久连接插件。用户完成授权后，E1d实现7ca922a和结果ec519b5已推送；git ls-remote核对远程main为ec519b53065170d1b9b99ba753c312c42e792216。随后提交本交付记录。本次仅完成发布，不重做实验；临时认证文件在最终推送核对后清理。
