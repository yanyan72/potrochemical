# SESSION_HANDOFF

更新2026-09-20。读取AGENTS、PROJECT_BRIEF、ARCHITECTURE、DATA_SPEC、EXPERIMENT_PLAN、TASKS、DECISIONS、STATUS、当前ExecPlan。
用户确认运输仓储主场景、生产Future Work、真实数据后续，并再次授权继续推进和上传GitHub；不需重复征求开发/推送许可。

## 最新里程碑

E1d实现有符号EWMA（alpha=1/0.2），47测试通过；源码7ca922a，正式运行results/logistics_e1d/logistics_e1d_20260920_v1，输出67hash通过，源177hash通过，alpha=1复现E1c，test未评分。1倍偏置条件F1 0.0956→0.3031，新告警事件检出45.83%→54.17%，但已检出延迟3.85→4.58步，强偏置拖尾严重。不能宣称普遍收益或工况延迟已解决。详见RESULTS_LOGISTICS_E1D。

## 上一里程碑

E1c已完成5种子×10场景×5方法，全部synthetic validation。新增`logistics_robustness.py`、`run_logistics_robustness.py`、`configs/logistics_e1c.yaml`及7项测试。完整39测试通过（Python3.11.16）；源码提交ec46936；运行`results/logistics_e1c/logistics_e1c_20260919_v1/`，177个产物hash核验。test保存未评价。

关键修正：条件残差不是始终更差，中等偏置有检出收益；强偏置会牵连干净通道。弱偏置召回仍低。延迟12步造成分状态边际整体clean FPR由0.98%增至7.92%，多通道高Top-1不能代替完整故障集合检出。详情见RESULTS_LOGISTICS_E1C和METHOD_LOGISTICS_E1C。不得直接用E1b/E1c跨协议指标差解释算法进步。

## 下一步

下一步用有限记忆有符号滑动均值作对照，检查弱偏置、启动期覆盖、告警延迟和拖尾；保持边际/条件及逐点/EWMA基线，不预设胜出。再处理记录状态不确定性、更强基线、未知/估计工况。名义校准预算一致不等于实际validation误报一致。
质量任务需要具体产品、指标、取样与报告可用时间；无质量预测/真实数据结果。外部PDF未改；结果文档已提供论文逐项对应和用户补充信息。

## 复现

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m src.run_temporal_e1d --config configs/logistics_e1d.yaml
```

新run_id默认唯一；不要覆盖正式目录。源代码修改先提交，再跑正式实验；结果作为后续提交。若.venv失效，按Python3.11和requirements-core重建；不要绕过科研测试。
E1b和更早结果保留。2026-09-19已核对远程main包含E1c实现ec46936、结果a85cfef和交接631fa1a；不在仓库或聊天中保存访问令牌。

2026-09-19发布完成：首次push因无登录凭据失败，用户完成分步设备登录后推送成功。临时认证文件于交付后清理；本轮无需重做实验。

2026-09-20：用户选择继续使用设备授权码上传，不安装持久连接插件。用户完成授权后，E1d实现7ca922a和结果ec519b5已推送；git ls-remote核对远程main为ec519b53065170d1b9b99ba753c312c42e792216。随后提交本交付记录。本次仅完成发布，不重做实验；临时认证文件在最终推送核对后清理。
