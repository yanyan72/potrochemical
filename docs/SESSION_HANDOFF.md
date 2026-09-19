# SESSION_HANDOFF

更新2026-09-19。读取AGENTS、PROJECT_BRIEF、ARCHITECTURE、DATA_SPEC、EXPERIMENT_PLAN、TASKS、DECISIONS、STATUS、当前ExecPlan。
用户确认运输仓储主场景、生产Future Work、真实数据后续，并再次授权继续推进和上传GitHub；不需重复征求开发/推送许可。

## 最新里程碑

E1c已完成5种子×10场景×5方法，全部synthetic validation。新增`logistics_robustness.py`、`run_logistics_robustness.py`、`configs/logistics_e1c.yaml`及7项测试。完整39测试通过（Python3.11.16）；源码提交ec46936；运行`results/logistics_e1c/logistics_e1c_20260919_v1/`，177个产物hash核验。test保存未评价。

关键修正：条件残差不是始终更差，中等偏置有检出收益；强偏置会牵连干净通道。弱偏置召回仍低。延迟12步造成分状态边际整体clean FPR由0.98%增至7.92%，多通道高Top-1不能代替完整故障集合检出。详情见RESULTS_LOGISTICS_E1C和METHOD_LOGISTICS_E1C。不得直接用E1b/E1c跨协议指标差解释算法进步。

## 下一步

先试单一因果时序累积候选，在同等校准误报预算下比较弱偏置、告警延迟和转场误报；保持边际/条件两个基线，不预设最终方法胜出。再处理记录状态不确定性、更强基线、未知/估计工况。
质量任务需要具体产品、指标、取样与报告可用时间；无质量预测/真实数据结果。外部PDF未改；结果文档已提供论文逐项对应和用户补充信息。

## 复现

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m src.run_logistics_robustness --config configs/logistics_e1c.yaml
```

新run_id默认唯一；不要覆盖正式目录。源代码修改先提交，再跑正式实验；结果作为后续提交。若.venv失效，按Python3.11和requirements-core重建；不要绕过科研测试。
E1b和更早结果保留。2026-09-19已核对远程main包含E1c实现ec46936、结果a85cfef和交接631fa1a；不在仓库或聊天中保存访问令牌。

2026-09-19发布完成：首次push因无登录凭据失败，用户完成分步设备登录后推送成功。临时认证文件于交付后清理；本轮无需重做实验。
