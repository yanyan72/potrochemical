# SESSION_HANDOFF

当前读取顺序：AGENTS、PROJECT_BRIEF、ARCHITECTURE、DATA_SPEC、EXPERIMENT_PLAN、TASKS、DECISIONS、STATUS、当前ExecPlan。

用户2026-09-18确认：主场景运输仓储；生产放Future Work；真实数据后续提供，先推进模拟算法，并授权上传GitHub。当前不需要重新征求开发或推送许可。

本轮新增logistics_simulator/sensor_reliability/run_logistics_e1b、configs/logistics_e1b.yaml和测试。
目标是完成储运RQ1/RQ2的5种子validation评价。没有质量预测结果，没有真实数据。方法和边界见METHOD_LOGISTICS_E1B.md与PROJECT_EVIDENCE_MATRIX.md。

Python 3.11.16完整测试32通过；旧浮点测试已修复。正式实验应在已提交源码上执行，并将结果元数据中的commit与Git对应。

运行：
```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m src.run_logistics_e1b --config configs/logistics_e1b.yaml
```

旧normal全局参考85.47%误报是历史失败结果，不删除；新储运数据不与旧数据进行跨版本指标归因。
下一步依据本轮结果检查简单边际与条件残差是否有增益，补弱污染/转场；真实质量目标落实后普通预测基线可以并行。

上传目前缺少CLI凭据；可用远程浏览器也未登录，与用户Mac Chrome是独立会话。不要读取或索取聊天中的密码/令牌；只用支持的安全登录或用户本地正常Git流程。
