# SESSION_HANDOFF

> 新会话首先读取本文件、`docs/STATUS.md`、`docs/project_log.md` 和 `docs/plans/current_execplan.md`。

## 当前阶段

Milestone 0–3 和 Milestone 4 的第一版 E1 已完成。E1 已实现训练参考 q90/q99、连续 trust、逐点输出、第一张 distance/trust 图和第一组检测指标，但未通过进入预测实验的门控。下一步只做工况条件化参考和 E1 重评，不做 LSTM。

真实数据为 `Pending external data / 等待外部数据`；全部当前结果是 synthetic/simulated。

## 本次完成

- q90/q99 只从 800 个 normal-train clean 参考平方距离拟合；
- 可信组边界：`d² <= q90` 为 high，`q90 < d² <= q99` 为 uncertain，`d² > q99` 为 low；
- 连续映射：`trust=exp(-d²/(2q90))`；
- 新增无泄漏、分位数、边界、单调性、范围、指标和端到端测试；
- 生成 4,800 行逐点审计 CSV；
- 生成 train/validation 检测表、分异常类型召回表和分工况误报表；
- 生成 validation 序列 `val_0003` 的 distance/trust 时间图；
- test 逐点结果保留，但 test 指标未计算；
- 正式 run 完成，并在 `/tmp` 独立重跑，六个核心产物逐字节一致；
- 根据验证结果决定暂不进入预测实验。

## 本轮新增或修改文件

- `src/trust_score.py`
- `src/evaluate_trust.py`
- `src/run_e1_evaluation.py`
- `configs/milestone4_e1_trust.yaml`
- `tests/test_trust_score.py`
- `tests/test_evaluate_trust.py`
- `tests/test_e1_pipeline.py`
- `README.md`
- `docs/ARCHITECTURE.md`
- `docs/DECISIONS.md`
- `docs/STATUS.md`
- `docs/TASKS.md`
- `docs/project_log.md`
- `docs/plans/current_execplan.md`
- `docs/SESSION_HANDOFF.md`
- `data/processed/e1trust_20260910T035417735017Z_29f039af/`
- `results/tables/e1trust_20260910T035417735017Z_29f039af/`
- `results/figures/e1trust_20260910T035417735017Z_29f039af/`

## 正式 run

- run ID：`e1trust_20260910T035417735017Z_29f039af`；
- config SHA-256：`29f039af462a1d0777768b33445411330a8409a3d6bdc3cf76d6de7bdd9d2fef`；
- code version：`6dc612a9bd485c729761376a19f1088f21fbae8c`；
- q90：`7.497480759602013`；q99：`13.833465760805584`；
- validation q99：Precision `0.0967`、Recall `0.9302`、F1 `0.1752`、PR-AUC `0.4797`、FPR `0.8547`；
- validation normal 未污染点 q99 FPR：`0.0078`；
- validation compound/high-temperature/high-vibration 未污染点 q99 FPR：均为 `1.00`。

## 科学结论

距离排序有信号，但当前统一 normal 参考把合法工况变化误认为传感器污染。若直接将该 trust 用于数据修正或预测，会系统性压低合法非 normal 工况的权重。因此预测阶段暂停，先修订参考集合。该结论只适用于当前 synthetic 实验，不能解释为工业性能。

## 正式结果路径

- `data/processed/e1trust_20260910T035417735017Z_29f039af/pointwise_trust_scores.csv`
- `data/processed/e1trust_20260910T035417735017Z_29f039af/trust_calibration.json`
- `data/processed/e1trust_20260910T035417735017Z_29f039af/metadata.json`
- `results/tables/e1trust_20260910T035417735017Z_29f039af/detection_metrics.csv`
- `results/tables/e1trust_20260910T035417735017Z_29f039af/corruption_type_recall.csv`
- `results/tables/e1trust_20260910T035417735017Z_29f039af/condition_false_positive_rates.csv`
- `results/figures/e1trust_20260910T035417735017Z_29f039af/distance_trust_timeseries.png`

## 测试与复现

- E1 专项：`11 passed`；
- 最终全量：`24 passed in 4.12s`；
- `compileall`：通过；
- 第一次端到端测试失败来自测试夹具未覆盖 uncertain 区间，修正夹具后通过，算法未因此改变；
- 第一次 `/tmp` 审计配置错误地改写了输入路径而失败，修正审计配置后成功；正式结果未受影响；
- 独立重跑的逐点 CSV、校准 JSON、三张表和 PNG 哈希与正式结果一致。

## Git 状态

- GitHub：`https://github.com/yanyan72/potrochemical`；
- 分支：`main`；
- E1 实现提交：`6dc612a9bd485c729761376a19f1088f21fbae8c`；
- 正式结果和文档需在最终交付提交后检查本地与远端是否同步。

## 下一步唯一优先任务

对每个已知 synthetic condition 使用对应 clean train 数据独立拟合中心、收缩协方差、q90/q99 和 tau，并复用本轮相同的无泄漏测试、逐点输出和 train/validation 指标。只有工况误报显著下降且检测指标合理后，才进入 persistence 与线性回归预测基线。
