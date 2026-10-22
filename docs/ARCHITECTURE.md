# ARCHITECTURE

## 2026-10-22：只读评分—预测配对诊断

rq3_score_prediction_audit.py读取旧val可见观测、冻结参考、已有事件和预测；重算评分并独立记录事件新告警/延续告警、未污染通道告警及不可评分缺失。prediction_rows仅用已有B2/B3预测与评价标签计算配对差、改善批次比例和相对原始的退化。没有模型拟合或test预测入口。

paired_batches.npz保存逐批误差和事件评分摘要，score_metrics/prediction_pairs/coverage及五seed汇总提供两条证据链。独立verify脚本从原模型存档重建误差、复算预测配对表并对齐旧结果；评分AP由标准库计算，事件逻辑由手算测试验证。

## 2026-10-22：正常测量噪声敏感性入口

新增rq3_noise_sensitivity.py，在新namespace用原generate_split生成三档匹配val流；只持久化新敏感性样本，不生成或拟合新train/test。predict_views接受严格可见输入字典，加载原岭回归/LSTM；source_check验证旧数据、模型和final产物，仅对旧test文件做hash。每seed先存可见inputs和独立evaluation_only，再保存predictions，随后读标签计算指标。

独立scripts/verify_rq3_noise_sensitivity.py不导入本轮预测/汇总代码，按随机流复核观测噪声及标签并从预测重算指标。新增报告生成器和复现说明。已有生成器、评分器、预测模型实现与参数均未改。报告属于正常观测的局部敏感性，旧test已使用状态不变。

## 2026-10-21：冻结模型测试入口与独立复核

新增src/rq3_final_test.py，只读取归档参数；显式--execute-frozen-test及协议/实现commit保护正式入口，旧开发入口的test guard保持不变。先写test_use.json，再推断并保存predictions.npz，随后读标签评价。输出预测轴为架构×视图×组×批次×目标；五种子各自保存并拒绝覆盖。source_manifests.json和identity.json记录来源、源码、环境及冻结提交。

--validate-only在旧val上重放，--summarize只汇总已有种子产物。scripts/verify_rq3_final_test.py独立核验源/输出哈希，直接从预测与原标签重算指标、配对差及t区间；不训练模型。全部执行和来源状态见RQ3_FINAL_TEST_TEACHER_REVIEW_20261021.md。完成标志在metadata.json（test_status=used），test_use.json保留首次开始记录。后续新实验不得覆盖此目录。

## 2026-10-21：普通LSTM开发验证路径

rq3_lstm_pilot.py复用rq3_prediction_pilot的开发输入、评分和特征allowlist，以及固定岭回归产物中的评分/填补参数。逐通道train统计、恒定输入屏蔽、单层LSTM均值读出、独立标量目标、原始val选checkpoint；逐seed保存并拒绝覆盖。scripts/verify_rq3_lstm_pilot.py独立复算存档指标，scripts/build_rq3_lstm_teacher_report.py生成审核稿。当前只允许val预测，无test入口；原生成器及岭回归代码保持不变。

## 2026-10-19：RQ3开发预测路径

`rq3_release_audit.py`在原结构审计外检查固定manifest身份、元数据来源及负对照随机流；不修改原数据。`rq3_prediction_pilot.py`复用分工况参考和temporal_reliability，在118步历史窗口上拟合train预处理和岭回归；original val选择alpha，锁定模型后评价十个val视图。预测入口拒绝test，不接受隐藏过程或事件特征。模型、逐批预测和表位于 `results/rq3_prediction/rq3_ridge_pilot_20261019_v1/`；`scripts/verify_rq3_ridge_pilot.py`独立复算存档指标，`scripts/build_rq3_teacher_report.py`从表生成审核Markdown。

## v0.1统一组件（2026-09-27）

`scoring_component.py`加载和校验JSON bundle，验证字段/单位/时间/缺失并按完整序列调用因果评分；`score_batch.py`提供CSV→逐传感器scores.csv+哈希metadata入口。`prepare_scoring_demo.py`从已冻结seed42训练模型导出手工示例，不读取测试序列。`run_final_evaluation.py`按冻结协议读取E1c原test、复用E1f受控事件，保存四种固定方法的最终评价。使用说明见SCORING_GUIDE；无跨文件流式状态、真实数据自动训练或质量预测API。

## 当前储运路径

1. `configs/logistics_e1b.yaml`：示意分布、状态、种子、split规模、污染参数、校准分位数。
2. `src/logistics_simulator.py`：生成均衡的独立储运状态序列；复用污染注入，保留clean/observed/mask/events。
3. `src/sensor_reliability.py`：train_fit标准化和LedoitWolf参考；train_cal校准；观测+工况输出逐通道偏离、分数、可用性和告警。
4. `src/run_logistics_e1b.py`：5种子运行、共同完整行支持集评价、缺失与牵连误报、参数和数据保存、版本与哈希审计。
5. `results/logistics_e1b/<run_id>/`：唯一运行目录。

评分API不接收隐藏污染标签；未知状态下分工况评分不可用。边际方法只需本通道值，条件残差需要同一时刻的全部输入。缺失不能由隐藏真值填补。

## E1c压力实验路径

`configs/logistics_e1c.yaml -> logistics_robustness.py -> run_logistics_robustness.py`。训练参考复用E1b算法；场景数据包含逐时刻true_regimes/recorded_regimes和transition标记，评分仅接收记录状态。新数据/事件/评分、逐种子分层指标、配对差和图位于`results/logistics_e1c/<run_id>/`。这一路径不修改E1b生成器或评分公式。

## E1d时序消融路径

`configs/logistics_e1d.yaml -> temporal_reliability.py -> run_temporal_e1d.py`。审核并复用E1c数据/参考，新增因果有符号EWMA和train_cal阈值；评分只接受observed/recorded_regimes。alpha=1逐场景复现原评分；结果保存于`results/logistics_e1d/<run_id>/`，input_manifest指向既有数据，避免复制源运行。

## E1e有限窗口路径

`configs/logistics_e1e.yaml -> finite_memory.py -> run_finite_memory_e1e.py`。审核E1c输入和E1d存档基线，SMA独立校准；统一评价六种方法的共同完整行、各自支持、全时间轴漏检和全事件。新SMA评分/模型保存在`results/logistics_e1e/<run_id>/`，基线通过manifest引用。

## E1f受控事件路径

`configs/logistics_e1f.yaml -> controlled_events.py + cusum_reliability.py -> run_controlled_e1f.py`。审核E1c训练参考和E1d既有模型，独立随机流生成新开发validation，CUSUM仅用旧train_cal校准。保存配对事件、六种组合双视图分数及相位/方法/场景差至`results/logistics_e1f/<run_id>/`，test不生成、不评价。

## E1g重置消融路径

E1g：`configs/logistics_e1g.yaml -> temporal_reliability.py -> run_transition_e1g.py`。EWMA新增reset_on_record_change开关，默认true；读取E1f旧分数和E1c校准数据，验证旧基线/固定工况不变，新增carry分数与可见记录切换分层。输出在`results/logistics_e1g/<run_id>/`，旧产物不修改。

## 历史路径（继续可复现）

`simulator.py -> corruption.py -> run_preprocessing.py -> run_trust_reference.py -> run_trust_scoring.py -> run_e1_evaluation.py`。
这是四变量全局评分路径，历史数据/结果不覆盖，不能代替新通道评分。

## 后续预测路径

实际数据字典/化验时间 -> 序列级划分 -> 因果窗口与标签对齐 -> 相同预测器的观测、观测+工况、观测+工况+评分三组对照。
历史质量只有已经出具才可作为输入。无质量标签时不生成伪连续真值。LSTM、选择性修复和真实接入待后续计划。

生产推广、物理约束、区块链仅为远期扩展，不属于当前已实现架构。
