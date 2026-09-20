# PROJECT LOG

## 2026-09-20：E1d因果EWMA单模块消融

复用E1c相同数据/参考，固定alpha=1/0.2，加入独立校准、有符号时间累积和事件/拖尾评价。源码7ca922a；正式运行logistics_e1d_20260920_v1，alpha=1完整复现原评分，源177和输出67hash核验，test未评分。47测试通过（9.85秒）；首次测试7通过1失败源自忽略720点分位数的1/n计数分辨率，修正测试而未改算法。1倍偏置条件F1改善，但强偏置拖尾和事件响应代价明显；全部保留，论文对应和下轮有限记忆对照已记录。当前无质量预测或真实数据结论。

## 2026-09-19：E1c评分边界与记录延迟

交付记录：实现ec46936、结果a85cfef已上传GitHub main，2026-09-19核对远程631fa1a包含全部本轮工作。

完成5种子×10成对场景×5方法，配置logistics_e1c.yaml，正式运行logistics_e1c_20260919_v1，源码ec46936且运行时实现干净。完整39测试通过，177产物哈希通过，test未评价。虚拟环境入口失效后按Python3.11.16和既有依赖重建。中等偏置下条件残差有收益，强偏置有牵连；弱偏置识别不足；延迟12步使分状态边际整体clean FPR由0.98%到7.92%。多通道高Top-1掩盖完整检出低的问题。保留全部结果，更新论文主张与下轮单模块计划；外部PDF未编辑，质量预测和真实数据仍未验证。

## 2026-09-18：储运范围收敛与E1b实现

用户确认运输仓储为当前主场景、生产为未来推广、真实数据后续提供，并授权GitHub交付。
新增独立储运环境模拟器、5种逐通道参考方法、train_fit/train_cal分离、缺失/未知状态处理、通道定位和牵连评价；保留旧数据和归档计划。
Python 3.11.16完整测试32通过（4.34秒），compileall与diff检查通过。修复旧CSV往返容差；新增支持集测试第一次因忽略不同状态缺失数量而失败，修正断言为方法之间共同支持一致，未更改算法。
正式5种子实验基于c4a8bae已完成，run=logistics_e1b_20260918_v1，30个产物hash通过。分状态边际F1=0.6250±0.0579；分状态条件F1=0.5182±0.0190且污染行干净通道牵连FPR=29.22%。选择简单边际作为下一阶段主基线，负面结果保留。质量标签尚未定义，不实现伪质量预测。初次CLI和浏览器授权未能返回凭据，随后通过分步设备授权完成交付；实现提交c4a8bae和结果提交48c433b已推送至远程main。

本文件按日期记录已经真实完成并验证的科研工程工作。所有结果均应区分“代码已实现”“测试已通过”和“实验已经证明有效”。当前数据来源全部为 synthetic/simulated；真实数据状态为 `Pending external data / 等待外部数据`。

## 2026-09-10：Milestone 4 第一版 E1 可信度校准与检测评价

### 本次目标

只使用正常训练参考距离实现 q90/q99、连续 trust 和三档可信组；输出每个时间点的可审计结果，生成第一张 distance/trust 时间图和第一组异常检测指标，并根据 validation 结果决定是否进入预测实验。本轮不实现任何预测模型。

### 路线审查与约束

路线可行，但为避免测试泄漏和错误归因，增加三项约束：阈值和 tau 只能来自800个 normal-train clean 参考距离；首次指标只计算 train/validation，test 只保存逐点结果；除总体检测指标外，必须检查各合法工况未污染点的误报率。

### 完成内容

- 在 `src/trust_score.py` 实现训练参考 q90/q99 校准、三档边界归组和指数 trust；
- 新增 `src/evaluate_trust.py`，计算 Precision、Recall、F1、PR-AUC、FPR、FNR、分异常类型召回和分工况误报；
- 新增 `src/run_e1_evaluation.py` 和 `configs/milestone4_e1_trust.yaml`，完成上游 run/hash/身份检查、唯一 run 和全部产物持久化；
- 新增或扩展 `tests/test_trust_score.py`、`tests/test_evaluate_trust.py`、`tests/test_e1_pipeline.py`；
- 为4,800个时间点输出 `sequence_id`、`time_index`、`split`、`condition`、`corruption_label`、`corruption_type`、`mahalanobis_distance_squared`、`mahalanobis_distance`、`trust_score`、`trust_group`；
- 生成第一张 validation distance/trust 时间序列图和三张 CSV 指标表；
- 完成正式 synthetic run 和独立 `/tmp` 复现审计；
- 实现提交：`6dc612a9bd485c729761376a19f1088f21fbae8c`。

### 方法说明

参考平方马氏距离记为 `d²_ref`。第一版校准为：

```text
q90 = Quantile(d²_ref, 0.90, method="linear")
q99 = Quantile(d²_ref, 0.99, method="linear")

d² <= q90          -> high
q90 < d² <= q99    -> uncertain
d² > q99           -> low

trust = exp(-d² / (2 * q90))
```

该映射在零距离处为1，随距离增加单调不升，范围为 `[0,1]`。q90/q99和tau均不接收观测标签、validation或test数据。二元检测分别用 `d²>q90` 和 `d²>q99`，PR-AUC使用连续 `d²` 作为异常分数。

### 实际命令

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/petrochemical_mplconfig \
  .venv/bin/python -m pytest tests/test_trust_score.py -q -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/petrochemical_mplconfig \
  .venv/bin/python -m pytest tests/test_trust_score.py \
  tests/test_evaluate_trust.py tests/test_e1_pipeline.py -q -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/petrochemical_mplconfig \
  .venv/bin/python -m pytest -q -p no:cacheprovider
PYTHONPYCACHEPREFIX=/tmp/petrochemical_pycache \
  .venv/bin/python -m compileall -q src tests
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/petrochemical_mplconfig \
  .venv/bin/python -m src.run_e1_evaluation \
  --config configs/milestone4_e1_trust.yaml
```

复现审计使用临时配置将输出写入 `/tmp/petrochemical_e1_audit/`，不覆盖正式结果。

### 测试与问题记录

- q90/q99、无泄漏接口、边界、单调性和非法输入：`7 passed`；
- 第一轮包含端到端测试：`1 failed, 10 passed`。原因是测试夹具没有数值落入 q90–q99，却错误要求存在 uncertain 组；修正一项夹具值后算法无需修改；
- 修正后 E1 专项：`11 passed`；
- 实现完成后的全套：`24 passed in 3.63s`；
- 文档与结果复核后的最终全套：`24 passed in 4.12s`；
- `compileall`：通过；
- 系统 `python3` 缺少 pandas，统一改用项目 `.venv/bin/python`；这不是代码失败；
- 第一次审计临时配置错误改写了输入路径，产生 `FileNotFoundError`；修正临时配置后成功，正式结果未受影响；
- 独立重跑的逐点CSV、校准JSON、三张指标表和PNG均与正式产物逐字节一致；
- 当前无未解决的代码错误或失败测试。

### 正式实验结果

- run ID：`e1trust_20260910T035417735017Z_29f039af`；
- 数据来源：synthetic/simulated；
- code version：`6dc612a9bd485c729761376a19f1088f21fbae8c`；
- 配置 SHA-256：`29f039af462a1d0777768b33445411330a8409a3d6bdc3cf76d6de7bdd9d2fef`；
- 参考距离数：800；q90=`7.497480759602013`；q99=`13.833465760805584`；
- 逐点结果：4,800行；trust范围`0.0`至`0.9984665059040305`；
- 分组：high 1,192、uncertain 116、low 3,492；
- 绘图序列：validation 的 `val_0003`。

validation q99 结果：

- 960点中污染86点，污染率`0.0896`；
- TP=80、FP=747、FN=6、TN=127；
- Precision=`0.0967`、Recall=`0.9302`、F1=`0.1752`、PR-AUC=`0.4797`；
- FPR=`0.8547`、FNR=`0.0698`；
- 分类型召回：bias=`1.00`、drift=`1.00`、missing=`0.75`、random replacement=`0.9091`、spike=`1.00`；
- 未污染点 q99 FPR：normal=`0.0078`；compound、high-temperature、high-vibration均为`1.00`。

test有960行逐点结果，但没有test指标，且没有用于门控或调参。

### 结果解释与门控决定

PR-AUC高于validation污染率，说明距离能够在一定程度上给污染点排序；但q99下85%以上未污染点被误报，原因集中在合法的非normal工况。第一版单一normal参考没有学会“不同合法工况也可能正常”，不能直接用于可信度修正或预测输入。因此本轮结论不是“检测有效”，而是“发现了参考集合定义的结构性问题”。

**阶段门控：暂不进入预测实验。** 下一步先实现工况条件化训练参考并用同一套指标重评。

### 正式结果路径

- `data/processed/e1trust_20260910T035417735017Z_29f039af/pointwise_trust_scores.csv`；
- `data/processed/e1trust_20260910T035417735017Z_29f039af/trust_calibration.json`；
- `data/processed/e1trust_20260910T035417735017Z_29f039af/metadata.json`；
- `data/processed/e1trust_20260910T035417735017Z_29f039af/config_snapshot.yaml`；
- `results/tables/e1trust_20260910T035417735017Z_29f039af/detection_metrics.csv`；
- `results/tables/e1trust_20260910T035417735017Z_29f039af/corruption_type_recall.csv`；
- `results/tables/e1trust_20260910T035417735017Z_29f039af/condition_false_positive_rates.csv`；
- `results/figures/e1trust_20260910T035417735017Z_29f039af/distance_trust_timeseries.png`。

### 已知限制

- 全部结果为synthetic，不能替代真实工业或实验室验证；
- clean oracle参考不代表真实部署可见隐藏真值；
- 单一normal参考无法覆盖合法非normal工况；
- test尚未评价，这是防止泄漏的主动设计；
- 当前结果未证明可信度可改善预测；
- 当前Python 3.14尚未在建议的Python 3.11复验。

### 下一步唯一计划

按synthetic工况建立训练参考：每个condition只使用对应clean train序列拟合中心、收缩协方差、q90/q99和tau，再复用相同的无泄漏测试、逐点输出与train/validation指标。只有工况误报显著下降后，才进入persistence和线性回归预测基线。

## 2026-08-14：Milestone 3 平方马氏距离

### 本次目标

使用 Milestone 2 已保存的正常参考中心和精度矩阵，计算 Milestone 1 全部标准化观测及正常参考集自身的平方马氏距离。本轮只验证距离公式、输入身份、持久化和复现性，不选择阈值，不评价异常识别性能。

### 完成内容

- 新增 `src/trust_score.py`，实现保留任意前导 shape 的向量化平方马氏距离；
- 新增 `src/run_trust_scoring.py`，核对 M1/M2 run ID、SHA-256、特征顺序、sequence/time/split 身份和参考参数跨文件一致性；
- 新增 `configs/milestone3_mahalanobis.yaml`；
- 新增 `tests/test_trust_score.py` 和 `tests/test_trust_scoring_pipeline.py`；
- 保存全部观测距离、参考距离、序列 ID、时间索引、split、参考序列 ID、配置和元数据；
- 正式结果记录实现提交 `e1748bf0f9c2096ad700ec7d1e61212122f4a609`；
- 仓库当前已初始化 Git，`main` 跟踪 GitHub `yanyan72/potrochemical`。历史 M0–M2 产物中的 `unavailable_no_git_repository` 保持不变，因为它们反映生成当时状态。

### 方法说明

对标准化四维向量 `x`，计算：

```text
d² = (x - μ)ᵀ P (x - μ)
```

其中 `μ` 是 M2 正常训练参考中心，`P` 是 Ledoit–Wolf 收缩协方差的精度矩阵。M3 不重新拟合任何参数；train、val、test 都只应用同一组冻结参数。精度矩阵必须有限、对称和正定，距离必须有限、非负。

### 实际命令

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/petrochemical_mplconfig \
  .venv/bin/python -m pytest tests/test_trust_score.py \
  tests/test_trust_scoring_pipeline.py -q -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/petrochemical_mplconfig \
  .venv/bin/python -m pytest -q -p no:cacheprovider
PYTHONPYCACHEPREFIX=/tmp/petrochemical_pycache \
  .venv/bin/python -m compileall -q src tests
.venv/bin/python -m src.run_trust_scoring \
  --config configs/milestone3_mahalanobis.yaml
.venv/bin/python -m src.run_trust_scoring \
  --config configs/milestone3_mahalanobis.yaml \
  --output-root /tmp/petrochemical_trustscore_audit_20260814 \
  --run-id trustscore_audit_20260814
```

### 测试记录

- 首次专项测试：`4 passed in 19.42s`；首次运行包含 Matplotlib 字体缓存初始化；
- 加强参数来源检查后的专项测试：`4 passed in 2.13s`；
- 实现完成后的全套测试：`16 passed in 3.45s`；
- 文档与结果审查后的最终全套测试：`16 passed in 3.63s`；
- `compileall`：通过；
- 没有失败测试、未解决错误或运行警告；
- 独立 `/tmp` 重跑：NPZ 中8个数组全部逐值一致，输出 NPZ SHA-256 相同。

### 正式实验结果

- run ID：`trustscore_20260814T140937863216Z_09a4bd41`；
- 数据来源：synthetic/simulated；
- 输入观测：30条序列×160步×4特征；train/val/test序列数为18/6/6；
- 观测距离 shape：`(30,160)`，共4,800个；
- 参考距离 shape：`(800,)`；
- 观测 `d²`：最小`0.023012334111355618`，最大`13004.834691635755`，均值`283.3951245872942`，中位数`171.45521619499675`；
- 参考 `d²`：最小`0.12766020043397946`，最大`24.026885127638717`，均值`3.6697625736134127`，中位数`2.914257021883051`；
- 两组距离的非有限值数和负值数均为0；
- 配置 SHA-256：`09a4bd41b49bfa332a97c256315e8abcec6a5dcf24ea9bb3f8d6b1bd3f30326a`；
- 距离 NPZ SHA-256：`587892147cb6a818213c85fe69a0180deb20e99069829affdc532b36755dfd53`；
- code version：`e1748bf0f9c2096ad700ec7d1e61212122f4a609`。

观测距离整体大于 normal 参考距离，可能同时受到已设计的非 normal 工况和人工污染影响；本轮没有阈值、分组或标签指标，因此不能据此声称异常已经被正确识别。

### 结果路径

- `data/processed/trustscore_20260814T140937863216Z_09a4bd41/mahalanobis_distances.npz`；
- `data/processed/trustscore_20260814T140937863216Z_09a4bd41/metadata.json`；
- `data/processed/trustscore_20260814T140937863216Z_09a4bd41/config_snapshot.yaml`；
- 独立审计：`/tmp/petrochemical_trustscore_audit_20260814/trustscore_audit_20260814/`。

### 已知限制

- 全部输入和结果为 synthetic，不能替代真实工业验证；
- M2 使用 synthetic clean oracle 参考，真实部署不能假设可见隐藏真值；
- 参考仅含5条 normal train 序列，800个时间点存在序列内相关性；
- 当前未计算阈值、可信组、连续可信度、PCA或任何异常识别指标；
- 当前 Python 3.14 尚未在规范建议的 Python 3.11 复验。

### 下一步唯一计划

只使用已保存的800个正常训练参考距离计算 q90/q99 基线阈值，并对全部距离划分 `high/uncertain/low` 可信组；补充分位数公式、边界归组、train-only 阈值来源、shape 和复现测试。本小步先不计算连续可信度、PCA 或 Precision/Recall/F1/PR-AUC。

## 2026-08-13：Milestone 2 正常参考集合与收缩协方差

### 本次目标

在不使用验证集、测试集和人工异常标签的前提下，从训练集正常工况建立多维统计参考集合，并得到可逆、数值稳定的协方差和精度矩阵，为下一步马氏距离计算提供参数。

### 完成内容

- 安装 requirements 已声明但当前 `.venv` 缺失的 scikit-learn 1.9.0；
- 新增 `src/trust_reference.py`，封装 Ledoit–Wolf 参考估计和数值诊断；
- 新增 `src/run_trust_reference.py`，核对 M0/M1 输入 run、SHA-256、序列顺序、split 和特征顺序；
- 新增 `configs/milestone2_trust_reference.yaml`；
- 从 M0 选择5条完整 `train + normal` 序列；
- 沿用 M1 clean-train 参数，将 clean 参考值标准化为800×4矩阵；
- 保存中心、收缩协方差、精度矩阵、收缩系数、序列 ID、配置、元数据和哈希；
- 新增 `tests/test_trust_reference.py` 和 `tests/test_trust_reference_pipeline.py`。

### 方法说明

Ledoit–Wolf 收缩协方差可以简单理解为：经验协方差能够描述温度、压力、振动和浓度如何共同变化，但变量相关性较强时直接求逆可能不稳定；收缩估计会把经验协方差适度拉向更规则的矩阵，从而得到更稳定的精度矩阵。

本次参考集合直接使用 synthetic clean 真值，是为了先建立可审计的 oracle baseline。它不是现实部署假设，不能描述成真实传感器系统能够知道隐藏正确值。

### 实际命令

```bash
.venv/bin/pip install 'scikit-learn>=1.4,<2'
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/petrochemical_mplconfig \
  .venv/bin/python -m pytest -q -p no:cacheprovider
PYTHONPYCACHEPREFIX=/tmp/petrochemical_pycache \
  .venv/bin/python -m compileall -q src tests
.venv/bin/python -m src.run_trust_reference \
  --config configs/milestone2_trust_reference.yaml
```

### 测试记录

- 首次全套测试：`1 failed, 11 passed in 16.55s`；
- 失败原因：NPZ中的参考序列ID为object dtype，默认安全加载不允许pickle；
- 修复：保存前强制转为Unicode数组，并新增dtype断言；
- 修复后专项测试：`3 passed in 2.14s`；
- 最终全套测试：`12 passed in 2.92s`；
- `compileall`：通过；
- 独立 `/tmp` 重跑：全部NPZ数组和参数JSON逐值一致。

### 正式实验结果

- run ID：`trustref_20260813T074437348987Z_34d5fa5c`；
- 数据来源：synthetic；
- 参考序列ID：`train_0004`、`train_0008`、`train_0010`、`train_0015`、`train_0017`；
- 参考矩阵：800×4；
- 收缩系数：`0.0036545028372638863`；
- 协方差特征值：`0.0051141175843415695`、`0.028090600353115384`、`0.041727836195232425`、`1.3954002296960009`；
- 协方差条件数：`272.85259024321965`；
- 协方差对称误差：`0.0`；
- 精度矩阵对称误差：`3.552713678800501e-15`；
- `covariance @ precision` 最大单位阵误差：`2.4868995751603507e-14`；
- NPZ SHA-256：`f22e4f7b90558a363b2a527ea84633f6fa64df165d2ebae524252f6015caccd2`；
- 参数JSON SHA-256：`00d3a9718c77f67157437b94339299d283c9f7e99a5bbad5709589e8748e109b`。

这些结果说明当前矩阵有限、对称、正定、可逆且可复现；它们不能说明异常检测已经有效。

### 结果路径

- `data/processed/trustref_20260813T074437348987Z_34d5fa5c/reference_set.npz`；
- `data/processed/trustref_20260813T074437348987Z_34d5fa5c/trust_reference_params.json`；
- `data/processed/trustref_20260813T074437348987Z_34d5fa5c/metadata.json`；
- `data/processed/trustref_20260813T074437348987Z_34d5fa5c/config_snapshot.yaml`。

### 已知限制

- 只有5条normal train序列，800个时间点存在序列内相关性；
- M1标准化参数来自全部clean train，非normal train会间接影响尺度；
- 当前没有真实实验室或企业数据；
- 当前Python 3.14尚未在规范建议的Python 3.11复验；
- 仓库无Git元数据。

### 下一步唯一计划

使用已保存中心和精度矩阵计算参考集与全部标准化观测的平方马氏距离，并添加公式数值测试、shape/有限性检查、输入身份核验和复现测试。本步先不选择q90/q99阈值，不计算Precision、Recall、F1或PR-AUC。
