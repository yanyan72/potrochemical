# Milestone 0 执行计划：可复现时序仿真与基础异常注入

## Purpose

本计划交付第一阶段最小、可验证的数据闭环：从 YAML 配置生成多条独立石化过程时序，保留干净传感器真值和归一化质量真值，在观测特征中注入 `spike`、`bias`、`drift`、`missing`、`random_replacement` 五类人工异常，输出逐特征异常掩码与事件元数据，并通过自动测试和示例图证明数据范围、复现性和标签一致性。该闭环是后续可信度识别的输入基础；本轮不训练任何模型。

## Current state

- 仓库包含完整的研究说明、数据规范、实验计划、基础配置和项目工作规范。
- `src/` 已包含配置、仿真、污染、绘图和生成入口；`tests/` 已包含仿真、污染和端到端测试。
- `configs/baseline.yaml` 已描述完整后续实验规模（1000/200/200 条序列）和未来模块参数，但不适合作为第一轮快速烟雾测试配置。
- `docs/source_materials/` 中两份原始材料讨论 LSTM-PINN、阿伦尼乌斯约束、动态物理参数和区块链设想；这些均被项目主文档明确排到后续阶段。
- 当前目录没有 `.git/`，因此暂时无法记录 Git commit；所有产物必须明确写入 `code_version: unavailable_no_git_repository`。
- 历史 run 均保留且不覆盖；当前最终结果为 `milestone0_20260719T153039079827Z_0abad9ce`。

## Scientific assumptions

1. 一条序列表示一个独立的储运或连续工艺片段，时间步暂用抽象采样单位；`delta_t=1.0` 不宣称等于分钟或小时。
2. 传感器变量及单位为：温度 K、压力 MPa、振动 mm/s、浓度为 `[0,1]` 无量纲比例；质量 `Q` 为 `[0,1]` 无量纲综合指标。
3. 第一版包含 `normal`、`high_temperature`、`high_vibration`、`compound` 四类过程工况。工况变化属于干净过程真值，不属于传感器污染，也不代表真实工业数据。
4. 质量由可配置的离散经验递推生成：`Q[t+1] = clip(Q[t] - r[t] * delta_t + epsilon[t], 0, 1)`。`r[t]` 受干净温度、压力、振动和浓度影响，但该关系仅用于可控仿真，不声称是经过工业验证的机理。
5. 人工污染仅修改观测特征，不修改干净特征、质量真值、工况或序列划分。
6. 污染比例定义为“被污染的序列-时间行数 / 全部序列-时间行数”。第一版每个被污染时刻只对应一个特征和一种异常，避免长表标签歧义。
7. `spike` 为单点脉冲，`bias` 为连续常量偏移，`drift` 为连续渐变偏移；幅度依据各特征干净数据标准差缩放并写入物理单位元数据。
8. 训练、验证、测试按完整序列独立生成和标注；本轮不拟合统计量或模型，也不使用测试集选择阈值。
9. 数据来源全部为仿真；本轮结果只能证明实现正确性与可复现性，不能证明真实工业性能。

## Known facts, unknowns, and risks

### Known facts

- 后续可信度算法需要四维数值特征 `temperature`、`pressure`、`vibration`、`concentration`。
- 数据规范要求同时保留干净真值、污染观测、异常类型、异常特征、异常幅度、序列 ID、时间索引和 split。
- 当前立即任务仅要求 `spike`、`bias`、`drift`；`missing`、`random_replacement` 和时间错位后续实现。

### Unknowns

- 真实采样间隔、具体产品、装置和变量工程范围尚未确认。
- 质量综合指标 `Q` 的真实定义和标定方式尚未确认。
- 暂无导师提供的真实或半真实数据。

### Risks and controls

- 风险：异常过强或过弱导致后续检测任务失真。控制：幅度完全配置化，并在元数据中记录实际幅度。
- 风险：工况异常与传感器污染概念混淆。控制：分别保存 `condition` 和 corruption 字段，测试污染不改变工况与质量。
- 风险：随机数调用顺序变化破坏复现。控制：显式传递 `numpy.random.Generator`/seed，并用相同配置的逐值相等测试。
- 风险：Parquet 需要未声明的 `pyarrow`。控制：第一轮主数据使用标准 CSV，事件元数据与运行元数据使用 JSON；不新增依赖。
- 风险：已有完整配置规模使烟雾测试过慢。控制：新增独立 `configs/milestone0.yaml` 小规模配置，不改写 `baseline.yaml`。

## Scope

### Included

- YAML 配置读取与必要字段校验。
- 多序列、多工况、四传感器与质量真值仿真。
- 按完整序列生成 train/val/test split。
- `spike`、`bias`、`drift` 污染注入。
- 逐特征布尔异常掩码、行级标签和事件级 JSON 元数据。
- CSV 数据、NPZ 掩码、配置快照、运行元数据和 SHA-256 数据哈希。
- 命令行可复现入口、单元/集成测试和示例图。
- 更新状态、任务清单、决策记录与会话交接。

### Excluded

- timestamp misalignment。
- 标准化、马氏距离、可信度映射或数据修正。
- Persistence、线性回归、LSTM、Transformer、PINN 或任何训练流程。
- 阿伦尼乌斯约束、风险预测和阈值预警。
- 区块链、哈希链、智能合约或界面。
- 真实工业数据结论。

## Milestones

### M0.1 仓库体检与计划

- 输入：全部仓库文件、项目规范和原始 Word 材料。
- 修改文件：`docs/plans/current_execplan.md`。
- 输出：自包含的本计划，记录现状、假设、范围、风险与验收标准。
- 运行命令：`find . -type f` 及逐文件阅读；Word 材料在临时目录渲染并提取文本。
- 测试：人工核对目录、文档约束和实际文件一致。
- 验收标准：计划不依赖丢失的聊天记忆，且明确排除后续模型与平台工作。

### M0.2 可复现干净时序仿真器

- 输入：`configs/milestone0.yaml` 的项目、数据和 simulation 参数。
- 修改文件：`src/__init__.py`、`src/config.py`、`src/simulator.py`、`tests/test_simulator.py`。
- 输出：固定 shape 的干净多序列数据，包含 split、condition、四个干净特征和质量真值。
- 运行命令：`python -m pytest tests/test_simulator.py -q`。
- 测试：相同 seed 完全一致、不同 seed 至少一列不同；字段、shape、有限性、范围、质量边界和完整序列 split 正确。
- 验收标准：全部测试通过，`0 <= quality <= 1`，每个 `sequence_id` 只属于一个 split。

### M0.3 三类污染、掩码与事件元数据

- 输入：M0.2 干净数据和 corruption 配置。
- 修改文件：`src/corruption.py`、`tests/test_corruption.py`。
- 输出：污染观测、形状为 `[n_sequence, n_time, n_feature]` 的布尔掩码，以及事件元数据列表。
- 运行命令：`python -m pytest tests/test_corruption.py -q`。
- 测试：三种类型均出现；掩码与实际变化逐元素一致；事件范围、特征、类型和幅度与掩码一致；干净输入未被原地修改；固定 seed 可复现。
- 验收标准：目标污染行数精确命中，且每个污染行只有一个特征被修改。

### M0.4 命令入口、产物和示例图

- 输入：仿真器、污染器和 `configs/milestone0.yaml`。
- 修改文件：`src/plotting.py`、`src/generate_phase1.py`、`tests/test_pipeline.py`。
- 输出：唯一 run ID 目录中的 `simulated_timeseries.csv`、`corruption_mask.npz`、`corruption_events.json`、`metadata.json`、`config_snapshot.yaml`，以及 `results/figures/<run_id>/corruption_examples.png`。
- 运行命令：`python -m src.generate_phase1 --config configs/milestone0.yaml`。
- 测试：临时目录端到端生成；文件存在且非空；CSV 与掩码行数/标签一致；元数据含 seed、配置哈希、数据哈希、代码版本和运行路径。
- 验收标准：命令退出码为 0；不覆盖已有 run；图中可人工辨识三类污染。

### M0.5 验证、审查与交接

- 输入：全部代码、测试和烟雾实验产物。
- 修改文件：`docs/STATUS.md`、`docs/TASKS.md`、`docs/DECISIONS.md`、`docs/SESSION_HANDOFF.md` 和本计划。
- 输出：真实测试结果、准确产物路径、限制和下一步唯一任务。
- 运行命令：`python -m pytest -q`；重新执行一次烟雾生成命令；检查文件与图像。
- 测试：全套测试；相同配置函数级复现测试；人工查看示例图；审查所有改动。
- 验收标准：测试全部通过、烟雾实验成功、文档与代码一致，未实现任何排除模块。

### M0.6 五类基础污染闭环（2026-07-19 范围增补）

- 输入：已验证的三类污染实现、用户确认的无真实数据状态、`configs/milestone0.yaml`。
- 修改文件：`src/config.py`、`src/corruption.py`、`src/plotting.py`、`src/generate_phase1.py`、相关测试和配置。
- 输出：新增连续缺失段和随机替换点；五类异常共享逐特征 mask、行级标签与事件元数据；异常强度参考值仅由 clean train 序列计算。
- 运行命令：`python -m pytest -q -p no:cacheprovider`；`python -m src.generate_phase1 --config configs/milestone0.yaml`。
- 测试：missing 仅在 mask 位置产生 NaN；random replacement 在 mask 位置产生有限且不同的训练参考值；允许的 NaN 与 missing mask 精确一致；五类事件、行标签、事件元数据和重跑结果一致。
- 验收标准：全量测试通过；五类污染均存在；目标污染行数精确；非 missing 观测均有限；最终新 run 不覆盖历史结果；示例图人工检查通过。

## Data leakage controls

- split 在序列层确定，一个 `sequence_id` 只能属于 train、val、test 中一个集合。
- 本轮不构造跨序列窗口，不存在窗口跨 split。
- 质量只由干净过程状态生成；污染注入发生在质量生成之后，防止污染标签意外改变预测真值。
- 后续标准化、协方差和阈值只能在 train 或正常 train 子集 `fit`；本轮产物保留 split 字段以支持该约束。
- 测试集只用于未来最终评价，本轮不据其结果调整异常幅度或工况参数。

## Reproducibility

- 所有随机过程由配置中的整数 seed 驱动，并使用显式 `numpy.random.Generator`。
- 运行时保存完整配置快照和规范化配置 SHA-256。
- 主 CSV 保存 SHA-256；元数据记录数据集 run ID、生成时间、seed、数据哈希和代码版本状态。
- 每次命令生成唯一 run ID 目录，若目录存在则立即失败而非覆盖。
- 当前无 Git 仓库，元数据明确记录代码版本不可用；初始化 Git 不属于本轮任务。
- 单元测试在临时目录运行，不污染正式 `data/` 和 `results/`。

## Validation

- 单元测试：配置校验、仿真 shape/范围/复现性、异常注入类型/幅度/掩码/元数据。
- 集成测试：从 YAML 到 CSV、NPZ、JSON 和 PNG 的完整命令内核。
- 烟雾实验：使用 `configs/milestone0.yaml` 生成一份小规模正式产物。
- 人工检查：查看示例图，确认 spike 为瞬时脉冲、bias 为平台偏移、drift 为渐变偏移；核对 metadata、CSV 和掩码一致。
- 差异审查：确认没有 LSTM、PINN、区块链或界面代码，没有覆盖原始材料或既有结果。

## Acceptance criteria

1. `python -m pytest -q` 全部通过。
2. 相同配置和 seed 的函数级仿真与污染结果逐值一致。
3. 四个干净特征和 `quality` 有限；观测特征除 masked missing 外均有限。
4. 掩码与 `clean != observed` 逐元素一致，目标污染行数精确，五类异常均有事件记录。
5. CSV、NPZ、事件 JSON、元数据、配置快照和示例 PNG 全部实际生成且非空。
6. 每条序列只属于一个 split；污染不改变 clean、quality、condition 或 split。
7. 结果目录唯一且不覆盖历史运行。
8. 状态、任务、决策、handoff 和本计划与真实执行结果一致。

## Progress

- [x] 2026-07-19：完整读取入口文件、研究文档、配置、提示词、项目技能和两份 Word 原始材料；确认仓库无源码、测试、结果和 Git 元数据。
- [x] 2026-07-19：形成 Milestone 0 自包含实施计划。
- [x] 2026-07-19：完成 M0.2；四变量、多工况和质量递推测试通过。
- [x] 2026-07-19：完成 M0.3；三类污染精确命中，掩码与实际改动逐元素一致。
- [x] 2026-07-19：完成 M0.4；命令入口生成 CSV、NPZ、JSON、YAML 和示例 PNG。
- [x] 2026-07-19：完成 M0.5；5 项测试通过，最终烟雾实验和人工图检通过。
- [x] 2026-07-19：完成 M0.6；补充 missing、random replacement 与 clean train 参考统计，6 项测试和最终新 run 通过。

## Decision log

- 2026-07-19：时间步采用抽象采样单位，因为真实采样频率未知；保留 `delta_t` 配置供后续校准。
- 2026-07-19：第一轮主数据用 CSV、掩码用压缩 NPZ、元数据用 JSON；避免为 Parquet 新增 `pyarrow`。
- 2026-07-19：污染比例按序列-时间行定义，每行最多一种污染；这与 `DATA_SPEC.md` 的行级 `is_corrupted` 和单值 corruption 字段一致。
- 2026-07-19：新增小规模 `milestone0.yaml`，不改写面向后续完整实验的 `baseline.yaml`。
- 2026-07-19：质量递推属于可控经验仿真，不引入或暗示阿伦尼乌斯、PINN 或真实机理有效性。
- 2026-07-19：异常幅度和随机替换候选值仅从 clean train 序列得到，避免测试序列参与数据生成参考统计。
- 2026-07-19：真实数据状态为 `Pending external data / 等待外部数据`；本计划只保留接口与字段说明，不实现真实数据依赖代码。

## Surprises and discoveries

- 仓库不是 Git 工作树，虽然规范要求记录 commit；本轮只能显式记录“无 Git 版本”。
- 原始材料将一阶质量变化率称为“下降加速度”，并将区块链上链描述为保证输入“绝对真实”；项目主文档已经明确纠正这两点，实施严格遵循主文档。
- 原始材料强调动态图物理参数和贝叶斯优化，但这些会显著扩大范围且削弱第一轮可归因性，因此全部延后。
- 当前机器默认 Python 3.14 且无依赖；创建项目 `.venv` 后安装本轮最小依赖完成验证，但仍需在规范建议的 Python 3.11 环境复验。
- 初版图的 drift 上下文包含另一处 spike，缩短绘图上下文后重新生成；历史 run 按不覆盖规则保留。
- 空字符串异常特征标签在 CSV 重读时成为 NaN，最终改为显式 `none` 并增加“无意外缺失”集成测试。
- 审计发现旧三类实现用全部 clean split 计算异常尺度；M0.6 已改为仅使用 clean train。
- CSV 原使用 10 位有效数字，事件 JSON 保留完整精度；为支持跨文件数值审计，已改为 17 位有效数字。

## Outcomes and retrospective

Milestone 0 与 M0.6 已完成。实际交付包括配置校验、可复现四变量多序列仿真器、离散质量递推、五类异常注入、逐特征布尔掩码、事件元数据、唯一 run 持久化、五类示例图和 6 项自动测试。最终 run 为 `milestone0_20260719T153039079827Z_0abad9ce`，数据 4800 行、17 列、无重复，480 个污染时刻与 mask/事件完全一致；其中 96 个 NaN 仅对应 missing，随机替换仅引用 clean train。独立重跑的 CSV、mask、事件和图完全一致。本计划扩展后的验收标准均满足，但 Python 3.11 复验、真实采样语义、真实质量定义和外部有效性仍待后续处理。下一轮只进入训练集预处理和标准化，不同时实现可信度或预测模型。

---

# Milestone 1 执行增补：无泄漏缺失值插补与多维标准化

## Purpose

本增补交付后续可信度计算所需的最小预处理闭环：从已验证的 Milestone 0 仿真长表读取四个污染观测特征，仅用 clean train 真值拟合每个特征的缺失值填充值、均值和样本标准差，再对 train/val/test 观测数据执行相同转换。输出保留三维序列结构、split 和输入数据身份，并保存参数、标准化数组、元数据、配置快照和哈希。本轮不计算马氏距离、可信度或预测模型。

## Current state

- 输入数据固定为 `milestone0_20260719T153039079827Z_0abad9ce`，共 30 条完整序列、每条 160 步、4 个特征。
- train/val/test 为 18/6/6 条完整序列；输入观测包含 96 个由 missing 异常产生的 NaN。
- 当前仓库还没有 `src/preprocess.py`、预处理测试或预处理结果目录。
- 仓库仍未初始化 Git；VS Code 登录不产生 `.git/`，代码版本仍只能记录为 `unavailable_no_git_repository`。
- 真实数据接入继续保持 `Pending external data / 等待外部数据`。

## Scientific assumptions

1. 第一版 missing 插补采用逐特征常量中位数。中位数比均值更不容易被极端值影响，且行为简单、可解释、可复现。
2. 插补值从 clean train 真值计算。这是仿真阶段的受控基线，用于隔离预处理实现正确性；不得解释为真实部署时可获得被污染点真值。
3. 标准化中心和尺度同样只从 clean train 计算：`z=(x-mean_train)/std_train`，标准差使用样本标准差 `ddof=1`。
4. train/val/test 的污染观测全部只调用 `transform`；验证集和测试集的数值变化不能影响已拟合参数。
5. 预处理不修改输入 CSV，不覆盖 Milestone 0 结果；输出建立唯一的 `preprocess_<timestamp>_<config_hash>` 目录。
6. 本轮只验证预处理正确性，不使用异常标签拟合参数，不评价异常识别效果。

## Scope

### Included

- 可序列化的预处理参数对象；
- clean train 中位数插补参数、均值和样本标准差；
- 对任意形状 `[..., feature]` 数组执行一致的插补和标准化；
- 从 Milestone 0 CSV 恢复完整序列顺序并验证 split 隔离；
- 输出标准化观测数组、序列 ID、时间索引、split、参数 JSON、配置快照和运行元数据；
- 无泄漏、无 NaN、shape/顺序保持、参数持久化和不覆盖测试。

### Excluded

- 前向填充、插值或模型型插补方法比较；
- 收缩协方差、马氏距离、q90/q99、连续可信度和 PCA；
- 异常修正、滑动窗口、预测模型、风险、区块链和界面；
- 任何真实数据实验或工业有效性结论。

## Milestones

### M1.1 预处理器与配置

- 输入：Milestone 0 的 clean train 数组与观测数组。
- 修改文件：`src/preprocess.py`、`configs/milestone1_preprocess.yaml`。
- 输出：`fit`/`transform` API、参数 JSON 表示和配置校验。
- 测试：参数只等于 clean train 的直接计算结果；transform 后全部有限；输入不被原地修改。
- 验收标准：变更 val/test 数值不改变任何拟合参数，相同输入重复转换逐值一致。

### M1.2 命令入口与结果持久化

- 输入：最新 Milestone 0 数据目录和 M1 配置。
- 修改文件：`src/run_preprocessing.py`、端到端测试。
- 输出：唯一 run 目录中的 `standardized_observations.npz`、`preprocessor_params.json`、`metadata.json` 和 `config_snapshot.yaml`。
- 运行命令：`python -m src.run_preprocessing --config configs/milestone1_preprocess.yaml`。
- 测试：输入数据哈希匹配、序列和时间索引完整、split 不交叉、输出无 NaN/Inf、重复 run ID 拒绝覆盖。
- 验收标准：正式烟雾运行成功，全部输出存在且非空，元数据明确 `fit_split=clean_train_only` 和 `data_source=synthetic`。

### M1.3 验证与交接

- 输入：全部源码、测试和正式预处理结果。
- 修改文件：README、STATUS、TASKS、DECISIONS、SESSION_HANDOFF 和本计划。
- 输出：真实命令、测试结果、路径、限制和下一步唯一任务。
- 验收标准：全套测试通过，产物人工/程序核对一致，未提前实现可信度或模型。

## Data leakage controls

- 由 CSV 的 `sequence_id` 和 `split` 恢复完整序列；每条序列必须且只能属于一个 split。
- 插补中位数、标准化均值和标准差只从后缀为 `_clean` 且 split 为 `train` 的四列计算。
- `_obs` 列只作为 transform 输入；异常标签、验证集和测试集不得参与 fit。
- 单元测试显式构造极端 val/test 值，确认拟合参数完全不变。
- 输出保留序列 ID、时间索引和 split，禁止在本轮生成跨序列窗口。

## Reproducibility

- 本预处理为确定性操作，不新增随机过程。
- 配置保存输入 run ID、特征顺序、插补策略、标准差 `ddof` 和输出根目录。
- 结果记录输入 CSV SHA-256、配置 SHA-256、输出 NPZ SHA-256、代码版本和绝对路径。
- 每次正式运行使用唯一 run ID；已有目录立即报错，不覆盖历史结果。

## Validation

- 单元测试：fit 数值、transform 公式、NaN 插补、输入不变、零尺度拒绝或保护。
- 泄漏测试：修改 val/test 后参数不变；只允许 clean train 进入 fit。
- 集成测试：CSV 到 NPZ/JSON/YAML 的完整链路、序列顺序、split 隔离、参数快照和不覆盖。
- 烟雾实验：对当前最终 Milestone 0 run 执行一次正式预处理并审计全部文件。

## Progress

- [x] 2026-07-28：恢复仓库状态并确认 VS Code 登录后仓库仍无 Git 元数据。
- [x] 2026-07-28：确定第一版采用 clean-train feature median 插补与 clean-train mean/sample-std 标准化。
- [x] 2026-07-28：完成 M1.1；实现预处理器、配置与参数序列化。
- [x] 2026-07-28：完成 M1.2；实现唯一 run 入口、NPZ/JSON/YAML 持久化和 3 项新增测试。
- [x] 2026-07-28：完成 M1.3；全套 9 项测试通过，正式 run 和独立重跑审计通过。

## Decision log

- 2026-07-28：第一版 missing 基线选择逐特征训练中位数，而不是前向填充。原因是中位数实现简单、对异常值稳健、没有序列开头无历史值的边界问题；时间连续性的优劣留给后续单因素对照实验。
- 2026-07-28：本轮使用 clean train 拟合参数，目的是建立受控仿真基线并严格排除 split 泄漏；这一选择不声称真实部署可访问缺失点真值。
- 2026-07-28：不因已登录 VS Code 而自动初始化 Git；版本控制属于独立、会改变仓库状态的操作，等待用户明确授权。

## Surprises and discoveries

- VS Code 账户登录没有为当前目录创建 `.git/`，因此项目仍不是 Git 工作树。
- 第一遍 `compileall` 尝试写仓库 `__pycache__` 时被沙箱拒绝；指定 `PYTHONPYCACHEPREFIX=/tmp/petrochemical_pycache` 后编译通过，确认不是代码语法错误。

## Outcomes and retrospective

Milestone 1 已完成。实际交付包括 `src/preprocess.py`、`src/run_preprocessing.py`、独立 YAML 配置、3 项新增测试和唯一结果目录。正式 run `preprocess_20260728T101115651491Z_40442611` 将 96 个 synthetic missing NaN 全部插补，标准化后非有限值为 0，数组 shape 保持 `(30, 160, 4)`；拟合参数与 clean train 直接计算一致，独立重跑的数组和参数逐值一致。最新全套测试为 `9 passed in 1.56s`。验收标准全部满足，未实现马氏距离、阈值、预测模型或其他排除模块。下一轮只进入标准化 clean train 参考集合与收缩协方差估计。

---

# Milestone 2 执行增补：正常参考集合与收缩协方差

## Purpose

本增补建立 E1 可信度识别的统计参考基础：从 Milestone 0 长表确定完整的 `train + normal` 序列，从 Milestone 1 保存的预处理参数重新标准化这些序列的 clean 真值，形成不含人工传感器污染的正常训练参考矩阵；随后使用 Ledoit–Wolf 方法估计收缩协方差、精度矩阵和中心，并保存全部参数、输入身份和数值诊断。本轮不计算马氏距离、阈值或异常指标。

## Current state

- Milestone 1 已输出 `(30,160,4)` 的标准化污染观测，但 NPZ 没有保存标准化 clean 数组和 condition。
- Milestone 0 CSV 完整保存 clean 特征、split 和 condition，可与 Milestone 1 的参数和数据身份交叉核对。
- 当前 `.venv` 最初缺少已在 `requirements.txt` 声明的 scikit-learn；2026-08-13 已安装 scikit-learn 1.9.0 及其依赖。
- 仓库仍无 Git 元数据，真实数据继续为 `Pending external data / 等待外部数据`。

## Scientific assumptions

1. 第一版正常参考集合只使用 `split=train` 且 `condition=normal` 的完整序列；不使用人工异常标签筛选参考点。
2. 参考输入采用 clean 真值而非污染观测，这是 synthetic 阶段用于验证统计实现的受控 oracle baseline，不代表真实部署能访问隐藏真值。
3. clean 参考值必须使用 Milestone 1 保存的全部 clean train 参数标准化，不能在正常子集上重新计算均值或尺度。因此非 normal train 会通过上游标准化参数间接影响参考坐标，但不会直接进入协方差拟合行；这是既定训练预处理定义，不是 val/test 泄漏。
4. 协方差采用 scikit-learn `LedoitWolf(assume_centered=False)`；该方法将经验协方差向缩放单位阵收缩，以提高相关特征或有限样本下的可逆性和数值稳定性。
5. 本轮保存精度矩阵仅用于数值审计和下一阶段距离计算；不据此选择 q90/q99，也不报告异常识别性能。

## Scope

### Included

- `train + normal` 完整序列选择与身份验证；
- 使用已保存预处理参数转换 clean 参考值；
- Ledoit–Wolf 中心、协方差、精度矩阵和收缩系数；
- 对称性、正定性、条件数和 `covariance @ precision ≈ I` 数值诊断；
- JSON/NPZ/YAML/metadata 持久化、输入/输出哈希和唯一 run；
- 单元测试、泄漏测试、集成测试和独立重跑审计。

### Excluded

- 马氏距离、q90/q99、连续可信度、高/中/低分组；
- PCA、Precision/Recall/F1/PR-AUC；
- 异常修正、预测模型、物理约束、风险、区块链和界面；
- 真实数据实验和工业有效性结论。

## Milestones

### M2.1 参考估计器

- 输入：标准化 clean-train-normal 二维矩阵。
- 修改文件：`src/trust_reference.py`、`tests/test_trust_reference.py`。
- 输出：中心、Ledoit–Wolf 协方差、精度矩阵、收缩系数和诊断。
- 测试：与 scikit-learn 直接拟合结果一致；矩阵有限、对称、正定且可逆；输入不被修改。
- 验收标准：改变 val/test 或非 normal train 不改变拟合参数，相同参考矩阵产生完全一致结果。

### M2.2 命令入口和产物

- 输入：Milestone 0 CSV、Milestone 1 NPZ/参数/metadata 和独立 M2 配置。
- 修改文件：`src/run_trust_reference.py`、`configs/milestone2_trust_reference.yaml`、集成测试。
- 输出：`reference_set.npz`、`trust_reference_params.json`、`metadata.json`、`config_snapshot.yaml`。
- 运行命令：`python -m src.run_trust_reference --config configs/milestone2_trust_reference.yaml`。
- 验收标准：输入哈希和 dataset ID 一致；只选择 train-normal 完整序列；结果目录不覆盖；输出参数和数组可独立重算。

### M2.3 正式实验与记录

- 输入：全部代码、测试和正式 M2 run。
- 修改文件：README、`docs/project_log.md`、STATUS、DECISIONS、TASKS、SESSION_HANDOFF 和本计划。
- 验收标准：全套测试与语法编译通过；独立重跑参数一致；结果只表述为 synthetic 统计参考估计，不越界声称异常识别有效。

## Data leakage controls

- split 和 condition 只从不可变的 Milestone 0 CSV读取；参考序列必须同时满足 `split=train`、`condition=normal`。
- 必须按完整 sequence ID 选择，不从 val/test 或非 normal train 抽取任何时间点。
- 标准化参数只读取 Milestone 1 的 clean-train 参数，不重新 fit。
- 人工 `is_corrupted`、`corruption_type` 和 mask 不参与参考集合筛选或协方差拟合。
- 单元测试显式改变 val/test 数据，确认参数不变。非 normal train 会参与既定的 Milestone 1 标准化参数，不宣称其变化对 M2 参数无影响。

## Reproducibility

- Ledoit–Wolf 为确定性估计，本轮不新增随机种子。
- 配置固定两个输入 run ID、各文件名和 SHA-256、特征顺序、split、condition 和 estimator 参数。
- 结果保存参考 sequence IDs、标准化参考矩阵、中心、协方差、精度、特征顺序和全部哈希。
- 每次正式运行使用唯一 `trustref_<timestamp>_<config_hash>` 目录；已有 run ID 拒绝覆盖。

## Validation

- 单元测试：已知矩阵拟合、scikit-learn 一致性、正定性、逆矩阵残差和无输入修改。
- 泄漏测试：极端改变非参考 split/condition 后参数不变。
- 集成测试：两个上游 run 到 NPZ/JSON/YAML 的完整链路、序列筛选、哈希、参数重算和不覆盖。
- 正式烟雾实验：当前最终 M0/M1 输入运行一次，独立 `/tmp` 重跑并比较参数与参考数组。

## Progress

- [x] 2026-08-13：恢复 M0/M1 状态，确认 `docs/project_log.md` 尚不存在且仓库仍无 Git。
- [x] 2026-08-13：补装 requirements 已声明但环境缺失的 scikit-learn 1.9.0。
- [x] 2026-08-13：确定参考集合为完整 `clean train + normal` 序列，估计器为 Ledoit–Wolf。
- [x] 2026-08-13：完成 M2.1；实现参考估计器、诊断和 sklearn 一致性测试。
- [x] 2026-08-13：完成 M2.2；实现输入哈希/身份校验、唯一 run、四类产物和端到端测试。
- [x] 2026-08-13：完成 M2.3；12项测试、正式 run、独立重跑和文档更新完成。

## Decision log

- 2026-08-13：只使用 `train + normal` 完整序列作为正常参考，避免把高温/高振动/复合工况混入第一版正常统计集合。
- 2026-08-13：synthetic 阶段使用 clean 真值建立 oracle 参考基线，目的在于先验证统计方法；未来真实部署必须改用经确认的正常训练观测，不能假设存在隐藏真值。
- 2026-08-13：使用 scikit-learn 官方 Ledoit–Wolf 实现，不手写收缩公式，以减少数值和统计实现错误。

## Surprises and discoveries

- `requirements.txt` 已声明 scikit-learn，但当前 `.venv` 未安装；安装 1.9.0 后才具备 Ledoit–Wolf 实现。
- 第一轮集成测试发现 pandas 导出的 reference sequence IDs 被保存为 NPZ object 数组，默认 `allow_pickle=False` 无法安全加载；已改为 Unicode 数组并增加 dtype 断言。

## Outcomes and retrospective

Milestone 2 已完成。正式 run `trustref_20260813T074437348987Z_34d5fa5c` 从5条完整 normal train synthetic clean 序列建立800×4参考矩阵，Ledoit–Wolf收缩系数为`0.0036545028372638863`。协方差最小特征值为`0.0051141175843415695`、条件数为`272.85259024321965`，逆矩阵单位阵最大误差为`2.4868995751603507e-14`；全部值有限且协方差正定。首次集成测试暴露NPZ object序列ID安全加载失败，修复为Unicode后专项3项和最终全套12项测试通过（`12 passed in 2.92s`）。独立重跑的参考数组和参数逐值一致。验收标准满足，但本次只证明统计参考估计实现正确、数值稳定和可复现，未证明异常识别有效。下一轮只计算平方马氏距离，不同时设置阈值或报告检测指标。

---

# Milestone 3 执行增补：平方马氏距离

## Purpose

本增补把 Milestone 2 已冻结的正常参考中心和精度矩阵应用到 Milestone 1 的全部标准化污染观测，计算每个序列、每个时间点的平方马氏距离 `d²`，并同时计算正常参考集自身的 `d²` 以供下一轮只用训练参考分布选择阈值。本轮交付公式实现、输入身份校验、可复现结果文件和测试；不选择 q90/q99、不映射连续可信度，也不报告异常检测性能。

## Current state

- M1 run `preprocess_20260728T101115651491Z_40442611` 已保存 `(30, 160, 4)` 的有限标准化观测、完整 sequence/time/split 身份和特征顺序。
- M2 run `trustref_20260813T074437348987Z_34d5fa5c` 已保存 800×4 正常参考矩阵、中心、正定收缩协方差和精度矩阵。
- 当前 12 项测试通过；尚无马氏距离实现、距离产物、阈值、可信度或检测指标。
- 仓库已于 2026-08-14 初始化 Git，`main` 已跟踪 GitHub `yanyan72/potrochemical`；历史产物中的无 Git 版本标记保持原样。
- 真实数据仍为 `Pending external data / 等待外部数据`。

## Scientific assumptions

1. 对标准化特征向量 `x` 使用平方马氏距离 `d²=(x-μ)^T P (x-μ)`，其中 `μ` 和精度矩阵 `P=Σ⁻¹` 完全读取 M2，不在 M3 重新拟合。
2. `d²` 保留平方尺度，避免本轮无必要的开方；数值上只允许把浮点舍入产生的极小负值裁剪为 0，显著负值应报错。
3. M1 的 train/val/test 全部只做 transform。val/test 不参与中心、协方差、精度或本轮任何阈值拟合。
4. 参考集距离用于下一轮形成训练参考距离分布，但本轮不计算分位数，避免把距离实现与阈值选择混在同一小步。
5. 输入仍全部为 synthetic；距离大不等于已经证明异常，必须等阈值、标签评估和多异常类型分析完成后才能讨论识别效果。

## Scope

### Included

- 向量化平方马氏距离函数，保留任意前导 shape；
- 公式、shape、有限性、非负性、输入不修改和手工计算一致性测试；
- M1/M2 run ID、SHA-256、特征顺序、序列顺序、split、中心和精度一致性校验；
- 全部标准化观测及参考集距离的 NPZ 持久化；
- 唯一 run、配置快照、元数据、输出哈希和 Git code version；
- 正式 synthetic run、独立临时目录重跑和文档交接。

### Excluded

- q90/q99 或其他阈值；
- high/uncertain/low 分组和连续可信度；
- Precision、Recall、F1、PR-AUC、PCA 或混淆矩阵；
- 数据修正、预测模型、物理约束、风险、区块链或界面；
- 真实数据实验和工业结论。

## Milestones

### M3.1 距离函数

- 输入：形状 `[..., feature]` 的有限标准化数组、M2 中心和精度矩阵。
- 修改文件：`src/trust_score.py`、`tests/test_trust_score.py`。
- 输出：与显式二次型一致、shape 保持且非负有限的 `d²`。
- 测试：手工公式一致、中心距离为0、三维输入输出二维、输入不变、非法 shape/非有限值/非正定精度拒绝。
- 验收标准：确定性、无参数拟合、数值误差处理有明确边界。

### M3.2 命令入口与产物

- 输入：固定的 M1 NPZ/metadata 和 M2 NPZ/params/metadata。
- 修改文件：`src/run_trust_scoring.py`、`configs/milestone3_mahalanobis.yaml`、`tests/test_trust_scoring_pipeline.py`。
- 输出：`mahalanobis_distances.npz`、`metadata.json`、`config_snapshot.yaml`。
- 运行命令：`python -m src.run_trust_scoring --config configs/milestone3_mahalanobis.yaml`。
- 验收标准：观测距离 shape 为 `(30,160)`，参考距离为 `(800,)`，全部有限非负；输入身份或哈希不一致立即失败；已有 run 不覆盖。

### M3.3 正式运行、复现与记录

- 输入：通过测试的实现和固定配置。
- 修改文件：README、`docs/project_log.md`、STATUS、TASKS、DECISIONS、ARCHITECTURE、SESSION_HANDOFF 和本计划。
- 验收标准：全套测试、语法编译和正式 run 成功；独立重跑数组逐值一致；文档只陈述距离计算正确性，不越界声称异常识别有效。

## Data leakage controls

- M3 不调用任何 `fit`；中心和精度只来自 M2 的完整 `train + normal + synthetic clean` 参考。
- 标准化观测只读取 M1 产物，而 M1 参数只来自 clean train。
- val/test 只应用既定中心和精度，不能影响任何统计参数。
- 人工异常标签和 mask 不参与距离计算；它们只保留给后续独立评价。
- 下一轮阈值只能从参考集或正常训练距离分布估计，不能从 val/test 标签调参。

## Reproducibility

- 距离计算为确定性向量运算，不新增随机种子。
- 配置冻结上游 run ID、文件名、SHA-256、特征顺序和距离类型。
- 输出记录输入/输出哈希、shape、有限性/非负性诊断、绝对路径和 Git commit。
- 正式结果使用唯一 `trustscore_<timestamp>_<config_hash>` 目录；独立重跑写入 `/tmp`，不覆盖正式结果。

## Validation

- 单元测试：公式、shape、边界、输入不变和异常输入。
- 集成测试：上游三阶段生成到距离产物、身份核验、哈希和不覆盖。
- 正式实验：固定当前 M1/M2 产物运行一次。
- 复现审计：独立 `/tmp` 重跑，比较全部 NPZ 数组逐值一致，并比较除运行时间/路径/run ID 外的确定性元数据。

## Progress

- [x] 2026-08-14：恢复仓库、Git、M0–M2、测试与正式结果状态，确定本轮只实现平方马氏距离。
- [x] 2026-08-14：完成 M3.1 距离函数和3项单元测试，公式、shape、有限性、正定性与输入不变检查通过。
- [x] 2026-08-14：完成 M3.2 命令入口、配置和端到端身份核验测试；专项4项、全量16项测试通过。
- [x] 2026-08-14：完成 M3.3 正式运行、独立复现、文档和 Git 记录；正式结果记录代码提交 `e1748bf`，8个 NPZ 数组重跑逐值一致，最终交付提交推送至 GitHub `main`。

## Decision log

- 2026-08-14：本轮保存 `d²` 而不是 `d`，与项目公式和后续基于参考距离分布的 q90/q99 定义保持一致。
- 2026-08-14：距离阶段与阈值阶段分开，确保本轮只验证二次型实现和输入身份，不利用标签选择阈值。

## Surprises and discoveries

- 状态文档仍记录“仓库无 Git”，但实际 Git 已于 2026-08-14 初始化并推送；M3 完成记录中需修正当前状态，同时保留历史 run 的原始 code version。

## Outcomes and retrospective

Milestone 3 已完成。实际交付包括平方马氏距离函数、冻结 M2 参数的 transform-only 命令入口、独立配置、4项新增测试、输入身份/哈希核验和唯一结果目录。正式 run `trustscore_20260814T140937863216Z_09a4bd41` 生成 `(30,160)` 的观测距离和 `(800,)` 的参考距离，所有值有限非负；最新全套16项测试和语法编译通过，独立重跑的8个NPZ数组逐值一致。该结果只证明公式实现、输入链路和复现性正确，尚未证明异常识别有效。下一轮只从正常训练参考距离估计q90/q99并划分可信组，不同时实现连续可信度或评价指标。
