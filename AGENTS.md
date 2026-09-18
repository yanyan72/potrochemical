# AGENTS.md

## Project mission

当前主线（用户2026-09-18确认）：石化产品运输与仓储中的工况感知传感器级可信度及质量预测。生产仅为未来推广；真实数据后续提供，先做明确标记的模拟算法。原全生命周期、区块链和物理扩展不作为本篇必做内容。核心贡献必须由可验证算法和实验支持。

## Read first

开始任何任务前，按顺序阅读：

1. `docs/PROJECT_BRIEF.md`
2. `docs/ARCHITECTURE.md`
3. `docs/DATA_SPEC.md`
4. `docs/EXPERIMENT_PLAN.md`
5. `docs/TASKS.md`
6. `docs/DECISIONS.md`
7. `docs/STATUS.md`
8. `.agents/PLANS.md`
9. 与当前任务直接相关的源代码和测试

复杂功能、跨模块修改或预计超过 1 小时的任务，必须先创建或更新 `docs/plans/current_execplan.md`，并按照 `.agents/PLANS.md` 执行。

## Scientific priorities

优先级从高到低：

1. 仿真数据和异常标签正确；
2. 可信度算法可解释且可评估；
3. 基线模型和数据划分无泄漏；
4. 可信度增强的增益可由消融实验归因；
5. 动力学约束具有明确数学含义；
6. 区块链只承担完整性与追溯，不替代数据真实性判断；
7. 结果可复现；
8. 最后才考虑界面和演示系统。

## Non-negotiable corrections

- `dQ/dt` 是变化速率，不是加速度。
- 离散 LSTM 第一版使用有限差分或离散递推，不虚构连续时间自动微分。
- 不宣称“上链即真实”。
- 动态 `A_t`、`E_a,t` 必须受约束并进行可辨识性分析。
- PCA 默认用于可视化；异常判定使用训练集拟合的有效特征空间。
- 标准化、协方差、阈值只从训练集计算。
- 不伪造真实数据、引用、指标或实验结论。

## Repository layout

- `src/`：可复用源代码
- `tests/`：单元与集成测试
- `configs/`：实验配置
- `data/raw/`：原始或只读数据
- `data/processed/`：处理后的数据
- `results/figures/`：图
- `results/tables/`：表
- `results/models/`：模型
- `docs/`：研究说明、实验计划、状态和决策
- `docs/plans/`：执行计划
- `.agents/skills/`：项目技能

## Engineering rules

- Python 3.11。
- 优先使用 PyTorch、NumPy、pandas、scikit-learn、SciPy、matplotlib、PyYAML、pytest。
- 公共函数必须有类型标注和 docstring。
- 新算法必须附测试；重要数学公式必须做数值测试。
- 所有随机过程必须允许传入 seed。
- 所有实验必须从配置文件读取参数。
- 所有图表和指标必须自动落盘。
- 不覆盖已有实验目录；使用时间戳或实验 ID。
- 不把核心逻辑只写在 Notebook。
- 不随意增加依赖；新增依赖需说明理由。
- 修改后运行相关测试、最小训练或烟雾测试。
- 任何性能声明必须附实验文件路径和配置。

## Data rules

- 按完整序列划分 train/validation/test。
- 不允许同一序列的窗口跨集合。
- 清洁真值和污染观测分开保存。
- 异常注入必须返回掩码、类型、幅度和时间范围。
- 测试集不用于阈值选择或调参。
- 数据预处理器只在训练集 `fit`，验证/测试仅 `transform`。

## Experiment rules

- 质量目标冻结后至少包含线性回归和普通 LSTM；持久性仅在历史质量结果于预测时已经可用时使用，不读取潜在真实质量。传感器评分阶段不虚构质量标签。
- 每次实验只改变一个主要因素。
- 主要结果至少 5 个随机种子。
- 报告均值、标准差和逐工况指标。
- 失败结果也要保存并解释。
- 所有表格记录配置、数据版本和代码版本。

## Definition of done

任务完成前必须：

1. 运行相关测试；
2. 运行最小可复现实验；
3. 检查输出文件；
4. 审查 diff；
5. 更新 `docs/STATUS.md`；
6. 如有新假设或取舍，更新 `docs/DECISIONS.md`；
7. 在回复中列出修改、命令、结果、限制和下一步。
