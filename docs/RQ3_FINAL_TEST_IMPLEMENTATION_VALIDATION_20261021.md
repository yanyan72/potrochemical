# 冻结测试实现验证（正式test预测前）

协议先推送：82d3fff44bba01628a250af31f765b0f8c2cdfdb。原发生器、岭回归和LSTM模块未修改，仅新增独立只读推断/评价入口。

相关四个测试文件执行：42 passed in 12.82s。覆盖原数据与预测测试，以及新授权guard、标签ID、配对误差、五seed Student-t区间手算、隔离夹具端到端及禁止fit检查。随后将独立复核脚本纳入隔离集成测试，再执行新文件的5项测试，日志单独保存。这些是相关专项测试，不是全仓库回归。

临时夹具明确标为isolated_final_test_fixture，2seed×5split×16批，共160批，独立于正式10,000批；测试里的全零commit只在临时目录。测试最终入口不重新拟合模型，标签必须在预测文件保存后读取，完整输出清单和独立算术复核通过。测试不读取正式test性能。

正式开发val重放检查：五个seed×两模型家族×十视图×七组×两目标，与原存档预测逐元素最大绝对差均为0。新入口正确复现旧模型推断，无重新拟合、无test预测。记录见results/rq3_prediction/rq3_final_test_20261021_val_replay.json及相关日志。

实现/复核脚本经过编译和差异检查。正式test执行要求显式--execute-frozen-test以及协议、实现提交ID；创建test_use标记后仅加载既有模型推断。test开始使用后不得再称未见测试。配置、源文件、数据、模型的固定身份在执行前后核对。

正式命令模式：

```text
python -m src.rq3_final_test --data <existing_data> --ridge <ridge_run> --lstm <lstm_run> --config configs/rq3_final_test_20261021.yaml --output results/rq3_prediction/rq3_final_test_20261021_v1 --protocol-commit 82d3fff44bba01628a250af31f765b0f8c2cdfdb --implementation-commit <published_commit> --execute-frozen-test --seed <42..46>
```

本验证文档不包含正式test结果；实现在本阶段推送后才执行。
