# v0.1 批量评分使用说明

## 适用范围
这是可复用的模拟评分组件。预置模型针对示意温度/湿度/振动和三种工况训练，不能直接用于真实产品安全判定。真实接入需另做字段映射、可靠训练样本识别、训练/校准及外部评价。

默认：分工况边际残差、因果有符号EWMA alpha=0.2、train_cal逐工况逐通道q99，序列开始/可见记录改变/缺失清零。条件残差、逐点评分可使用最终产物中的其他bundle；SMA/CUSUM/carry保留研究实现，不作为本组件选项。

## 一次运行
在仓库根目录，Python3.11安装requirements-core.txt后：

```bash
python -m src.score_batch \
  --model examples/scoring/model.json \
  --observations examples/scoring/observations.csv \
  --contract examples/scoring/contract.json \
  --output results/my_first_score
```

输出目录必须不存在，防止覆盖。生成`scores.csv`和`metadata.json`。仓库`examples/scoring/output/`保存示例的预期输出；示例是手工构造接口演示，不属于性能实验。重新导出演示资产：

```bash
python -m src.prepare_scoring_demo --output results/my_demo_assets
```

## 输入契约

| 字段 | 要求 |
| --- | --- |
| sequence_id | 非空字符串；每个ID表示一段独立完整轨迹，不允许把连续轨迹分多次调用 |
| time_step | 非负整数，同一ID内严格递增且连续，允许从任意非负步开始 |
| recorded_regime | 该时刻已可用的工况记录；已知值warehouse/transit_smooth/transit_rough；未识别值原样输出unknown_regime |
| temperature | K；允许空单元格/NaN/nan表示缺失 |
| relative_humidity | percent_RH；同上 |
| vibration | mm_per_s；同上 |

CSV仅允许这些列，不接受隐藏真值、污染mask或质量标签。数值列中的错误字符串、无穷、重复列、乱序/重复/跳步时间、空ID/记录均拒绝。整个采样时刻缺测也须补一行显式NaN，组件不会静默填充缺步。输入行顺序保留，不自动排序。

contract.json必须与模型声明的schema、synthetic来源、特征顺序、单位、abstract_sampling_step及sample_interval=1完全一致。不自动猜测摄氏/开尔文或真实采样频率；它是调用者声明的契约，不能证明数据内容确实使用正确单位。

## 输出含义

每个输入行产生3条传感器记录：输入行号/序列/时间/记录工况/传感器/单位/原观测、consistency_score、deviation_ratio、available、alarm、status。

- `deviation_ratio=abs(filtered_signed_residual)/training_threshold`，大于1告警。
- `consistency_score=exp(-0.5*deviation_ratio**2)`，它不是正确概率、合格概率或质量值。
- `available=false`时分数和alarm为空。不可用不能解释为正常；状态说明unknown_regime、missing_sensor或missing_dependency。
- 边际方法的某传感器缺失只影响该通道；条件残差缺失任一输入会使整行不可评分。
- 内部研究API用alarm=false标记不可用位置，CSV接口改为空值并保留available，防止误读为“确认正常”。

每次调用独立初始化；不提供跨文件增量状态接口。工况记录变更只按当前可见记录处理，不能用事后真实状态回填历史。EWMA有响应迟滞与拖尾，reset也会损失跨切换事件历史；详见总报告。

## 模型和追溯
JSON bundle保存schema/特征/单位/采样、训练来源、seed、参考矩阵及阈值。加载时检查尺度/阈值为正、有限、协方差/精度对称正定且互为逆矩阵，固定参数不符则拒绝。输入、模型、契约和输出CSV的SHA256写入运行metadata。哈希用于完整性核对，不证明数据真实或模型适用。

Python调用：

```python
import json
from pathlib import Path
from src.scoring_component import load_bundle, read_observations, score_frame
bundle = load_bundle("examples/scoring/model.json")
frame = read_observations("examples/scoring/observations.csv", bundle["features"])
contract = json.loads(Path("examples/scoring/contract.json").read_text())
scores = score_frame(frame, bundle, contract)
```

## 故障处理
目录已存在：换新输出目录。时间跳步：核实采样并显式补缺测行。单位/特征不匹配：更正源数据与声明，不得只改声明绕过校验。未知工况：暂时输出不可用，不自动套用别的工况。真实数据暂不支持直接套用示例模型。
