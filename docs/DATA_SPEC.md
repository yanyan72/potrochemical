# DATA_SPEC

## 数据来源与任务

当前全部为synthetic。用户确认真实数据后续提供，尚无产品、批次、仪表或化验来源可宣称。新储运数据与旧M0数据是不同版本，不能直接比较两者指标差值为方法增益。

## 储运E1c新增结构

`configs/logistics_e1c.yaml`使用180步序列，理想训练参考仍为固定状态。每seed保存base_data.npz（clean/sequence_ids/splits/regimes/features/units，含未评分test）及scales.json（仅train_fit工况内标准差）。每场景validation_data.npz保存clean/observed/mask、逐时刻true_regimes和recorded_regimes、切换窗口transition和sequence_ids；scores.npz保存两种视图下各方法的逐通道分数。

事件为分段偏置，可包含1/2个通道，记录起止、通道、符号、标准化幅度和尺度来源。掩码与实际变化逐单元一致；不沿用E1b单通道corruption_type语义。状态延迟仅改变记录标签；瞬时转场映射和跨转场尺度定义见METHOD_LOGISTICS_E1C。当前不宣称真实转场动力学、工况估计或共模故障已验证。

## 储运E1b结构

一条序列为独立的固定状态储运片段；时间为抽象采样步。状态：warehouse、transit_smooth、transit_rough。每个split均覆盖三个状态。示意通道为temperature(K)、relative_humidity(percent_RH)、vibration(mm_per_s)；实际监测通道需在数据到位后核对。

`data.npz`字段：
- clean、observed：[sequence,time,channel]；原始示意真值与人工污染观测。
- mask：同shape布尔污染真值，仅评价使用。
- corruption_type：[sequence,time]，none/spike/bias/drift/missing/random_replacement。
- sequence_ids、splits、regimes：[sequence]；features、units：[channel]。
- 不含quality：尚未定义可对应真实产品的标签。

train_fit每状态8条，train_cal每状态4条，val每状态4条，test每状态4条；默认每条160步，共60条。fit/cal是训练集内部互斥部分。模拟/污染使用split独立随机流，修改test规模不改变训练和validation。

污染率按序列-时间行定义；每个污染行只有一个通道。幅度缩放和随机替换候选只来自train_fit。当前覆盖5类污染；弱污染、重叠/多通道污染和转场需另设版本。

## 评分产物

validation_scores.npz包含每方法的deviation_ratio、reliability、available、alarm。shape为[validation sequence-time rows,channel]，行顺序与data中val子集reshape一致。
缺失和分工况参考未覆盖的状态，分数为NaN、available=false、alarm=false。假告警false不表示被判定为可靠，必须结合available。
条件残差对含缺失的整行不可评分；边际残差保留其他可见通道。公平数值检测主表只使用共同完整行，同时报告全通道可评分比例和缺失计数。

events.json保存事件时间、通道、幅度、单位和split；references.json保存标准化、协方差/精度与独立校准阈值；reference_sequences.json保存fit/cal/val身份。
metadata.json保存代码版本、源码哈希、配置哈希、环境和所有产物哈希，路径均相对运行目录。

## 真实数据最小接口（待提供）

批次/储运任务ID、采样时间、传感器值和单位、状态记录及其可用时间、质量指标定义、取样时间、报告出具时间、质检值和缺失说明。若只有到货标签，任务要按到货质量定义。环境温湿度不直接等同于货物内部状态；必须记录传感器位置。真实过程上人工注入故障仍不是实际故障标签。

## 历史版本

旧M0具有temperature/pressure/vibration/concentration四通道及经验quality；单位和定义见归档计划与其metadata。既有数据、配置、运行ID不改写。
