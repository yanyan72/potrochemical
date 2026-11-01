# 联合评分分支：论文公式与冻结实现对齐

2026-11-01。这是已运行算法的准确描述，不是算法修改。核对源：src/rq3_shared_mode_fault.py、src/rq3_shared_mode_fault_development.py、src/rq3_selective_confirmation.py。

旧硬工况评分可沿用其按记录工况校准的阈值。正式联合分支使用工况混合CDF残差和每通道统一阈值，不应直接写成旧硬工况残差除以q_{κ_t,j}。

令v为训练参考标准化的观测，z为三个工况，f∈{0,1,2,3}为无故障或单故障通道。冻结过滤为

\[
\alpha_t(z,f)\propto p(v_t\mid z,f)p(R_t\mid z)\sum_{z',f'}A_{z'z}B_{f'f}\alpha_{t-1}(z',f'),\quad q_t(z)=\sum_f\alpha_t(z,f).
\]

发射分布为均值μ_z、协方差Σ_z的高斯；f=j时使用Σ_z+κ²Σ_z,jj e_j e_j^T，κ=4；无故障不膨胀。记录似然匹配为0.9，其余各0.05。A及初始工况概率由reference_fit记录估计，B和故障初始概率沿用冻结candidate配置。

\[
u_{tj}=\sum_z q_t(z)\Phi((v_{tj}-\mu_{zj})/\sigma_{zj}),\quad a_{tj}=\Phi^{-1}(\operatorname{clip}(u_{tj},10^{-12},1-10^{-12})),
\]
\[
g_{tj}=0.2a_{tj}+0.8g_{t-1,j},\quad r_{tj}=|g_{tj}|/\tau_j,\quad S_{tj}=\exp(-r_{tj}^2/2).
\]

可见记录工况改变时，先将旧g置零再更新。τ_j是reference_cal在记录延迟0和12步两版本的绝对平滑残差合并后逐通道99%分位数，不读取质量标签或test。SA令κ=0，采用解析等价的普通全通道工况过滤并独立校准其τ_j；共同预测输入q仍取完整候选。

物理单位参考c_tj=Σ_z q_t(z)(μ_zj scale_j+center_j)。选择性λ_tj=[1−S_tj/exp(−1/2)]_+；x̃=x+λ(c−x)。在共同训练标准化坐标中计算等价，不改变该凸组合含义。

论文3.5保留原评分；3.6明确联合分支如何产生残差和阈值。SG/SA差异只作用评分/修正分支，不能写为删除整个联合工况信息。本文未重新核查外部文献，不以这套公式宣称基础HMM或门控思想首创。
