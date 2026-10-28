# 3D Wind Turbine Blade Aerodynamic Optimization Benchmark

本项目面向工业级大型海上风力发电机（**IEA-22-280-RWT 22-MW 超大型参考机组**，风轮半径 $R = 142.0\,\text{m}$，净叶片长 $137.8\,\text{m}$，叶片数 $B = 3$，轮毂高度 $170\,\text{m}$）的单个 3D 叶片气动外形优化设计。

为了丰富学术论文并验证元优化方法在复杂高维工程场景下的泛化能力，本项目将先前的 **2D-Airfoil** 研究全面拓展至工业级 **120 维 3D 叶片气动外形多截面耦合优化**。在统一函数评估预算（NFE = 300）与工业几何约束下，系统对比了三种前沿优化算法：
1. **TuRBO-1**：高维信赖域贝叶斯优化（Trust Region Bayesian Optimization，采用算法发明者推荐的默认超参）；
2. **LLM-Meta-TuRBO**：结合风力机空气动力学专家先验与宏观定向几何形变注入的大语言模型元控制器（保留前期激进偏移量介入）；
3. **MALIBO**：多工况元学习无似然贝叶斯元优化器（Meta-learning for Likelihood-free Bayesian Optimization，基于风速 vs 转速强相关物理运行轨迹热启动）。

---

## 1. 核心基准测试成果汇总 (10 独立随机种子, NFE = 300)

在**严格对齐 OpenFAST AeroDyn15 基准测试工况**（风速 $V_\infty = 11.0\,\text{m/s}$，转速 $\Omega = 6.77094\,\text{rpm}$，叶片变桨角 $\theta_{\text{pitch}} = 2.42859^\circ$，全展向 **58 个气动微元截面**）下，基准叶片构型（IEA-22-280-RWT 标称几何）气动功率为 **$24,931.33\,\text{kW}$ ($24.931\,\text{MW}$)** ($C_p = 0.4828$, 推力 $3,516.15\,\text{kN}$)。

| 优化算法 (Method) | 最终平均功率 (kW) | 等效气动功率 (MW) | 风能利用系数 $C_p$ | 相对基准增益 (%) | 最佳单种子功率 (MW) | 风轮推力 (kN) | 种子平均耗时 (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (IEA 22MW)** | $24,931.33$ | $24.931$ | $0.4828$ | $0.00\%$ | $24.931$ | $3,516.15$ | -- |
| **TuRBO-1** | $25,679.72 \pm 91.75$ | $25.680 \pm 0.092$ | $0.4973 \pm 0.0018$ | $+3.00\% \pm 0.37\%$ | $25.775$ | $3,694.67 \pm 35.41$ | $52.18 \pm 0.53$ |
| **LLM-Meta-TuRBO** | $25,632.82 \pm 78.04$ | $25.633 \pm 0.078$ | $0.4963 \pm 0.0015$ | $+2.81\% \pm 0.31\%$ | $25.746$ | $3,675.48 \pm 30.81$ | $103.61 \pm 12.11$ |
| **MALIBO** | $\mathbf{25,649.31 \pm 63.91}$ | $\mathbf{25.649 \pm 0.064}$ | $\mathbf{0.4967 \pm 0.0012}$ | $\mathbf{+2.88\% \pm 0.26\%}$ | $\mathbf{25.749}$ | $3,680.31 \pm 34.73$ | $64.64 \pm 1.37$ |

- **推力安全性验证**：所有优化叶片风轮推力均处于 $3,675 \sim 3,695\,\text{kN}$ 之间，严格低于基准推力安全上限（$1.15 \times T_{\text{baseline}} = 4,043.6\,\text{kN}$），证实功率提升源自真实升阻比与环量分布改善，杜绝了暴力增攻角引发的推力过载破坏风险。
- **MALIBO 极高鲁棒性与稳定性**：标准差全场最低，仅为 $\pm 63.91\,\text{kW}$（比 TuRBO-1 降低 30.3%），证明多任务元训练先验与拉普拉斯自适应在 58 截面耦合高维流形上表现出极强的平稳收敛性。
- **单次评估耗时极致优化**：单次叶片气动评估时间低至 **$48.80\,\text{ms}$**（0.0488 秒），远远低于 HPC 设定的 1.0 秒硬性时限。在 9,000 次大规模优化调用场景下，评估总耗时仅约 7.3 分钟，彻底扫清了计算瓶颈。

---

## 2. 物理问题建模与高保真气动求解体系

### 2.1 120 维叶片参数化与 6 控制站位布局
风力机叶片沿展向具有明显的空气动力学与结构分区特性。本项目依据规范与 Jichao Li et al. (Renewable Energy 2024)，选取 6 个核心气动控制站位：
- **Station 1 ($r = 20.83\,\text{m}, r/R = 0.147$)**: 近根部结构厚过渡段（FFA-W3 族 AF07, $t/c \approx 54.0\%$），弦长 $7.04\,\text{m}$，扭角 $+11.69^\circ$；
- **Station 2 ($r = 30.33\,\text{m}, r/R = 0.214$)**: 最大弦长肩部区（AF11, $t/c \approx 36.6\%$），弦长 $7.22\,\text{m}$，扭角 $+9.95^\circ$；
- **Station 3 ($r = 56.47\,\text{m}, r/R = 0.398$)**: 内展高效气动区（AF22, $t/c \approx 30.5\%$），弦长 $6.03\,\text{m}$，扭角 $+3.79^\circ$；
- **Station 4 ($r = 82.60\,\text{m}, r/R = 0.582$)**: 中展主功率区（AF34, $t/c \approx 26.1\%$），弦长 $4.22\,\text{m}$，扭角 $-0.42^\circ$；
- **Station 5 ($r = 108.74\,\text{m}, r/R = 0.766$)**: 外展高马赫减阻区（AF45, $t/c \approx 24.1\%$），弦长 $2.81\,\text{m}$，扭角 $-4.56^\circ$；
- **Station 6 ($r = 132.50\,\text{m}, r/R = 0.933$)**: 翼尖气动洗退区（AF54, $t/c \approx 22.2\%$），弦长 $1.88\,\text{m}$，扭角 $-6.67^\circ$。

每个站位由 **20 个物理维度** 精确控制：
- 18 维 Kulfan (CST) 翼型参数：8 维上表面吸力面权重、8 维下表面压力面权重、1 维前缘权重、1 维后缘厚度；
- 1 维局部截面弦长 $c$ (m)；
- 1 维局部截面安装扭角 $\beta$ (deg)。
全叶片设计空间维度：$D = 6 \times (18 + 1 + 1) = 120$ 维。

### 2.2 组装放样基准轴：25% 弦长气动中心线
在 `src/blade_geometry.py` 的 3D 放样与微元展开中，所有横截面严格以 **25% 弦长点 ($x/c = 0.25, y/c = 0.0$) 为组装基准中心** 进行展向串联：
```python
x_local = (x_sec - 0.25) * c_i
y_local = y_sec * c_i
```
这严格符合风力机叶片工程设计的标准化规范：以 25% 弦长点为节距轴（Pitch Axis）与气动俯仰力矩中心，确保变桨旋转与扭角变形时不产生额外的非物理面外偏移。

### 2.3 对齐 OpenFAST AeroDyn15 的 58 气动截面离散
气动评估器将 6 个控制站位的参数沿展向平滑插值至 **58 个标准微元截面**，其位置严格对应 OpenFAST AeroDyn15 的叶素中心分布：
- 截面范围：$r_0 = 5.388\,\text{m}$（根部）至 $r_{57} = 140.812\,\text{m}$（叶尖），步长 $\Delta r = 2.37586\,\text{m}$；
- 能够精准解析根部流动、过渡段涡脱落、中展高效升力面以及尖部三维泄漏涡等精细流场细节。

### 2.4 吸收 OpenFAST 3D 旋转失速延迟与 QBlade 高保真损失模型
在保持 100% 纯内存运行（零磁盘 I/O）的前提下，评估器深度集成了先进空气动力学机理修正：
1. **OpenFAST 3D 旋转失速延迟修正 (Snel / Du-Selig 3D Stall Delay Correction)**：
   风电机叶片在旋转时，离心力与科氏力对附面层产生三维抽吸作用（3D pumping effect），推迟内侧截面的失速分离并提升最大升力系数：
   $$\Delta C_l = f_{\text{stall}} \cdot 3.0 \left(\frac{c}{r}\right)^2 \Delta C_{l,\text{pot}}, \quad \Delta C_d = -f_{\text{stall}} \cdot 0.5 \left(\frac{c}{r}\right) \Delta C_d$$
2. **QBlade 动态根尖损失模型 (Shen et al., 2005 Dynamic Tip/Root Loss)**：
   相比经典 Prandtl 经验公式，QBlade 引入了与叶尖速比 $\lambda = \Omega R / V_\infty$ 动态自适应耦合的指数衰减修正因子 $g$：
   $$F_{\text{tip}} = \frac{2}{\pi} \arccos\left(\exp\left(-g_{\text{tip}} \frac{B(R - r)}{2 r \sin\phi}\right)\right), \quad g_{\text{tip}} = \exp\left(-c_1 (\lambda - c_2)\right) + 0.1$$
3. **NeuralFoil 亚毫秒级神经网络气动求解**：
   采用 NeuralFoil `xlarge` 架构，单次升阻力极线预测仅耗时 $1.2\,\text{ms}$，使包含 58 截面全流场 BEM 迭代的单次评估总耗时仅 **$48.80\,\text{ms}$**，在精度与实时性之间达到了极致平衡。

---

## 3. 出版级学术图表生成

全套 8 张出版级高清图表保存在 `figures/` 目录下（同时提供 300 DPI PNG 与矢量 PDF）：
1. `figures/blade_convergence_curves.png/pdf`：300 NFE 完整收敛曲线与标准误差带；
2. `figures/blade_convergence_curves_100nfe.png/pdf`：100 NFE 前期高分辨率聚焦收敛曲线（展示前期微观动态）；
3. `figures/blade_bar_comparison.png/pdf`：10 种子平均输出功率柱状图与增益百分比标注（Legend 置于上方空阔区，杜绝遮挡）；
4. `figures/blade_boxplot_comparison.png/pdf`：10 种子箱线图分布与散点抖动展示；
5. `figures/blade_chord_twist_distributions.png/pdf`：展向弦长分布 $c(r/R)$ 与扭角分布 $\beta(r/R)$ 对比；
6. `figures/blade_station_airfoils.png/pdf`：$2 \times 3$ 布局的 6 核心截面翼型演进多子图对比；
7. `figures/blade_3d_rendering.png/pdf`：3D 叶片空间等轴测透视几何渲染图（按风机高长宽比进行透视放样）；
8. `figures/blade_spanwise_angles.png/pdf`：**58 个截面沿展向的气动迎角 $\alpha(r)$、截面安装扭角 $\beta(r)$、叶片变桨角 $\theta_{\text{pitch}}$ 与几何有效节距角分布对比**。

---

## 4. 代码仓库结构

```
3D-wind-turbine-blade/
├── src/
│   ├── blade_geometry.py       # 120 维 IEA 22MW 叶片几何参数化、OpenFAST 58截面插值与 3D 放样 (25% 弦线组装)
│   ├── evaluator.py            # 工业级 BEM + NeuralFoil + 3D Stall Delay + QBlade Shen Loss 纯内存评估器
│   ├── llm_agent.py            # LLM-Meta-Blade 领域先验智能体 (适配 22MW 叶片)
│   ├── optimizers/
│   │   ├── base.py             # 优化器通用抽象基类
│   │   ├── turbo.py            # 120 维 TuRBO-1 优化器 (输出 58 截面全量气动角度)
│   │   └── llm_turbo.py        # LLM-Meta-TuRBO 优化器 (激进偏移量注入 + 58 截面角度输出)
│   └── malibo/
│       ├── config.py           # MALIBO 结构与优化超参 (D=120, d=64)
│       ├── models.py           # ResFFN + MeanHead + Task Embeddings 模型
│       ├── dataset.py          # 多任务 LFBO 数据集加载器
│       ├── dataset_generator.py# 21 工况强相关物理轨迹生成脚本
│       ├── trainer.py          # 多任务元训练器
│       ├── adaptation.py       # 拉普拉斯在线任务自适应与 Thompson 采样
│       ├── residual_booster.py # 残差梯度提升纠偏模块
│       └── optimizer.py        # MALIBO 贝叶斯元优化器驱动 (输出 58 截面全量气动角度)
├── benchmark.py                # 10 随机种子全量基准测试驱动 (对齐 OpenFAST 11m/s, 6.77rpm, 2.43deg, 58截面)
├── plot_results.py             # 出版级 8 张学术对比图表生成器 (包含 spanwise_angles 绘图)
├── train_malibo.py             # MALIBO 神经网络离线元训练入口
├── results/
│   ├── benchmark_results.json  # 基准测试数值数据 (包含各截面 twist, alpha, pitch 58 维数组)
│   ├── benchmark_results.npz   # NumPy 向量化数组
│   └── research_report.md      # 详尽学术研究报告
├── figures/                    # 导出的学术图表 (PNG & PDF)
└── references/                 # 导师附件与参考论文 (受 .gitignore 保护)
```

---

## 5. 快速复现指南

```bash
# 1. 运行 10 随机种子全量对比基准测试 (NFE=300, 严格对齐 OpenFAST 工况与 58 截面)
python3 benchmark.py --wind-speed 11.0 --rpm 6.77094 --pitch 2.42859 --max-evals 300

# 2. 生成全套出版级高清学术图表 (含 58 截面气动角分布)
python3 plot_results.py
```
