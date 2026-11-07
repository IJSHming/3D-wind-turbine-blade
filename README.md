# 3D Wind Turbine Blade Aerodynamic Optimization Benchmark

本项目面向工业级大型海上风力发电机（**IEA-22-280-RWT 22-MW 超大型参考机组**，风轮半径 $R = 142.0\,\text{m}$，净叶片长 $137.8\,\text{m}$，叶片数 $B = 3$，轮毂高度 $170\,\text{m}$）的单个 3D 叶片气动外形优化设计。

为了丰富学术论文并验证元优化方法在复杂高维工程场景下的泛化能力，本项目将先前的 **2D-Airfoil** 研究全面拓展至工业级 **120 维 3D 叶片气动外形多截面耦合优化**。在统一函数评估预算（NFE = 300）与工业几何约束下，系统对比了三种前沿优化算法：
1. **TuRBO-1**：高维信赖域贝叶斯优化（Trust Region Bayesian Optimization，采用算法发明者推荐的默认超参）；
2. **LLM-Meta-TuRBO**：结合风力机空气动力学专家先验与宏观定向几何形变注入的大语言模型元控制器（保留前期激进偏移量介入）；
3. **MALIBO**：多工况元学习无似然贝叶斯元优化器（Meta-learning for Likelihood-free Bayesian Optimization，基于风速 vs 转速强相关物理运行轨迹热启动）。

---

## 1. 核心基准测试成果汇总 (10 独立随机种子, NFE = 300)

在**严格对齐 OpenFAST AeroDyn15 基准测试工况**（风速 $V_\infty = 11.0\,\text{m/s}$，转速 $\Omega = 6.77094\,\text{rpm}$，叶片变桨角 $\theta_{\text{pitch}} = 2.42859^\circ$，全展向 **58 个气动微元截面**，倾角 $4^\circ$，预锥角 $-3.5^\circ$，风剪切 $\alpha = 0.14$）下，基准叶片构型（IEA-22-280-RWT 标称几何）气动功率为 **$22,797.15\,\text{kW}$ ($22.797\,\text{MW}$)** ($C_p = 0.4431$, 推力 $3,210.76\,\text{kN}$)。

| 优化算法 (Method) | 最终平均功率 (kW) | 等效气动功率 (MW) | 风能利用系数 $C_p$ | 相对基准增益 (%) | 最佳单种子功率 (MW) | 风轮推力 (kN) | 种子平均耗时 (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (IEA 22MW)** | $22,797.15$ | $22.797$ | $0.4431$ | $0.00\%$ | $22.797$ | $3,210.76$ | -- |
| **TuRBO-1** | $23,964.03 \pm 103.23$ | $23.964 \pm 0.103$ | $0.4658 \pm 0.0020$ | $+5.12\% \pm 0.45\%$ | $24.119$ | $3,561.98 \pm 36.93$ | $106.02 \pm 12.18$ |
| **LLM-Meta-TuRBO** | $24,006.41 \pm 129.46$ | $24.006 \pm 0.129$ | $0.4666 \pm 0.0025$ | $+5.30\% \pm 0.57\%$ | $24.194$ | $3,561.54 \pm 19.16$ | $157.14 \pm 15.28$ |
| **MALIBO** | $\mathbf{24,092.00 \pm 86.02}$ | $\mathbf{24.092 \pm 0.086}$ | $\mathbf{0.4683 \pm 0.0017}$ | $\mathbf{+5.68\% \pm 0.38\%}$ | $\mathbf{24.244}$ | $3,589.74 \pm 26.10$ | $147.91 \pm 15.12$ |

- **推力安全性验证**：所有优化叶片风轮推力均处于 $3,561 \sim 3,590\,\text{kN}$ 之间，严格低于基准推力安全上限（$1.15 \times T_{\text{baseline}} = 3,692.37\,\text{kN}$），证实功率提升源自真实升阻比与环量分布改善，杜绝了暴力增攻角引发的推力过载破坏风险。
- **MALIBO 极高鲁棒性与稳定性**：标准差全场最低，仅为 $\pm 86.02\,\text{kW}$（比 TuRBO-1 与 LLM-Meta 显著更稳定），最终平均功率全场第一（$24.092\,\text{MW}$，相对增益 $+5.68\%$），单种子最高突破 **$24.244\,\text{MW}$**（Seed 2024），证明多任务元训练先验在 58 截面耦合高维流形上表现出极强的全局寻优能力。
- **单次评估耗时极致优化**：即使在最高精度 `xxxlarge` 模式下，单次叶片气动评估时间仍稳定在 **$\sim 0.20\,\text{s}$**（200 毫秒），远远低于 HPC 设定的 1.0 秒硬性时限。在 9,000 次大规模优化调用场景下总耗时约 30 分钟，彻底扫清了计算瓶颈。

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
3. **Viterna 深度失速大攻角外推 (Viterna Post-Stall Extrapolation)**：
   保证在变工况或极端流动分离下，气动极线在 $[-180^\circ, +180^\circ]$ 全迎角范围内连续可微。
4. **NeuralFoil 'xxxlarge' 最高精度神经网络气动求解**：
   永久锁定 NeuralFoil `xxxlarge` 模型，兼具极高精度与亚秒级推理（单次评估 $\sim 0.20\,\text{s}$）。

### 2.5 工业级几何光滑度与多参数化建模评测 (B-样条连续模态 vs 离散站位)
针对工业制造中叶片外形“二次导数忽正忽负、局部波浪感（Waviness）”导致模具难以加工的问题，本项目在 L40S GPU + 10核 CPU 硬件平台上系统评测了三种不同的参数化架构：
1. **Standard 6-Station Model ($D=120$)**：升级前缘半径下限与二阶差分平滑约束的原版 6 截面离散模型；
2. **B-Spline Modal Model ($D=132$)**：6 翼型站位 CST(108维) + 12 阶三次 B 样条弦长 + 12 阶三次 B 样条扭角连续模态模型（数学上天然保证 $C^2$ 连续性，从根本上杜绝二次导数振荡）；
3. **Discrete 29-Station Model ($D=580$)**：29 截面直接高维离散控制模型（评估 580 维超高维在强大硬件上的步长耗时与寻优能力）。

#### 评测基准数据对比 (100 NFE TuRBO-1 优化, Seed=42, L40S 平台)：
| 参数化模型 | 优化维度 $D$ | 单步评估耗时 | 总耗时 (100 NFE) | 气动功率演进 (kW) | 功率增益 | 弦长曲率方差 $\text{Var}(\Delta^2 c)$ | 扭角曲率方差 $\text{Var}(\Delta^2 \beta)$ | 二阶导拐点振荡数 | 最小前缘半径 $r_{\text{LE}}/c$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard 6-Station** | $120$ | $529.4\,\text{ms}$ | $52.9\,\text{s}$ | $22,832.0 \to 23,662.8$ | $+3.64\%$ | $0.000753$ | $0.014139$ | 弦长: 23, 扭角: 24 | $0.0210$ (尖角杜绝) |
| **B-Spline Modal (推荐)** | $132$ | $\mathbf{294.7\,\text{ms}}$ | $\mathbf{29.5\,\text{s}}$ | $22,837.3 \to 23,527.7$ | $+3.02\%$ | $0.001194$ | $\mathbf{0.007880}$ | **弦长: 5, 扭角: 7** | $\mathbf{0.0248}$ (圆润前缘) |
| **Discrete 29-Station** | $580$ | $363.3\,\text{ms}$ | $36.3\,\text{s}$ | $22,832.7 \to 22,911.5$ | $+0.35\%$ | $0.002619$ | $0.011103$ | 弦长: 24, 扭角: 19 | $0.0208$ |

- **算力耗时结论**：在 L40S GPU 上，单步耗时均处于 $0.29 \sim 0.53\,\text{s}$ 之间，均严格满足 $< 1.0\,\text{s}$ 预算上限。
- **高维离散维数灾难**：29 截面直接离散控制虽然硬件算力能够支撑，但由于 $D=580$ 维度过于庞大，在 100 步评估内仅提升 $+0.35\%$，陷入明显的维数灾难；且离散控制由于截面间缺乏模态约束，二阶导拐点变号高达 24 次，仍存在严重波浪感。
- **B-样条模态优势**：B 样条模态参数化将拐点变号从 24 次骤降至 5~7 次，消除了 70% 以上的高频曲率振荡，同时保持了高达 $+3.02\%$ 的气动增益，单步耗时仅 $294.7\,\text{ms}$，是工业级高平滑度制造的最优构型。
- **前缘尖角抑制**：在移植 RL-Airfoil 前缘半径约束 $r_{\text{LE}} \ge r_{\text{min}}$ 后，各站位前缘半径均保持在 $0.021 \sim 0.138$ 范围内，彻底杜绝了算法利用 NeuralFoil 2D 估计漏洞制造非物理“薄刀刃前缘”的缺陷。

---

## 3. 出版级学术图表

全套出版级高清图表保存在 `figures/` 目录下（同时提供 300 DPI PNG、矢量 PDF 及交互式 HTML）：
1. `figures/blade_convergence_curves.png/pdf`：300 NFE 完整收敛曲线与标准误差带；
2. `figures/blade_convergence_curves_100nfe.png/pdf`：100 NFE 前期高分辨率聚焦收敛曲线（展示前期微观动态）；
3. `figures/blade_bar_comparison.png/pdf`：10 种子独立对比分组柱状图；
4. `figures/blade_boxplot_comparison.png/pdf`：10 种子箱线图分布与散点抖动展示；
5. `figures/blade_chord_twist_distributions.png/pdf`：展向弦长分布 $c(r/R)$ 与扭角分布 $\beta(r/R)$ 对比；
6. `figures/blade_station_airfoils.png/pdf`：$2 \times 3$ 布局的 6 核心截面翼型演进多子图对比（采用 250 点半余弦致密采样消除多边形视觉棱角）；
7. `figures/blade_3d_rendering.html`：**3D 交互式叶片空间渲染网页 (基于 Plotly 3D Surface，支持自由 360° 旋转、缩放、平移及 Baseline/Optimized 构型一键切换)**；
8. `figures/blade_spanwise_angles.png/pdf`：**58 个截面沿展向的气动迎角 $\alpha(r)$、截面安装扭角 $\beta(r)$、叶片变桨角 $\theta_{\text{pitch}}$ 与几何有效节距角分布对比**；
9. `figures/param_benchmark_convergence.png/pdf`：3D 叶片不同参数化模型（6 站位 D=120、B-样条连续模态 D=132、29 站位 D=580）收敛曲线对比；
10. `figures/param_benchmark_smoothness.png/pdf`：58 截面弦长与扭角及其二阶差分曲率平滑度诊断图（深度剖析波浪感消除机理）。

---

## 4. 代码仓库结构

```
3D-wind-turbine-blade/
├── src/
│   ├── blade_geometry.py       # 120 维 IEA 22MW 叶片几何参数化、OpenFAST 58截面插值与 3D 放样 (25% 弦线组装)
│   ├── parameterizations.py    # 多参数化模型库 (BSplineBladeModel D=132, Stations29 D=580, Stations6 D=120)
│   ├── evaluator.py            # 工业级 BEM + NeuralFoil(xxxlarge) + 3D Stall Delay + QBlade Shen Loss 纯内存评估器
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
├── benchmark_parameterizations.py # 多参数化模型 (6站位, B样条模态, 29站位) 算力耗时与平滑度评测脚本
├── plot_results.py             # 出版级学术对比图表生成器 (包含 3D HTML 网页生成与 spanwise_angles 绘图)
├── train_malibo.py             # MALIBO 神经网络离线元训练入口
├── results/
│   ├── benchmark_results.json  # 基准测试数值数据 (包含各截面 twist, alpha, pitch 58 维数组)
│   ├── param_benchmark_results.json # 多参数化对比评测原始数据
│   └── research_report.md      # 详尽学术研究报告
├── figures/                    # 导出的学术图表 (PNG, PDF & HTML)
└── references/                 # 导师附件与参考论文 (受 .gitignore 保护)
```

---

## 5. 快速复现指南

```bash
# 1. 运行 10 随机种子全量对比基准测试 (NFE=300, 严格对齐 OpenFAST 工况与 58 截面, NeuralFoil xxxlarge)
python3 benchmark.py --wind-speed 11.0 --rpm 6.77094 --pitch 2.42859 --max-evals 300

# 2. 运行不同参数化模型与平滑度硬件评测 (对比 6站位 D=120, B样条模态 D=132, 29站位 D=580)
python3 benchmark_parameterizations.py

# 3. 生成全套出版级高清学术图表与 3D 交互式 HTML 网页
python3 plot_results.py
```
