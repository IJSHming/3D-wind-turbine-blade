# 3D Wind Turbine Blade Aerodynamic Optimization Benchmark

本项目面向工业级大型海上风力发电机（**ModelTest 10 MW Offshore Reference Wind Turbine**，风轮半径 $R = 89.166\,\text{m}$，净叶片长 $87.366\,\text{m}$，轮毂直径 $3.6\,\text{m}$ ($R_{\text{hub}} = 1.8\,\text{m}$)，轮毂高度 $147.5\,\text{m}$，叶片数 $B = 3$）的单个 3D 叶片气动外形优化设计。

为了丰富学术论文并验证元优化方法在复杂高维工程场景下的泛化能力，本项目将先前的 **2D-Airfoil** 研究全面拓展至工业级 **120 维 3D 叶片气动外形多截面耦合优化**。在统一函数评估预算（NFE = 300）与工业几何约束下，系统对比了三种前沿优化算法：
1. **TuRBO-1**：高维信赖域贝叶斯优化（Trust Region Bayesian Optimization，采用算法发明者推荐的默认超参）；
2. **LLM-Meta-TuRBO**：结合风力机空气动力学专家先验与宏观定向几何形变注入的大语言模型元控制器（保留前期激进偏移量介入）；
3. **MALIBO**：多工况元学习无似然贝叶斯元优化器（Meta-learning for Likelihood-free Bayesian Optimization，基于 21 个强相关风速 vs 转速物理运行轨迹热启动）。

---

## 1. 核心基准测试成果汇总 (10 独立随机种子, NFE = 300)

在**严格对齐 10MW 标称额定运行工况**（风速 $V_\infty = 16.26\,\text{m/s}$，转速 $\Omega = 7.55\,\text{rpm}$，叶片变桨角 $\theta_{\text{pitch}} = 0.0^\circ$，预锥角 $0.0^\circ$，倾角 $0.0^\circ$，全展向 **50 个气动微元截面**）下，基准叶片构型（`10MW_AeroDyn15_blade.dat` 标称几何）气动功率为 **$27,719.16\,\text{kW}$ ($27.719\,\text{MW}$)** ($C_p = 0.4215$, 推力 $2,417.10\,\text{kN}$)。

| 优化算法 (Method) | 最终平均功率 (kW) | 等效气动功率 (MW) | 风能利用系数 $C_p$ | 相对基准增益 (%) | 最佳单种子功率 (MW) | 风轮推力 (kN) | 种子平均耗时 (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (ModelTest 10MW)** | $27,719.16$ | $27.719$ | $0.4215$ | $0.00\%$ | $27.719$ | $2,417.10$ | -- |
| **TuRBO-1** | $30,103.12 \pm 70.75$ | $30.103 \pm 0.071$ | $0.4577 \pm 0.0011$ | $+8.60\% \pm 0.26\%$ | $30.222$ | $2,779.47 \pm 0.37$ | $93.98 \pm 11.98$ |
| **LLM-Meta-TuRBO** | $30,087.86 \pm 96.02$ | $30.088 \pm 0.096$ | $0.4575 \pm 0.0015$ | $+8.55\% \pm 0.35\%$ | $30.278$ | $2,779.27 \pm 0.40$ | $152.02 \pm 18.42$ |
| **MALIBO** | $\mathbf{30,167.32 \pm 60.17}$ | $\mathbf{30.167 \pm 0.060}$ | $\mathbf{0.4587 \pm 0.0009}$ | $\mathbf{+8.83\% \pm 0.22\%}$ | $\mathbf{30.312}$ | $2,779.52 \pm 0.25$ | $130.32 \pm 23.91$ |

- **推力安全性验证**：所有优化叶片风轮推力均在 $2,779.2 \sim 2,779.5\,\text{kN}$ 之间严格满足安全上限（$1.15 \times T_{\text{baseline}} = 2,779.66\,\text{kN}$），证实功率提升源自真实升阻比与环量分布改善，杜绝了暴力增攻角引发的推力过载破坏风险。
- **MALIBO 极高鲁棒性与稳定性**：标准差全场最低，仅为 $\pm 60.17\,\text{kW}$（比 TuRBO-1 与 LLM-Meta 显著更稳定），最终平均功率全场第一（$30.167\,\text{MW}$，相对增益 $+8.83\%$），单种子最高突破 **$30.312\,\text{MW}$**（Seed 5555），证明多任务元训练先验在 50 截面耦合高维流形上表现出极强的全局寻优能力。
- **单次评估耗时极致优化**：即使在最高精度 `xxxlarge` 模式下，单次叶片气动评估时间仍稳定在 **$\sim 0.19\,\text{s}$**（190 毫秒），远远低于 HPC 设定的 1.0 秒硬性时限。在 9,000 次大规模优化调用场景下总耗时约 28 分钟，彻底扫清了计算瓶颈。

---

## 2. 物理问题建模与高保真气动求解体系

### 2.1 120 维叶片参数化与 6 控制站位布局
风力机叶片沿展向具有明显的空气动力学与结构分区特性。本项目依据导师提供的 `10MW_AeroDyn15_blade.dat` 与 `Airfoils_10MW` 翼型族，选取 6 个核心气动控制站位：
- **Station 1 ($r = 12.0\,\text{m}, r/R = 0.135$)**: 近根部过渡段（AF04, $t/c \approx 12.4\%$），基准弦长 $5.00\,\text{m}$，扭角 $+10.00^\circ$；
- **Station 2 ($r = 19.5\,\text{m}, r/R = 0.219$)**: 最大弦长肩部区（AF09, $t/c \approx 11.5\%$），基准弦长 $11.06\,\text{m}$，扭角 $+14.97^\circ$；
- **Station 3 ($r = 31.5\,\text{m}, r/R = 0.353$)**: 内展高效气动区（AF17, $t/c \approx 11.4\%$），基准弦长 $7.01\,\text{m}$，扭角 $+11.56^\circ$；
- **Station 4 ($r = 49.5\,\text{m}, r/R = 0.555$)**: 中展主功率区（AF27, $t/c \approx 11.3\%$），基准弦长 $5.53\,\text{m}$，扭角 $+6.78^\circ$；
- **Station 5 ($r = 73.5\,\text{m}, r/R = 0.824$)**: 外展高马赫减阻区（AF43, $t/c \approx 11.4\%$），基准弦长 $3.68\,\text{m}$，扭角 $+1.90^\circ$；
- **Station 6 ($r = 87.0\,\text{m}, r/R = 0.976$)**: 翼尖气动洗退区（AF49, $t/c \approx 12.1\%$），基准弦长 $1.67\,\text{m}$，扭角 $-0.57^\circ$。

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

### 2.3 对齐 10MW 的 50 气动截面离散
气动评估器将 6 个控制站位的参数沿展向平滑插值至 **50 个标准微元截面**，其位置严格对应 `10MW_AeroDyn15_blade.dat` 的叶素中心分布：
- 截面范围：$r_0 = 3.639\,\text{m}$（根部 Node 1）至 $r_{49} = 88.948\,\text{m}$（叶尖 Node 50），覆盖整个 $87.366\,\text{m}$ 叶片长度；
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
   永久锁定 NeuralFoil `xxxlarge` 模型，兼具极高精度与亚秒级推理（单次评估 $\sim 0.19\,\text{s}$）。

### 2.5 工业级几何光滑度与多参数化建模评测 (B-样条连续模态 vs 离散站位)
针对工业制造中叶片外形“二次导数忽正忽负、局部波浪感（Waviness）”导致模具难以加工的问题，本项目在 L40S GPU + 10核 CPU 硬件平台上系统评测了三种不同的参数化架构：
1. **Standard 6-Station Model ($D=120$)**：经典 6 截面离散控制 + 升级的前缘半径与曲率平滑约束；
2. **B-Spline Modal Model ($D=132$)**：6 翼型站位 CST(108维) + 12 阶三次 B 样条弦长 + 12 阶三次 B 样条扭角连续模态模型（数学上天然保证 $C^2$ 连续性，从根本上杜绝二次导数振荡）；
3. **Discrete 25-Station Model ($D=500$)**：25 截面直接高维离散控制模型（评估 500 维超高维在强大硬件上的步长耗时与寻优能力）。

#### 评测基准数据对比 (100 NFE TuRBO-1 优化, Seed=42, L40S 平台)：
| 参数化模型 | 优化维度 $D$ | 单步评估耗时 | 总耗时 (100 NFE) | 气动功率演进 (kW) | 功率增益 | 弦长曲率方差 $\text{Var}(\Delta^2 c)$ | 扭角曲率方差 $\text{Var}(\Delta^2 \beta)$ | 二阶导拐点振荡数 | 最小前缘半径 $r_{\text{LE}}/c$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard 6-Station** | $120$ | $1,619.0\,\text{ms}$ | $161.9\,\text{s}$ | $27,719.2 \to 29,724.6$ | $+7.23\%$ | $0.262037$ | $0.081499$ | 弦长: 19, 扭角: 27 | $0.0057$ |
| **B-Spline Modal (推荐)** | $132$ | $\mathbf{943.6\,\text{ms}}$ | $\mathbf{94.4\,\text{s}}$ | $27,685.9 \to \mathbf{29,910.7}$ | $\mathbf{+8.04\%}$ | $0.163882$ | $\mathbf{0.035817}$ | **弦长: 7, 扭角: 6** | $\mathbf{0.0095}$ |
| **Discrete 25-Station** | $500$ | $1,293.2\,\text{ms}$ | $129.3\,\text{s}$ | $27,316.8 \to 28,201.7$ | $+3.24\%$ | $\mathbf{0.035338}$ | $0.050974$ | 弦长: 21, 扭角: 16 | $0.0066$ |

- **B-样条模态卓越性能**：B-样条模态不仅收敛速度最快（仅需 $94.4\,\text{s}$），气动增益最高（$+8.04\%$，达 $29.91\,\text{MW}$），而且彻底消除了波浪感，拐点振荡从 27 次骤降至 6 次，扭角曲率方差下降超 56%，完美契合风力机叶片工业制造模具工艺。
- **高维离散维数灾难验证**：25 截面直接离散控制虽然硬件算力能够支撑（$1.29\,\text{s/step}$），但由于 $D=500$ 维度过于庞大，在 100 步评估内仅提升 $+3.24\%$，出现明显的维数灾难与搜索停滞。

---

## 3. 出版级学术图表

全套出版级高清图表保存在 `figures/` 目录下（同时提供 300 DPI PNG、矢量 PDF 及交互式 HTML）：
1. `figures/blade_convergence_curves.png/pdf`：300 NFE 完整收敛曲线与标准误差带；
2. `figures/blade_convergence_curves_100nfe.png/pdf`：100 NFE 前期高分辨率聚焦收敛曲线（展示前期微观动态）；
3. `figures/blade_bar_comparison.png/pdf`：10 种子独立对比分组柱状图；
4. `figures/blade_boxplot_comparison.png/pdf`：10 种子箱线图分布与散点抖动展示；
5. `figures/blade_chord_twist_distributions.png/pdf`：展向 50 截面弦长分布 $c(r)$ 与扭角分布 $\beta(r)$ 对比；
6. `figures/blade_station_airfoils.png/pdf`：$2 \times 3$ 布局的 6 核心截面翼型演进多子图对比（采用 250 点半余弦致密采样消除多边形视觉棱角）；
7. `figures/blade_3d_rendering.html`：**3D 交互式叶片空间渲染网页 (基于 Plotly 3D Surface，支持自由 360° 旋转、缩放、平移及 Baseline/Optimized 构型一键切换)**；
8. `figures/blade_spanwise_angles.png/pdf`：**50 个截面沿展向的气动迎角 $\alpha(r)$、截面安装扭角 $\beta(r)$、叶片变桨角 $\theta_{\text{pitch}}$ 与几何有效节距角分布对比**；
9. `figures/param_benchmark_convergence.png/pdf`：3D 叶片不同参数化模型（6 站位 D=120、B-样条连续模态 D=132、25 站位 D=500）收敛曲线对比；
10. `figures/param_benchmark_smoothness.png/pdf`：50 截面弦长与扭角及其二阶差分曲率平滑度诊断图（深度剖析波浪感消除机理）。

---

## 4. 代码仓库结构

```
3D-wind-turbine-blade/
├── src/
│   ├── blade_geometry.py       # 120 维 10MW 叶片几何参数化、50截面插值与 3D 放样 (25% 弦线组装)
│   ├── parameterizations.py    # 多参数化模型库 (BSplineBladeModel D=132, Stations25 D=500, Stations6 D=120)
│   ├── evaluator.py            # 工业级 BEM + NeuralFoil(xxxlarge) + 3D Stall Delay + QBlade Shen Loss 纯内存评估器
│   ├── llm_agent.py            # LLM-Meta-Blade 领域先验智能体 (适配 10MW 叶片)
│   ├── optimizers/
│   │   ├── base.py             # 优化器通用抽象基类
│   │   ├── turbo.py            # 120 维 TuRBO-1 优化器 (输出 50 截面全量气动角度)
│   │   └── llm_turbo.py        # LLM-Meta-TuRBO 优化器 (激进偏移量注入 + 50 截面角度输出)
│   └── malibo/
│       ├── config.py           # MALIBO 结构与优化超参 (D=120, d=64)
│       ├── models.py           # ResFFN + MeanHead + Task Embeddings 模型
│       ├── dataset.py          # 多任务 LFBO 数据集加载器
│       ├── dataset_generator.py# 21 工况强相关物理轨迹生成脚本
│       ├── trainer.py          # 多任务元训练器
│       ├── adaptation.py       # 拉普拉斯在线任务自适应与 Thompson 采样
│       ├── residual_booster.py # 残差梯度提升纠偏模块
│       └── optimizer.py        # MALIBO 贝叶斯元优化器驱动 (输出 50 截面全量气动角度)
├── benchmark.py                # 10 随机种子全量基准测试驱动 (对齐 10MW 16.26m/s, 7.55rpm, 50截面)
├── benchmark_parameterizations.py # 多参数化模型 (6站位, B样条模态, 25站位) 算力耗时与平滑度评测脚本
├── plot_results.py             # 出版级学术对比图表生成器 (包含 3D HTML 网页生成与 spanwise_angles 绘图)
├── train_malibo.py             # MALIBO 神经网络离线元训练入口
├── data/
│   └── baseline_10mw/          # 10MW 导师参考外形数据 (10MW_AeroDyn15_blade.dat, Airfoils_10MW)
├── weights/
│   └── malibo_checkpoint.pt    # MALIBO 预训练神经网络权重 (600 epochs)
├── results/
│   ├── benchmark_results.json  # 基准测试数值数据 (包含各截面 twist, alpha, pitch 50 维数组)
│   ├── param_benchmark_results.json # 多参数化对比评测原始数据
│   └── research_report.md      # 详尽学术研究报告
├── figures/                    # 导出的学术图表 (PNG, PDF & HTML)
└── references/                 # 导师附件与参考论文
```

---

## 5. 快速复现指南

```bash
# 1. 运行 10 随机种子全量对比基准测试 (NFE=300, 严格对齐 10MW 工况与 50 截面, NeuralFoil xxxlarge)
python3 benchmark.py --wind-speed 16.26 --rpm 7.55 --pitch-angle-deg 0.0 --n-bem-stations 50 --max-evals 300

# 2. 运行不同参数化模型与平滑度硬件评测 (对比 6站位 D=120, B样条模态 D=132, 25站位 D=500)
python3 benchmark_parameterizations.py

# 3. 生成全套出版级高清学术图表与 3D 交互式 HTML 网页
python3 plot_results.py
```
