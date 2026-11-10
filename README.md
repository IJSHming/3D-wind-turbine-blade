# 3D Wind Turbine Blade Aerodynamic Optimization Benchmark

本项目面向工业级小型/中型风力发电机（**1.5m Reference Turbine Blade**，风轮半径 $R = 1.50\,\text{m}$，轮毂半径 $R_{\text{hub}} = 0.25\,\text{m}$，叶片净长 $L = 1.25\,\text{m}$，轮毂高度 $H = 10.0\,\text{m}$，叶片数 $B = 3$）的单个 3D 叶片气动外形优化设计。

为了进一步放大各优化算法的能力差异并验证元优化方法在复杂高维工程场景下的泛化能力，本项目严格遵照导师指导要求：
1. **采用未优化版本作为 Baseline**：将基准叶片设定为未优化的原始构型（`baseline/1p5mBlade/`，27 个展向截面），其根部弦长过大（$1.11\,\text{m}$）且翼型截面型阻较高，故意采用此“弱基准”以显著拉大优化算法之间的性能差距；
2. **严格采用 27 个截面进行高保真气动评估**：全面对齐导师附件的 27 个展向站位（$r \in [0.25\,\text{m}, 1.50\,\text{m}]$），结合高保真 BEM、NeuralFoil `xxxlarge`、Snel 3D 旋转失速延迟修正、Viterna 深失速外推以及 QBlade 动态根尖损失模型；
3. **扩展 MALIBO 训练集至 100 个工况任务**：涵盖 20 个密集环绕测试风速的点（$7.0 \sim 13.0\,\text{m/s}$）与 5 个转速等级（标称转速的 $0, \pm 3\%, \pm 6\%$），充分强化元学习无似然贝叶斯优化的跨工况迁移能力；
4. **统一高维参数化**：锁定为 $C^2$ 连续的 **B-Spline Delta-Modal Model ($D=132$)**（108 维 CST + 12 维 B 样条弦长增量 + 12 维 B 样条扭角增量），在 $x_0 = \mathbf{0}$ 处比特级完全重构基准叶片。

---

## 1. 核心基准测试成果汇总 (10 独立随机种子, NFE = 300)

在**物理校准额定运行工况**（来流风速 $V_\infty = 10.00\,\text{m/s}$，最佳转速 $\Omega = 300.00\,\text{rpm}$，叶尖速比 $\lambda = 4.71$，叶尖线速度 $47.12\,\text{m/s}$，叶片安装/变桨角 $\theta_{\text{pitch}} = 0.0^\circ$，预锥角 $0.0^\circ$，倾角 $0.0^\circ$，全展向 **27 个气动微元截面**）下：
- **未优化基准构型 (Baseline 1.5m Blade)**：气动功率为 **$1,660.65\,\text{W} \approx 1.661\,\text{kW}$** ($C_p = 0.3836$, 推力 $466.80\,\text{N} = 0.467\,\text{kN}$, $C_t = 1.078$)；
- **优化参考上限 (1p5mBlade-Opt)**：导师提供的已优化版本在此工况下可达 **$1,943.1\,\text{W}$ ($C_p = 0.4488$)**，表明流场中存在超过 $+17\%$ 的真实气动优化提升空间！

| 优化算法 (Method) | 最终平均功率 (kW) | 等效气动功率 (W) | 风能利用系数 $C_p$ | 相对基准增益 (%) | 最佳单种子功率 (kW) | 风轮推力 (kN) | 种子平均耗时 (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (1.5m Blade)** | $1.661$ | $1,660.7$ | $0.3836$ | $+0.00\%$ | $1.661$ | $0.467$ | -- |
| **TuRBO-1** | $2.043 \pm 0.180$ | $2,043.1 \pm 180.2$ | $0.4718 \pm 0.0415$ | $+23.00\% \pm 10.85\%$ | $\mathbf{2.357}$ | $0.511 \pm 0.022$ | $\mathbf{33.77 \pm 4.19}$ |
| **LLM-Meta-TuRBO** | $\mathbf{2.110 \pm 0.138}$ | $\mathbf{2,110.2 \pm 137.9}$ | $\mathbf{0.4873 \pm 0.0319}$ | $\mathbf{+27.05\% \pm 8.30\%}$ | $2.272$ | $0.525 \pm 0.021$ | $99.65 \pm 12.59$ |
| **MALIBO** | $2.007 \pm 0.161$ | $2,007.4 \pm 161.2$ | $0.4636 \pm 0.0373$ | $+20.86\% \pm 9.71\%$ | $2.204$ | $0.516 \pm 0.022$ | $66.89 \pm 8.59$ |

- **弱基准成功拉大性能区分度**：三大优化算法均大幅超越未优化基准（平均增益 $+20.8\% \sim +27.1\%$），且性能层级清晰显现；
- **LLM-Meta-TuRBO 拔得头筹**：凭借风机领域大模型元控制对根部过大弦长的快速收缩决策与全展向扭角宏观校正，取得了全场最高的平均功率（**$2.110\,\text{kW}$，增益 $+27.05\%$**）与最低的跨种子标准差（$\pm 0.138\,\text{kW}$）；
- **TuRBO-1 展现出强劲的单点突破能力**：在 Seed 256 下探索出高达 **$2.357\,\text{kW}$ ($C_p = 0.5445$, $+41.95\%$)** 的极致设计；
- **MALIBO 在前期快速收敛**：得益于 100 任务离线元训练，MALIBO 在前 50 步与前 100 步的平均表现均为全场最高（100 步时达 $1.942\,\text{kW}$），为预算受限场景提供了极佳的加速效果；
- **推力安全性验证**：所有优化叶片风轮推力均在 $0.51 \sim 0.53\,\text{kN}$ 之间，严格处于安全上限之内（$1.15 \times T_{\text{baseline}} = 0.537\,\text{kN}$）；
- **极速求解**：向量化极线预计算使得单次评估耗时仅需 **$\sim 145\,\text{ms}$**，单种子 300 步优化仅需 33~100 秒。

---

## 2. 物理问题建模与工业级平滑度工程约束

### 2.1 132 维 B 样条模态参数化与 Delta 增量架构
风力机叶片全长 $1.25\,\text{m}$，调控维度锁定为 **B-Spline Delta-Modal Model ($D=132$)**：
- 6 个控制站位的 18 维 CST 翼型参数：共 $6 \times 18 = 108$ 维；
- 12 阶三次 B 样条弦长增量控制向量 $\Delta c_{\text{ctrl}} \in \mathbb{R}^{12}$；
- 12 阶三次 B 样条扭角增量控制向量 $\Delta \beta_{\text{ctrl}} \in \mathbb{R}^{12}$；
- 全叶片设计维度：$D = 108 + 12 + 12 = 132$ 维。

在初始点 $x_0$ 处，$\Delta c_{\text{ctrl}} = \mathbf{0}, \Delta \beta_{\text{ctrl}} = \mathbf{0}$，保证 **100.000% 比特级无误差重构基准叶片外形**。

### 2.2 工业级防波浪感 (Anti-Waviness) 与中段直线收缩工程约束
为了彻底杜绝展向弦长与扭角曲线的“波浪感”（二次导数正负震荡）：
1. **中段直线回归包络约束 ($r \in [0.40\,\text{m}, 1.30\,\text{m}]$)**：
   在中段主功率与主承力区（截面 4 至 22），强制弦长偏离线性拟合直线的偏差 $\le 0.095\,\text{m}$。
2. **外展弦长单调衰减约束**：
   从根部最大弦长截面向翼尖方向，差分 $\Delta c / \Delta r \le 0$，彻底杜绝中段局部鼓包。
3. **展向扭角单调洗退约束**：
   从根部向翼尖强制扭角单调递减洗退（$\Delta \beta / \Delta r \le 0$），杜绝反向迎角突起。
4. **二阶曲率平滑正则项**：
   对翼型区弦长与扭角的二阶差分方差进行曲率惩罚，保证 $C^2$ 光滑过渡。

### 2.3 组装放样基准轴：25% 弦长气动中心线
所有横截面严格以 **25% 弦长点 ($x/c = 0.25, y/c = 0.0$) 为组装基准中心** 进行展向串联：
```python
x_local = (x_sec - 0.25) * c_i
y_local = y_sec * c_i
```
这严格符合风力机叶片工程设计的标准化规范：以 25% 弦长点为节距轴（Pitch Axis）与气动俯仰力矩中心，确保变桨旋转与扭角变形时不产生额外的非物理面外偏移。

### 2.4 对齐导师数据的 27 气动截面离散
气动评估器将展向参数平滑投影至 **27 个标准微元截面**，其位置严格对应导师附件 `modified_data.txt` 与 `Sheet1.txt` $\sim$ `Sheet27.txt` 的展向节点分布：
- 截面范围：$r_0 = 0.25\,\text{m}$（根部 Node 1）至 $r_{26} = 1.50\,\text{m}$（叶尖 Node 27），覆盖整个 $1.25\,\text{m}$ 叶片长度；
- 能够精准解析根部流动、过渡段涡脱落、中展高效升力面以及尖部三维泄漏涡等精细流场细节。

---

## 3. 出版级学术图表

全套出版级高清图表保存在 `figures/` 目录下（同时提供 300 DPI PNG、矢量 PDF 及交互式 HTML）：
1. `figures/blade_convergence_curves.png/pdf`：300 NFE 完整收敛曲线与标准误差带；
2. `figures/blade_convergence_curves_100nfe.png/pdf`：100 NFE 前期高分辨率聚焦收敛曲线；
3. `figures/blade_bar_comparison.png/pdf`：10 种子独立对比分组柱状图（顶端留白充足，图例置于左上角不遮挡数据）；
4. `figures/blade_boxplot_comparison.png/pdf`：10 种子箱线图分布（去除 baseline 横线，纵向舒展完整展现箱体四分位与离群点）；
5. `figures/blade_chord_twist_distributions.png/pdf`：展向 27 截面弦长分布 $c(r)$ 与扭角分布 $\beta(r)$ 对比（中段近直线，无波浪震荡）；
6. `figures/blade_station_airfoils.png/pdf`：$2 \times 3$ 布局的 6 核心截面翼型演进多子图对比（采用 250 点半余弦致密采样消除多边形视觉棱角）；
7. `figures/blade_3d_rendering.html`：**3D 交互式叶片空间渲染网页（完整呈现 $r=0.25\,\text{m}$ 到 $r=1.50\,\text{m}$ 的全尺度叶片，支持自由 360° 旋转、缩放、平移及 Baseline/Optimized 构型一键切换）**；
8. `figures/blade_spanwise_angles.png/pdf`：**27 个截面沿展向的气动迎角 $\alpha(r)$、截面安装扭角 $\beta(r)$、叶片变桨角 $\theta_{\text{pitch}}$ 与几何有效节距角分布对比**。

---

## 4. 代码仓库结构

```
3D-wind-turbine-blade/
├── src/
│   ├── blade_geometry.py       # 132 维 1.5m 叶片几何放样 (27 截面无截断，25% 弦线组装)
│   ├── parameterizations.py    # B-Spline Delta-Modal 模型 (D=132, 中段直线回归约束, 单调洗退)
│   ├── evaluator.py            # 工业级 BEM + NeuralFoil(xxxlarge) + 3D Stall Delay + QBlade Shen Loss 评估器
│   ├── llm_agent.py            # LLM-Meta-Blade 领域先验智能体 (适配 1.5m 10m/s 300rpm)
│   ├── optimizers/
│   │   ├── base.py             # 优化器通用抽象基类
│   │   ├── turbo.py            # 132 维 TuRBO-1 优化器
│   │   └── llm_turbo.py        # LLM-Meta-TuRBO 优化器
│   └── malibo/
│       ├── config.py           # MALIBO 结构与优化超参 (D=132, d=64)
│       ├── models.py           # ResFFN + MeanHead + Task Embeddings 模型
│       ├── dataset.py          # 多任务 LFBO 数据集加载器
│       ├── dataset_generator.py# 100 任务物理轨迹生成脚本 (20 风速点 x 5 转速等级)
│       ├── trainer.py          # 多任务元训练器
│       ├── adaptation.py       # 拉普拉斯在线任务自适应与 Thompson 采样
│       ├── residual_booster.py # 残差梯度提升纠偏模块
│       └── optimizer.py        # MALIBO 贝叶斯元优化器驱动
├── benchmark.py                # 10 随机种子全量基准测试驱动 (对齐 1.5m 10.0m/s, 300.0rpm, 27截面)
├── plot_results.py             # 出版级学术对比图表生成器 (包含 3D HTML 网页生成与图例优化)
├── train_malibo.py             # MALIBO 神经网络离线元训练入口
├── data/
│   ├── baseline_1p5m/          # 1.5m 导师参考外形数据 (baseline_27_sections.npz, Sheet1~27)
│   └── baseline_10mw/          # 10MW 参考外形数据存档
├── weights/
│   └── malibo_checkpoint.pt    # MALIBO 预训练神经网络权重 (D=132)
├── results/
│   ├── benchmark_results.json  # 基准测试数值数据 (10 种子 300 步全量数据)
│   └── research_report.md      # 详尽学术研究报告
├── figures/                    # 导出的学术图表 (PNG, PDF & HTML)
└── references/                 # 导师附件与参考文件
```

---

## 5. 快速复现指南

```bash
# 1. 启动本地 SGLang 大模型服务 (如未启动)
bash /home/msds/c250089/notebook/repos/RL-Airfoil/launch_sglang.sh

# 2. 生成 MALIBO 100 任务离线数据集并训练元模型
python3 src/malibo/dataset_generator.py
python3 train_malibo.py

# 3. 运行 10 随机种子全量对比基准测试 (NFE=300, 严格对齐 1.5m 额定工况与 27 截面, NeuralFoil xxxlarge)
python3 benchmark.py --wind-speed 10.0 --rpm 300.0 --pitch-angle-deg 0.0 --n-bem-stations 27 --max-evals 300

# 4. 生成全套出版级高清学术图表与 3D 交互式 HTML 网页
python3 plot_results.py
```
