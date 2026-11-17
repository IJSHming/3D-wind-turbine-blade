# 3D Wind Turbine Blade Aerodynamic Optimization Benchmark (NREL Phase VI)

本项目面向经典水平轴风洞实验风力机 **NREL Phase VI (Sequence H)**（双叶片上风向风轮 $B = 2$，风轮半径 $R = 5.029\,\text{m}$ / 直径 $D = 10.058\,\text{m}$，物理轮毂法兰 $r_{\text{flange}} = 0.5083\,\text{m}$，圆柱根段 $0.5083 \sim 0.8835\,\text{m}$，S809 气动升力主体段 $r \in [1.2575\,\text{m}\,(0.25R), 5.0290\,\text{m}\,(1.00R)]$，净气动展长 $L = 3.7715\,\text{m}$，轮毂中心高 $H = 12.192\,\text{m}$，基准翼型 **NREL S809**，工作介质为标准干燥**空气** $\rho = 1.225\,\text{kg/m}^3, \mu = 1.789 \times 10^{-5}\,\text{Pa}\cdot\text{s}$）的 3D 叶片高维气动外形优化基准测试。

对照 Hand et al. (2001, `NREL/TP-500-29955`) 与 Kaya (2019, `sustainability-11-04502`) 的 NREL Phase VI 优化实验设置，本项目核心工程与科研对齐要点如下：
1. **NREL Phase VI (Sequence H) 官方高保真基准**：基于官方 66 点 Selig 格式 NREL S809 坐标与 TP-500-29955 展向弦长/扭角分布构建 `data/baseline_nrel_phase6/`，30% 弦长处为俯仰/扭转中心轴（$x/c = 0.30$），变桨角锁定 $\theta_{p,75\%} = 4.815^\circ$（等效叶尖 $3.0^\circ$ 桨距角）；
2. **27 截面矢量 BEM + NeuralFoil (`xxxlarge`) 气动评估器**：全面耦合 Snel 3D 旋转失速延迟修正、动态端点绑定 Viterna-Corrigan 深失速外推与 Shen/QBlade 根尖损失修正，逐截面按真实局部雷诺数（$Re \approx 6 \times 10^5 \sim 1.0 \times 10^6$）批量计算，单步评估实测仅需 **$\approx 0.10\,\text{s}$**；
3. **114 维 B-Spline Delta-Modal 参数化与严格可行域隔离**：6 个独立控制站位的 17 维闭口尾缘 CST（102 维，$t_{\text{TE}} \equiv 0.0$）+ 6 阶三次 B 样条弦长增量 $\Delta c_{\text{ctrl}}$（6 维）+ 6 阶三次 B 样条扭角增量 $\Delta \beta_{\text{ctrl}}$（6 维），共 $D = 114$ 维。各站位独立表征"根部高抗弯厚度—中段大弯度主扭矩—外段低型阻微洗退" 3D 展向规律；任何违反几何单调/平滑度或推力安全上限（$T \le 1.15\,T_{\text{baseline}} = 1.3917\,\text{kN}$）的设计严格返回负值；
4. **增强版 MALIBO（严格对齐 Pan et al. 2024 + D=114 连续域适配）**：
   - **Multi-Pitch + Multi-Start 离线元数据集**：围绕测试工况构建 20 档风速（$4.50 \sim 9.50\,\text{m/s} \setminus \{7.0\}$）$\times$ 5 档转速（$\pm 8\%$）$\times$ 6 档变桨角（$\setminus \{4.815^\circ\}$）共 100 任务 $\times 200$ 步（20,000 样本），结合 60% Multi-Start 多样化可行初始叶片与前置致命几何筛除，深度覆盖推力临界带与收敛流形；
   - **ResFFN + LayerNorm + Classifier Residual Booster**：在特征末端引入 `LayerNorm` 防止外推 Logit 爆炸，对齐 Pan et al. (2024, Eq. 10) 采用基于早停 `GradientBoostingClassifier(init=MALIBOInitialEstimator)` 的连续 `decision_function` 残差提升边距、Sobol 准蒙特卡洛热启动候选池及适配 114 维的信赖域尺度（`tr_len_init = 0.80, prob_perturb = 0.18, failure_tol = 6`）；
5. **停滞自适应真实大模型驱动 LLM-Meta-TuRBO**：直连本地 SGLang 服务推理 `Qwen/Qwen3.8-27B-FP8`，在 $n_{\text{init}} = 15$ 后第 16 步完成一次 `post_init` 初始审视，之后完全由**真实停滞计数器触发**（`stag_threshold = 10`），零静默启发式降级、零后验数据窥探。

---

## 1. 核心基准测试成果汇总 (10 独立随机种子, NFE = 300)

在 **NREL Phase VI (Sequence H) 基准工况**（来流风速 $V_\infty = 7.00\,\text{m/s}$，转速 $\Omega = 72.00\,\text{rpm}$，$\omega = 7.53982\,\text{rad/s}$，叶尖速比 $\lambda = 5.417$，变桨角 $\theta_{p,75\%} = 4.815^\circ$，27 个展向 BEM 截面）下：
- **Baseline (NREL Phase VI)**：气动功率 **$6.0712\,\text{kW}$**，轴转矩 **$805.22\,\text{N}\cdot\text{m}$**，风能利用系数 **$C_p = 0.3637$**，轴向推力 **$1.2102\,\text{kN}$**（$C_t = 0.5075$，结构推力安全上限 $1.3917\,\text{kN}$）。

| 优化算法 (Method) | 最终平均功率 (kW) | 转矩 (N·m) | 风能系数 $C_p$ | 相对基准提升 (%) | 最佳单种子功率 (kW) | 轴向推力 (kN) | 可行率 | 平均耗时 (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Phase VI)** | $6.0712$ | $805.22$ | $0.3637$ | $+0.00\%$ | $6.0712$ | $1.2102$ | Yes | -- |
| **TuRBO-1** | $6.5449 \pm 0.0638$ | $868.05 \pm 8.46$ | $0.3921 \pm 0.0038$ | $+7.80\%$ | $6.6274$ (`8888`) | $1.3910 \pm 0.0007$ | $10/10$ | $\mathbf{79.40}$ |
| **LLM-Meta-TuRBO** | $6.6031 \pm 0.0358$ | $875.77 \pm 4.75$ | $0.3956 \pm 0.0021$ | $+8.76\%$ | $6.6552$ (`9999`) | $1.3908 \pm 0.0013$ | $10/10$ | $141.01$ |
| **MALIBO** | $\mathbf{6.6255 \pm 0.0216}$ | $\mathbf{878.73 \pm 2.87}$ | $\mathbf{0.3969 \pm 0.0013}$ | $\mathbf{+9.13\%}$ | $\mathbf{6.6628}$ (`777`) | $1.3914 \pm 0.0003$ | $10/10$ | $148.11$ |

- **MALIBO 取得全场最高均值与最低方差（$+9.13\%$，$\sigma = 0.0216\,\text{kW}$，峰值 Seed 777 达 $6.6628\,\text{kW}$ / $+9.74\%$）**：完成 `LayerNorm` 稳态化、早停分类残差树、Sobol 热启动候选池与 Multi-Pitch/Multi-Start 20,000 样本训练升级后，MALIBO **在全程任意评估节点（`NFE = 30, 50, 75, 100, 150, 200, 300`）均稳居 TuRBO-1 之上**，彻底消除了此前 100 步附近的暂时落差，标准差缩减为 TuRBO-1 的约 $1/3$；
- **LLM-Meta-TuRBO 表现稳健且无定时阶跃伪影（$+8.76\%$，峰值 Seed 9999 达 $6.6552\,\text{kW}$ / $+9.62\%$）**：移除固定中期里程碑步后，大模型仅在真实停滞时介入（每种子仅触发 $3 \sim 8$ 次，`Seed 9999` 仅 3 次干预即夺得 $+9.62\%$），有效引导中展翼型弯度进化并显著降低方差（$\sigma = 0.0358\,\text{kW}$ vs TuRBO `0.0638 kW`）；
- **100% 物理可行与零惩罚**：三大方法全部 $30/30$ 个运行均严格零违约（`max_penalty = 0.000000`），推力稳定贴合结构安全上限 $1.3917\,\text{kN}$，与 Kaya (2019) 报导的 Phase VI ~10% 优化幅值完全契合。

---

## 2. 物理建模与 114 维参数化空间

### 2.1 B-Spline Delta-Modal Model ($D=114$)
- **6 控制站位 CST 翼型参数（102 维）**：位于 $r = [1.2575, 1.9828, 2.7081, 3.5784, 4.3037, 5.0290]\,\text{m}$（27 截面索引 `[0, 5, 10, 16, 21, 26]`），每站位 8 上翼面权重 + 8 下翼面权重 + 1 前缘权重，闭口后缘 $t_{\text{TE}} \equiv 0.0$；基准 S809 CST 拟合精度 `CL MAE = 8.40e-5, CD MAE = 1.00e-5`（相对厚度 $21.07\%$）；
- **6 阶三次 B 样条弦长增量 $\Delta c_{\text{ctrl}}$（6 维）**：叠加至 TP-500-29955 基准弦长 $[0.7370 \to 0.3560]\,\text{m}$；
- **6 阶三次 B 样条扭角增量 $\Delta \beta_{\text{ctrl}}$（6 维）**：叠加至 TP-500-29955 基准扭角 $[20.040^\circ \to -1.815^\circ]$（充分兼容 Kaya 最优根/尖扭角偏移域）。

### 2.2 工程可制造性与物理边界约束
1. **单调弦长收缩与扭角洗退**：$\Delta c / \Delta r \le 0$（最小弦长 $\ge 0.15\,\text{m}$）且 $\Delta \beta / \Delta r \le 0$，防止逆向扭转凸起与波浪鼓包；
2. **中展平滑与线性回归偏离校验**：控制中段线性偏差（$\le 0.045\,\text{m}$）与二阶曲率震荡，保持工业 CNC/模具可制造性；
3. **翼型不自交与厚度包络**：站位最大相对厚度按展向区间约束于 $[12\%, 28\%]$（根部）至 $[7\%, 24\%]$（叶尖），保证主梁刚度与前缘圆弧半径；
4. **整轮推力上限**：$T_{\text{rotor}} \le 1.15\,T_{\text{baseline}} = 1.3917\,\text{kN}$。

---

## 3. 出版级学术图表与水密 CFD 模型

全套图表保存于 `figures/`（300 DPI PNG、PDF 及 Plotly 3D HTML），CFD 几何保存于 `cfd_models/`：
1. `figures/blade_convergence_curves.png/pdf`：300 NFE 全程收敛曲线与 SEM 误差带；
2. `figures/blade_convergence_curves_100nfe.png/pdf`：前 100 NFE 快速启动收敛聚焦对比图；
3. `figures/blade_bar_comparison.png/pdf`：10 个独立随机种子逐一对照柱状图；
4. `figures/blade_boxplot_comparison.png/pdf`：10 种子统计分位箱线图与散点分布；
5. `figures/blade_chord_twist_distributions.png/pdf`：27 截面弦长 $c(r/R)$ 与扭角 $\beta(r/R)$ 展向分布对比；
6. `figures/blade_station_airfoils.png/pdf`：6 控制站位翼型轮廓演进对比（动态核验初始 S809 厚度 $21.1\%$）；
7. `figures/blade_spanwise_angles.png/pdf`：27 截面几何扭角 $\beta$、有效攻角 $\alpha$ 与总节距角分布；
8. `figures/blade_3d_rendering*.html`：Baseline vs MALIBO / TuRBO-1 / LLM-Meta-TuRBO / All-4-Methods 3D 交互渲染；
9. `cfd_models/blade_*.stl`：100% 水密（闭合 2-Manifold）3D STL 实体表面网格，统一以 30% 弦长为 $X$ 基准轴（详见 [cfd_models/README.md](cfd_models/README.md)）。

---

## 4. 代码仓库结构

```
3D-wind-turbine-blade/
├── src/
│   ├── blade_geometry.py         # NREL Phase VI 几何参数与三维放样 (R=5.029m, r_root=1.2575m, 0.30c 轴线)
│   ├── parameterizations.py      # B-Spline Delta-Modal 模型 (D=114) 与 Stations6 对比模型 (D=120)
│   ├── evaluator.py              # BEM + NeuralFoil(xxxlarge) + 动态 Viterna + Snel Stall Delay 评估器
│   ├── llm_agent.py              # 真实 SGLang 大模型元控制智能体 (区分 post_init 与 stagnation 提示词)
│   ├── optimizers/
│   │   ├── base.py               # 优化器基类 (严格可行域正分准入控门)
│   │   ├── turbo.py              # 高维信赖域 TuRBO-1 优化器
│   │   └── llm_turbo.py          # 停滞自适应 LLM-Meta-TuRBO (post_init + stag_threshold=10)
│   └── malibo/
│       ├── config.py             # MALIBO 结构与 114 维信赖域超参 (tr_len_init=0.80, prob_perturb=0.18)
│       ├── models.py             # ResFFN (含 LayerNorm 特征归一化) + MeanHead + Task Embeddings
│       ├── dataset.py            # 截断异常值鲁棒 LFBO 数据加载器
│       ├── dataset_generator.py  # 100 任务 Multi-Pitch/Multi-Start 200 NFE 物理轨迹采集器
│       ├── trainer.py            # KL 散度+去相关先验匹配 & weight_decay 解耦元训练器
│       ├── adaptation.py         # 未归一化一致性 Laplace MAP 在线任务自适应
│       ├── residual_booster.py   # 早停 GradientBoostingClassifier(init=MetaModel) 对数几率残差校准
│       └── optimizer.py          # Sobol 准蒙特卡洛热启动与 MALIBO 贝叶斯元优化器
├── data/
│   └── baseline_nrel_phase6/     # S809 坐标、TP-500-29955 几何表、build_baseline.py 与 27 截面 .npz
├── weights/malibo_checkpoint.pt  # 训练完成的增强版 MALIBO 元模型权重 (Val Loss=0.3658, z std=0.8417)
├── benchmark.py                  # 10 种子全量基准评测入口 (7.0 m/s, 72 rpm, 4.815 deg, 300 NFE)
├── benchmark_parameterizations.py# 参数化表示与光滑度对比脚本
├── train_malibo.py               # MALIBO 元训练命令行脚本
├── plot_results.py               # 全套出版级图表与 CFD 水密 STL 导出脚本
├── results/
│   ├── benchmark_results.json    # 10 种子全量对比实验原始数值数据
│   ├── benchmark_results.npz     # 压缩 NumPy 收敛曲线档案
│   └── research_report.md        # 详尽科研实验报告
├── figures/                      # PNG/PDF 图表与交互式 3D HTML
├── cfd_models/                   # 4 套水密 CFD STL 表面网格与 meshing 说明
└── tests/                        # 14 项单元测试与物理合规检验
```

---

## 5. 快速复现指南

```bash
# 1. 重建 NREL Phase VI 27 截面 S809 基准数据 (已内置)
python3 data/baseline_nrel_phase6/build_baseline.py

# 2. 生成 100 任务 Multi-Pitch/Multi-Start MALIBO 离线数据集并训练元模型
python3 -m src.malibo.dataset_generator
python3 train_malibo.py --epochs 1000

# 3. 执行 MALIBO 10 种子测试 (建议 GPU 独占)
python3 benchmark.py --force-rerun --skip-turbo --skip-llm-meta --workers 5

# 4. 启动本地 SGLang 服务 (端口 30000) 并执行 TuRBO-1 与 LLM-Meta-TuRBO 10 种子测试
bash /home/msds/c250089/notebook/repos/RL-Airfoil/launch_sglang.sh
python3 benchmark.py --skip-malibo --workers 5

# 5. 导出全部出版级图表与水密 CFD STL 并运行单元测试
python3 plot_results.py
python3 -m unittest discover -s tests -p "test_*.py" -v
```
