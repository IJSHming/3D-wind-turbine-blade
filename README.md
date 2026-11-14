# 3D Wind Turbine Blade Aerodynamic Optimization Benchmark (NREL Phase VI)

本项目面向经典水平轴风洞实验风力机 **NREL Phase VI (Sequence H)**（双叶片上风向风轮 $B = 2$，风轮半径 $R = 5.029\,\text{m}$ / 直径 $D = 10.058\,\text{m}$，物理轮毂法兰 $r_{\text{flange}} = 0.5083\,\text{m}$，圆柱根段 $0.5083 \sim 0.8835\,\text{m}$，S809 气动升力主体段 $r \in [1.2575\,\text{m}\,(0.25R), 5.0290\,\text{m}\,(1.00R)]$，净气动展长 $L = 3.7715\,\text{m}$，轮毂中心高 $H = 12.192\,\text{m}$，基准翼型 **NREL S809**，工作介质为标准干燥**空气** $\rho = 1.225\,\text{kg/m}^3, \mu = 1.789 \times 10^{-5}\,\text{Pa}\cdot\text{s}$）的 3D 叶片高维气动外形优化基准测试。

对照 Hand et al. (2001, `NREL/TP-500-29955`) 与 Kaya (2019, `sustainability-11-04502`) 的 NREL Phase VI 优化实验设置，本项目核心工程与科研对齐要点如下：
1. **NREL Phase VI (Sequence H) 官方高保真基准**：基于官方 66 点 Selig 格式 NREL S809 坐标与 TP-500-29955 展向弦长/扭角分布构建 `data/baseline_nrel_phase6/`，30% 弦长处为俯仰/扭转中心轴（$x/c = 0.30$），变桨角锁定 $\theta_{p,75\%} = 4.815^\circ$（等效叶尖 $3.0^\circ$ 桨距角）；
2. **27 截面矢量 BEM + NeuralFoil (`xxxlarge`) 气动评估器**：全面耦合 Snel 3D 旋转失速延迟修正、动态端点绑定 Viterna-Corrigan 深失速外推与 Shen/QBlade 根尖损失修正，逐截面按真实局部雷诺数（$Re \approx 6 \times 10^5 \sim 1.0 \times 10^6$）批量计算，单步评估实测仅需 **$\approx 0.10\,\text{s}$**；
3. **114 维 B-Spline Delta-Modal 参数化与严格可行域隔离**：6 个核心控制站位的 17 维闭口尾缘 CST（102 维，$t_{\text{TE}} \equiv 0.0$）+ 6 阶三次 B 样条弦长增量 $\Delta c_{\text{ctrl}}$（6 维）+ 6 阶三次 B 样条扭角增量 $\Delta \beta_{\text{ctrl}}$（6 维），共 $D = 114$ 维，任何违反几何单调/平滑度或推力安全上限（$T \le 1.15\,T_{\text{baseline}} = 1.3917\,\text{kN}$）的设计严格返回负值；
4. **MALIBO 100 工况元训练与规范校准**：围绕测试工况（严格避开 $V_\infty = 7.0\,\text{m/s}, 72.0\,\text{rpm}$）生成 20 档风速（$4.50 \sim 9.50\,\text{m/s}$）$\times$ 5 档正相关转速（$\pm 6\%$）共 $100$ 任务 / $12,000$ 样本轨迹，并以对角 KL + 协方差去相关先验与一致的未归一化 Laplace MAP 推断完成校准；
5. **真实本地大模型驱动 LLM-Meta-TuRBO**：直连本地 SGLang 服务推理 `Qwen/Qwen3.8-27B-FP8`，在 $\text{NFE} \ge 16$ 停滞时基于气动诊断实时输出非阻塞信任域调控与顺滑翼型/弦长/扭角方向引导，零静默启发式降级、零后验数据窥探。

---

## 1. 核心基准测试成果汇总 (10 独立随机种子, NFE = 300)

在 **NREL Phase VI (Sequence H) 基准工况**（来流风速 $V_\infty = 7.00\,\text{m/s}$，转速 $\Omega = 72.00\,\text{rpm}$，$\omega = 7.53982\,\text{rad/s}$，叶尖速比 $\lambda = 5.417$，变桨角 $\theta_{p,75\%} = 4.815^\circ$，27 个展向 BEM 截面）下：
- **Baseline (NREL Phase VI)**：气动功率 **$6.0712\,\text{kW}$**，轴转矩 **$805.22\,\text{N}\cdot\text{m}$**，风能利用系数 **$C_p = 0.3637$**，轴向推力 **$1.2102\,\text{kN}$**（$C_t = 0.5075$，结构推力安全上限 $1.3917\,\text{kN}$）。

| 优化算法 (Method) | 最终平均功率 (kW) | 转矩 (N·m) | 风能系数 $C_p$ | 相对基准提升 (%) | 最佳单种子功率 (kW) | 轴向推力 (kN) | 可行率 | 平均耗时 (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Phase VI)** | $6.0712$ | $805.22$ | $0.3637$ | $+0.00\%$ | $6.0712$ | $1.2102$ | Yes | -- |
| **TuRBO-1** | $6.545 \pm 0.064$ | $868.1 \pm 8.5$ | $0.3921 \pm 0.0038$ | $+7.80\%$ | $6.6274$ | $1.3910 \pm 0.0007$ | $10/10$ | $\mathbf{79.44}$ |
| **MALIBO** | $6.577 \pm 0.024$ | $872.3 \pm 3.2$ | $0.3940 \pm 0.0014$ | $+8.32\%$ | $6.6143$ | $1.3908 \pm 0.0009$ | $10/10$ | $110.29$ |
| **LLM-Meta-TuRBO** | $\mathbf{6.613 \pm 0.034}$ | $\mathbf{877.1 \pm 4.6}$ | $\mathbf{0.3962 \pm 0.0021}$ | $\mathbf{+8.92\%}$ | $\mathbf{6.6660}$ | $1.3911 \pm 0.0009$ | $10/10$ | $158.01$ |

- **LLM-Meta-TuRBO 综合表现领先（平均 $+8.92\%$，峰值 Seed 256 达 $6.6660\,\text{kW}$ / $+9.80\%$）**：在推力严格锁死在安全上限 $1.3917\,\text{kN}$ 以内且保持 100% 几何可行与零惩罚（`max_penalty = 0.000000`）的前提下，大模型准确识别中展段（$r = 2.71 \sim 3.58\,\text{m}$）增弯度减阻策略，以更温和的攻角分布实现更高功率输出，较 TuRBO-1 均值高出 $+0.068\,\text{kW}$ 且标准差由 $0.064\,\text{kW}$ 大幅收窄至 $0.034\,\text{kW}$；
- **MALIBO 跨种子稳定性最优（标准差仅 $0.024\,\text{kW}$，提升 $+8.32\%$）**：在 100 个严密排除测试风速的关联空气动力学任务预训练后，MALIBO 在 $\text{NFE}=30$ 前段即迅速超越标准 TuRBO-1（$6.3752\,\text{kW}$ vs $6.2039\,\text{kW}$），最终全种子稳定收敛至 $6.5294 \sim 6.6143\,\text{kW}$ 高位平台，方差较 TuRBO-1 降低超过 $62\%$；
- **与 Kaya (2019) 文献规律一致**：Kaya 在 NREL Phase VI 上仅优化扭角取得约 $10\%$ 转矩提升，本框架在引入 15% 推力上限与严格单调制造约束下的 3D 全翼型+平面+扭角协同优化取得 $8.92\% \sim 9.80\%$ 的真实无违约功率与转矩提升，物理自洽。

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
│   ├── llm_agent.py              # 真实 SGLang 大模型元控制智能体 (NREL Phase VI 物理提示词, 零泄露)
│   ├── optimizers/
│   │   ├── base.py               # 优化器基类 (严格可行域正分准入控门)
│   │   ├── turbo.py              # 高维信赖域 TuRBO-1 优化器
│   │   └── llm_turbo.py          # LLM-Meta-TuRBO (方向引导切片注入 + 临时步长缩放)
│   └── malibo/
│       ├── config.py             # MALIBO 结构与超参 (D=114, d=64, prob_perturb=0.20)
│       ├── models.py             # ResFFN + MeanHead + Task Embeddings (N(0, 0.5^2) init)
│       ├── dataset.py            # 截断异常值鲁棒 LFBO 数据加载器
│       ├── dataset_generator.py  # 100 工况多进程 TuRBO 轨迹采集器 (V in [4.5, 9.5], 避开 7.0 m/s)
│       ├── trainer.py            # KL 散度与协方差去相关先验匹配元训练器
│       ├── adaptation.py         # 未归一化一致性 Laplace MAP 在线任务自适应
│       ├── residual_booster.py   # 梯度提升残差校准器
│       └── optimizer.py          # MALIBO 贝叶斯元优化器
├── data/
│   └── baseline_nrel_phase6/     # S809 坐标、TP-500-29955 几何表、build_baseline.py 与 27 截面 .npz
├── weights/malibo_checkpoint.pt  # 训练完成的 NREL Phase VI MALIBO 元模型权重 (Val Loss=0.3082)
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

# 2. 生成 100 工况 MALIBO 离线数据集并训练元模型
python3 -m src.malibo.dataset_generator
python3 train_malibo.py --epochs 600

# 3. 启动本地 SGLang 大模型服务 (端口 30000) 并运行 10 种子全量测试
bash /home/msds/c250089/notebook/repos/RL-Airfoil/launch_sglang.sh
python3 benchmark.py --force-rerun --skip-malibo --workers 5
# (停止 SGLang 释放 L40S 显存后执行 MALIBO)
python3 benchmark.py --skip-turbo --skip-llm-meta --workers 5

# 4. 导出全部出版级图表与水密 CFD STL 并运行单元测试
python3 plot_results.py
python3 -m unittest discover -s tests -p "test_*.py" -v
```
