# 3D Wind Turbine Blade Aerodynamic Optimization Benchmark (NREL Phase VI)

本项目面向经典水平轴风洞实验风力机 **NREL Phase VI (Sequence H)**（双叶片上风向风轮 $B = 2$，风轮半径 $R = 5.029\,\text{m}$ / 直径 $D = 10.058\,\text{m}$，物理轮毂法兰 $r_{\text{flange}} = 0.5083\,\text{m}$，圆柱根段 $0.5083 \sim 0.8835\,\text{m}$，S809 气动升力主体段 $r \in [1.2575\,\text{m}\,(0.25R), 5.0290\,\text{m}\,(1.00R)]$，净气动展长 $L = 3.7715\,\text{m}$，轮毂中心高 $H = 12.192\,\text{m}$，基准翼型 **NREL S809**，工作介质为标准干燥**空气** $\rho = 1.225\,\text{kg/m}^3, \mu = 1.789 \times 10^{-5}\,\text{Pa}\cdot\text{s}$）的 3D 叶片气动外形优化基准测试。

对照 Hand et al. (2001, `NREL/TP-500-29955`) 与 Kaya (2019, `sustainability-11-04502`) 的 NREL Phase VI 优化实验设置，本项目核心工程与科研对齐要点如下：
1. **NREL Phase VI (Sequence H) 官方高保真基准**：基于官方 66 点 Selig 格式 NREL S809 坐标与 TP-500-29955 展向弦长/扭角分布构建 `data/baseline_nrel_phase6/baseline_31_sections.npz`，30% 弦长处为俯仰/扭转中心轴（$x/c = 0.30$），变桨角锁定 $\theta_{p,75\%} = 4.815^\circ$（等效叶尖 $3.0^\circ$ 桨距角）；
2. **31 截面全余弦加密矢量 BEM + NeuralFoil (`xxxlarge`) 气动评估器**：淘汰旧项目遗留的 27 均匀截面网格，升级为 **31 节点全余弦双端加密网格（Full-Cosine Clustering）**，在根部高扭角梯度区与叶尖 Prandtl/Shen 尖端损失陡降区自动加密离散，相较 201 节点高精度参考解将 BEM 积分功率误差压缩 **3 倍**（由 $0.411\%$ 降至 $0.139\%$），单步评估仍仅需 **$\approx 0.11\,\text{s}$**；
3. **63 维三站位工业级光滑参数化（$D=63$）**：将翼型控制站位精简至纯 S809 升力段头/中/尾 **3 个核心站位**（Root $1.2575\,\text{m}$、Mid $3.1433\,\text{m}$、Tip $5.0290\,\text{m}$，共 $3 \times 17 = 51$ 维闭口尾缘 CST，$t_{\text{TE}} \equiv 0.0$），沿展向通过**保单调三次 PCHIP 样条** $C^1$ 无过冲光滑放样，配合 6 阶三次 B 样条弦长增量（6 维）+ 6 阶三次 B 样条扭角增量（6 维），总维数由 $D=114$ 大幅降至 **$D=63$**，彻底消除 3D 叶片侧视表面波纹起伏；
4. **严格对齐原论文的纯净 MALIBO（Pan et al., ICML 2024）**：完整实现原论文 `ResFFN`（4 隐层 $\times$ 64 单元、ELU、无 LayerNorm、$d=50$）、`MeanHead`（50 单元）、精确 Eq. (2) LFBO 效用损失 + Eq. (4)/Appendix C 精确非参数排序 KS + 协方差高斯先验正则（元训练后 $z_t$ 精确收敛至 $\text{std}=1.0057, |\text{mean}|=0.0122$）、L-BFGS 拉普拉斯 MAP/Hessian 汤普森采样（固定 $\gamma=1/3$）、70/30 验证早停 LFBO 双标签残差梯度提升树（Eq. 10），以及原论文 Appendix G 规定的**纯全局 5,120 点随机采集搜索（彻底移除信赖域与稀疏坐标掩码）**，确保会议论文实验严格忠实于原论文基线，为后续期刊论文算法改进保留纯净参照基准；
5. **停滞自适应真实大模型驱动 LLM-Meta-TuRBO**：直连本地 SGLang 服务推理 `Qwen/Qwen3.8-27B-FP8`，在 $n_{\text{init}} = 15$ 后第 16 步完成一次 `post_init` 初始审视，之后完全由真实停滞计数器触发（`stag_threshold = 10`），零静默启发式降级、零后验数据窥探。

---

## 1. 核心基准测试成果汇总 (10 独立随机种子, NFE = 300)

在 **NREL Phase VI (Sequence H) 基准工况**（来流风速 $V_\infty = 7.00\,\text{m/s}$，转速 $\Omega = 72.00\,\text{rpm}$，$\omega = 7.53982\,\text{rad/s}$，叶尖速比 $\lambda = 5.417$，变桨角 $\theta_{p,75\%} = 4.815^\circ$，31 全余弦加密 BEM 截面，$D=63$）下：
- **Baseline (NREL Phase VI)**：气动功率 **$6.0884\,\text{kW}$**，轴转矩 **$807.50\,\text{N}\cdot\text{m}$**，风能利用系数 **$C_p = 0.3647$**，轴向推力 **$1.2134\,\text{kN}$**（$C_t = 0.5088$，结构推力安全上限 $1.3954\,\text{kN}$）。

| 优化算法 (Method) | 最终平均功率 (kW) | 转矩 (N·m) | 风能系数 $C_p$ | 相对基准提升 (%) | 最佳单种子功率 (kW) | 轴向推力 (kN) | 可行率 | 平均耗时 (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Phase VI)** | $6.0884$ | $807.50$ | $0.3647$ | $+0.00\%$ | $6.0884$ | $1.2134$ | Yes | -- |
| **TuRBO-1** | $6.5920 \pm 0.0463$ | $874.29 \pm 6.14$ | $0.3949 \pm 0.0028$ | $+8.27\%$ | $6.6526$ (`42`) | $1.3938 \pm 0.0018$ | $10/10$ | $\mathbf{70.87}$ |
| **LLM-Meta-TuRBO** | $\mathbf{6.6664 \pm 0.0196}$ | $\mathbf{884.16 \pm 2.60}$ | $\mathbf{0.3994 \pm 0.0012}$ | $\mathbf{+9.49\%}$ | $\mathbf{6.6955}$ (`1024`) | $1.3948 \pm 0.0003$ | $10/10$ | $157.86$ |
| **MALIBO (Paper-Exact, Global)** | $6.1773 \pm 0.0280$ | $819.29 \pm 3.72$ | $0.3701 \pm 0.0017$ | $+1.46\%$ | $6.2276$ (`2024`) | $1.2417 \pm 0.0116$ | $10/10$ | $111.00$ |

- **维数精简（$D=114 \to 63$）全面释放优化潜力并实现工业级 3D 表面光顺**：`LLM-Meta-TuRBO` 均值飙升至 **$6.6664 \pm 0.0196\,\text{kW}$（$+9.49\%$）**，峰值 Seed 1024 达 **$6.6955\,\text{kW}$（$+9.97\%$，$C_p = 0.4011$，$888.02\,\text{N}\cdot\text{m}$）**，且 STL 体积仅增加 $+1.6\%$（$0.1470$ vs 基准 $0.1446\,\text{m}^3$）；`TuRBO-1` 均值同步跃升至 **$6.5920 \pm 0.0463\,\text{kW}$（$+8.27\%$）**；
- **`LLM-Meta-TuRBO` 统计显著超越 `TuRBO-1`**：10 种子配对检验 $t = 4.556$（$p = 0.00137$），Wilcoxon $p = 0.00977$，效应量 Cohen's $d = 1.44$，标准差仅为 `TuRBO-1` 的 $42\%$，并在第 100 步（$6.6434\,\text{kW}$）即已大幅超越 `TuRBO-1` 完整 300 步终值（$6.5920\,\text{kW}$），样本效率提升 $>3\times$；
- **纯原论文版 `MALIBO` 诚实揭示高维强约束连续流形瓶颈**：严格遵循原论文 Appendix G 的全局 5,120 点随机搜索（无信赖域）在 63 维强几何约束空间中难以命中极薄可行流形（均匀采样可行率 $0/5,120$），仅保留共享 $n_{\text{init}}=15$ 初始化阶段所得 $+1.46\%$ 增益，客观印证了原版 MALIBO 面向高维受约束 3D 气动外形优化时引入流形/信赖域机制的迫切价值。

---

## 2. 物理建模与 63 维三站位参数化空间

### 2.1 B-Spline Delta-Modal Model ($D=63$)
- **3 核心控制站位 CST 翼型参数（51 维）**：严格布设于纯 S809 气动升力段头/中/尾（$r = [1.2575, 3.14325, 5.0290]\,\text{m}$，对应 31 全余弦加密网格对称索引 `[0, 15, 30]`），每站位 8 上翼面 + 8 下翼面 + 1 前缘权重，闭口尾缘 $t_{\text{TE}} \equiv 0.0$；沿展向采用保单调三次 `PchipInterpolator` 插值，确保 3D 表面 $C^1$ 连续且零过冲起伏；
- **6 阶三次 B 样条弦长增量 $\Delta c_{\text{ctrl}}$（6 维）**：叠加至 TP-500-29955 基准线性锥度 $[0.7370 \to 0.3560]\,\text{m}$；
- **6 阶三次 B 样条扭角增量 $\Delta \beta_{\text{ctrl}}$（6 维）**：叠加至 TP-500-29955 基准扭角 $[20.040^\circ \to -1.815^\circ]$。

### 2.2 工程可制造性与物理边界约束
1. **单调弦长收缩与扭角洗退**：全 31 余弦网格严格强制 $\Delta c / \Delta r \le 0$（最小弦长 $\ge 0.15\,\text{m}$）且 $\Delta \beta / \Delta r \le 0$；
2. **等距参考网格曲率与线性偏离校验**：在等距 27 点参考网格上评估中段线性偏离（$\le 0.045\,\text{m}$）与二阶差分曲率，避免非均匀余弦步长造成的虚假二阶差分偏差；
3. **翼型不自交与厚度包络**：Root $[12\%, 28\%]$、Mid $[9\%, 25\%]$、Tip $[7\%, 24\%]$，保证主梁刚度与前缘圆弧半径；
4. **整轮推力上限**：$T_{\text{rotor}} \le 1.15\,T_{\text{baseline}} = 1.3954\,\text{kN}$。

---

## 3. 出版级学术图表与水密 CFD 模型

全套图表保存于 `figures/`（300 DPI PNG、PDF 及 Plotly 3D HTML），CFD 几何保存于 `cfd_models/`：
1. `figures/blade_convergence_curves.png/pdf`：300 NFE 全程收敛曲线与 SEM 误差带；
2. `figures/blade_convergence_curves_100nfe.png/pdf`：前 100 NFE 快速启动收敛聚焦对比图；
3. `figures/blade_bar_comparison.png/pdf`：10 个独立随机种子逐一对照柱状图；
4. `figures/blade_boxplot_comparison.png/pdf`：10 种子统计分位箱线图与散点分布；
5. `figures/blade_chord_twist_distributions.png/pdf`：31 全余弦截面弦长 $c(r/R)$ 与扭角 $\beta(r/R)$ 展向分布对比；
6. `figures/blade_station_airfoils.png/pdf`：3 核心控制站位（Root / Mid / Tip）翼型轮廓演进对比；
7. `figures/blade_spanwise_angles.png/pdf`：31 全余弦截面几何扭角 $\beta$、有效攻角 $\alpha$ 与总节距角分布；
8. `figures/blade_3d_sideview.png/pdf`：3D 放样叶片俯视（$X$-$r$）与侧视（$Y$-$r$）正交光顺度检验图；
9. `figures/blade_3d_rendering*.html`：Baseline vs MALIBO / TuRBO-1 / LLM-Meta-TuRBO / All-4-Methods 3D 交互渲染；
10. `cfd_models/blade_*.stl`：100% 水密（闭合 2-Manifold）3D STL 实体表面网格，统一以 30% 弦长为 $X$ 基准轴（详见 [cfd_models/README.md](cfd_models/README.md)）。

---

## 4. 代码仓库结构

```
3D-wind-turbine-blade/
├── src/
│   ├── blade_geometry.py         # NREL Phase VI 31全余弦截面与3站位 PCHIP 三维放样 (R=5.029m, 0.30c 轴线)
│   ├── parameterizations.py      # B-Spline Delta-Modal 模型 (D=63) 与 Stations3 对比模型 (D=60)
│   ├── evaluator.py              # 31全余弦 BEM + NeuralFoil(xxxlarge) + 动态 Viterna + Snel Stall Delay 评估器
│   ├── llm_agent.py              # 真实 SGLang 大模型元控制智能体 (3 站位 D=63 提示词)
│   ├── optimizers/
│   │   ├── base.py               # 优化器基类 (严格可行域正分准入控门)
│   │   ├── turbo.py              # 高维信赖域 TuRBO-1 优化器 (p_perturb = 20/63)
│   │   └── llm_turbo.py          # 停滞自适应 LLM-Meta-TuRBO (post_init + stag_threshold=10)
│   └── malibo/
│       ├── config.py             # 严格对齐 Pan et al. (ICML 2024, Appendix G) 原始超参 (无 TR)
│       ├── losses.py             # 精确 Eq. (2) LFBOLoss 与 Eq. (4)/Appendix C 排序 KS + Cov 高斯先验正则
│       ├── models.py             # 原论文 ResFFN (4x64, ELU, 无 LayerNorm, d=50) + MeanHead (50)
│       ├── dataset.py            # 标准化任务尺度 + 精确 Eq. (2) LFBO 权重数据集
│       ├── dataset_generator.py  # 100 任务 Multi-Pitch/Multi-Start 200 NFE (D=63, 31 截面) 物理轨迹采集器
│       ├── trainer.py            # Eq. (3) Adam(lr=1e-3) + ExponentialLR(0.999) 元训练器
│       ├── adaptation.py         # 精确 Eq. (6)-(9) L-BFGS MAP + Laplace Precision + 纯净 Thompson 采样
│       ├── residual_booster.py   # Eq. (10) & Appendix G LFBO 双标签 70/30 早停残差梯度提升树
│       └── optimizer.py          # 严格对齐 Algorithm 1 全局 5,120 随机采样 MALIBO 优化器 (无信赖域)
├── data/
│   └── baseline_nrel_phase6/     # S809 坐标、TP-500-29955 几何表、build_baseline.py 与 31 全余弦 .npz
├── weights/malibo_checkpoint.pt  # 训练完成的原论文对齐版 MALIBO 权重 (Val Loss=0.0272, z std=1.0057)
├── benchmark.py                  # 10 种子全量基准评测入口 (7.0 m/s, 72 rpm, 4.815 deg, 31 截面, 300 NFE)
├── benchmark_parameterizations.py# 参数化表示与光滑度对比脚本 (D=60 vs D=63)
├── train_malibo.py               # MALIBO 元训练命令行脚本 (Appendix G epochs=2048)
├── plot_results.py               # 全套出版级图表、侧视光顺检验图与 CFD 水密 STL 导出脚本
├── results/
│   ├── benchmark_results.json    # 10 种子全量对比实验原始数值数据
│   ├── benchmark_results.npz     # 压缩 NumPy 收敛曲线档案
│   └── research_report.md        # 详尽科研实验报告
├── figures/                      # PNG/PDF 图表与交互式 3D HTML
├── cfd_models/                   # 4 套水密 CFD STL 表面网格与 meshing 说明
└── tests/                        # 14 项单元测试与物理合规/原论文对齐检验
```

---

## 5. 快速复现指南

```bash
# 1. 重建 NREL Phase VI 31 全余弦加密截面 S809 基准数据 (已内置)
python3 data/baseline_nrel_phase6/build_baseline.py

# 2. 生成 100 任务 D=63 MALIBO 离线数据集并按 Appendix G 训练元模型
python3 -m src.malibo.dataset_generator
python3 train_malibo.py

# 3. 执行原论文对齐版 MALIBO 10 种子测试 (建议 GPU 独占)
python3 benchmark.py --force-rerun --skip-turbo --skip-llm-meta --workers 5

# 4. 运行 TuRBO-1 10 种子测试，启动本地 SGLang (端口 30000) 并执行 LLM-Meta-TuRBO 10 种子测试
python3 benchmark.py --skip-malibo --skip-llm-meta --workers 5
bash /home/msds/c250089/notebook/repos/RL-Airfoil/launch_sglang.sh
python3 benchmark.py --skip-turbo --skip-malibo --workers 5

# 5. 导出全部出版级图表与水密 CFD STL 并运行单元测试
python3 plot_results.py
python3 -m unittest discover -s tests -p "test_*.py" -v
```
