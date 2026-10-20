# 3D Wind Turbine Blade Aerodynamic Optimization Benchmark

本项目面向大型兆瓦级风力发电机（NREL 5-MW 尺度，风轮半径 $R = 63.0$ m，叶片数 $B = 3$）的单个 3D 叶片气动外形优化设计。

为了丰富学术论文并验证元优化方法在复杂高维工程场景下的泛化能力，本项目将先前的 **2D-Airfoil** 研究拓展至工业级 **60 维 3D 叶片气动外形多截面耦合优化**。在统一函数评估预算（NFE = 300）与工业几何约束下，系统对比了三种前沿优化算法：
1. **TuRBO-1**：高维信赖域贝叶斯优化（Trust Region Bayesian Optimization，采用算法发明者推荐的默认超参）；
2. **LLM-Meta-TuRBO**：结合风力机空气动力学专家先验与宏观定向几何形变注入的大语言模型元控制器（保留前期激进偏移量介入）；
3. **MALIBO**：多工况元学习无似然贝叶斯元优化器（Meta-learning for Likelihood-free Bayesian Optimization，基于风速 vs 转速网格热启动）。

---

## 1. 物理问题建模与气动求解体系

### 1.1 60 维叶片参数化与三维放样 (Parameterization)
风力机叶片沿展向具有明显的空气动力学与结构分区特性。本项目选取 3 个核心控制站位：
- **Station 1 (内展段, $r/R = 0.25$, $r = 16.9$ m)**: 结构承载厚翼型区（基准 NACA 4424, $t/c \approx 24\%$），弦长 $4.25$ m，扭角 $11.5^\circ$；
- **Station 2 (中展段, $r/R = 0.55$, $r = 35.3$ m)**: 主气动扭矩产生区（基准 NACA 4418, $t/c \approx 18\%$），弦长 $3.45$ m，扭角 $5.2^\circ$；
- **Station 3 (外展段, $r/R = 0.85$, $r = 53.8$ m)**: 高速低阻叶尖区（基准 NACA 4412, $t/c \approx 12\%$），弦长 $2.15$ m，扭角 $1.2^\circ$。

每个站位由 **20 个物理维度** 精确控制：
- 18 维 Kulfan (CST) 翼型参数：8 维上表面吸力面权重、8 维下表面压力面权重、1 维前缘钝度权重、1 维后缘厚度；
- 1 维局部截面弦长 $c$ (m)；
- 1 维局部截面安装扭角 $\beta$ (deg)。
全叶片设计空间维度：$D = 3 \times (18 + 1 + 1) = 60$ 维。

### 1.2 微秒级纯内存气动评估器 (In-Memory BEM + NeuralFoil)
传统的叶片评估器（如 OpenFAST / AeroDyn 或 QBlade）依赖硬盘临时文件与外部进程，单次评估耗时数秒至数分钟且易锁死。本项目开发了纯内存、高度向量化的 Python 求解器：
- **叶素动量理论 (BEM)**：沿展向离散为 15 个叶素微元，模拟轴向流动与叶片旋转诱导的速度亏损；
- **损失修正**：耦合 Prandtl 根尖损失模型 ($F_{tip}$ 与 $F_{root}$)；
- **高推力修正**：嵌入 Glauert 湍流尾流经验修正公式，解决轴向诱导因子 $a > 0.33$ 时经典动量理论失效问题；
- **神经网络翼型代理 (NeuralFoil)**：实时推断各截面在当前局部雷诺数 $Re(r)$ 与攻角 $\alpha(r)$ 下的高保真升力系数 $C_l$ 与阻力系数 $C_d$；
- **单次评估速度**：单个 60 维叶片的全气动与几何评估仅需 **15 ~ 25 毫秒**，比传统硬盘仿真器提速 3 个数量级以上。

### 1.3 工业级气动与几何强约束 (Industrial Constraints & Dominant Penalty)
为确保优化生成的叶片具备真实的工程制造可行性与结构强度，设立多重强约束校验：
1. **各站位翼型厚度包络**：
   - Station 1: 相对厚度 $t/c \in [20\%, 35\%]$；
   - Station 2: 相对厚度 $t/c \in [15\%, 26\%]$；
   - Station 3: 相对厚度 $t/c \in [10\%, 18\%]$；
2. **结构主梁站位厚度**：40% 弦长位置保留足够的碳纤维/玻璃钢主梁与剪切腹板厚度；
3. **后缘防鱼尾单调性 (Anti-Fishtailing)**：从最大厚度位置到后缘，厚度必须严格单调递减；
4. **吸力面抗凹陷波纹 (Anti-Waviness)**：二阶导数曲率检验，杜绝 CST 高阶多项式振荡震荡与凹坑；
5. **展向宏观几何拓扑约束**：
   - 弦长平滑单调收缩：$c_1 \ge c_2 \ge c_3$（杜绝局部异常鼓包或反向扩散）；
   - 气动扭退单调洗出：$\beta_1 \ge \beta_2 \ge \beta_3$（匹配展向逐渐减小的入流角 $\phi$）；
6. **主导惩罚机制 (Dominant Penalty)**：物理缺陷前置拦截，微弱越界施加阶跃陡峭排斥惩罚（500 ~ 2000 kW），彻底杜绝不可行解破坏代理模型连续性。

---

## 2. 三种优化算法设计与对比

### 2.1 TuRBO-1 (高维基线)
- 严格遵从 Eriksson et al. (NeurIPS 2019) 发明者推荐的高维默认参数：
  - 初始信赖域宽度 $L_{init} = 0.8$, $L_{min} = 0.5^7$, $L_{max} = 1.6$；
  - 成功容忍步数 $\tau_{succ} = 3$，连续失败容忍步数 $\tau_{fail} = 6$（高维推荐）；
  - 坐标微扰概率 $p_{perturb} = \min(20.0 / 60, 1.0) = 0.333$；
  - 5,000 个 Sobol 拟蒙特卡洛候选点与 Thompson 采样。

### 2.2 LLM-Meta-TuRBO (大模型宏观元控制)
- 重新设计的提示词注入了三维叶片旋转动力学先验（中展与翼尖对转矩贡献 >75%、扭角与入流角 $\phi = \arctan(V_a / \Omega r)$ 匹配等）；
- 感知 60 维叶片各站位当前弦长、扭角、最大厚度、风能利用率 $C_p$、推力 $C_t$ 与信赖域停滞状态；
- **激进偏移量注入 (Aggressive Offset Injection)**：在前期探索阶段，将大模型决策的宏观气动形变方向直接转换为 60 维参数向量偏移量，前置检验通过后作为高优先级候选点直接真实评估，实现跨越式极速收敛。

### 2.3 MALIBO (多工况元学习贝叶斯元优化)
- **网格热启动数据集生成**：
  围绕基准测试工况（$V_{test} = 9.0$ m/s, $\Omega_{test} = 10.0$ rpm），构建了 $4 \times 4 = 16$ 个完全避开测试值的周边工况网格：
  $V \in [7.0, 8.0, 10.0, 11.0]$ m/s, $\Omega \in [8.0, 9.0, 11.0, 12.0]$ rpm。
  利用 PSO 收集了 2,400 条多工况优化轨迹数据；
- **元神经网络架构**：ResFFN（4 个残差块映射至 50 维特征空间）+ 任务无关均值网络 MeanHead + 16 个任务嵌入向量 $\{z_t\}$；
- **在线优化机制**：零观测步直接由 MeanHead 推荐全局先验热启动构型；随后通过拉普拉斯逼近在线求解 $z_{MAP}$ 与 Thompson 采样，结合残差梯度提升树 (GBDT) 极大化综合后验效用。

---

## 3. 代码仓库结构

```
3D-wind-turbine-blade/
├── src/
│   ├── blade_geometry.py       # 60 维叶片几何参数化、展向插值与 3D 放样
│   ├── evaluator.py            # 工业级 BEM + NeuralFoil 气动与几何综合评估器
│   ├── llm_agent.py            # LLM-Meta-Blade 领域先验智能体
│   ├── optimizers/
│   │   ├── base.py             # 优化器通用抽象基类
│   │   ├── turbo.py            # 标准 60 维 TuRBO-1 优化器
│   │   └── llm_turbo.py        # LLM-Meta-TuRBO 优化器
│   └── malibo/
│       ├── config.py           # MALIBO 结构与优化超参
│       ├── models.py           # ResFFN + MeanHead + Task Embeddings 模型
│       ├── dataset.py          # 多任务 LFBO 数据集加载器
│       ├── dataset_generator.py# 16 工况网格数据集生成脚本
│       ├── trainer.py          # 多任务元训练器
│       ├── adaptation.py       # 拉普拉斯在线任务自适应与 Thompson 采样
│       ├── residual_booster.py # 残差梯度提升纠偏模块
│       └── optimizer.py        # MALIBO 贝叶斯元优化器驱动
├── benchmark.py                # 10 随机种子全量基准测试入口
├── plot_results.py             # 出版级 4 张学术对比图表生成器
├── train_malibo.py             # MALIBO 神经网络离线元训练脚本
├── results/                    # 基准测试数值数据 (JSON & NPZ)
├── figures/                    # 导出的学术图表 (PNG & PDF)
└── weights/                    # 训练保存的模型权重 (受 .gitignore 保护)
```

---

## 4. 快速复现与实验运行指南

### 4.1 环境依赖与安装
本项目基于 Python 3.12 与 PyTorch。推荐使用 `uv` 极速安装核心依赖：
```bash
uv pip install numpy scipy matplotlib torch aerosandbox neuralfoil scikit-learn openai
```

### 4.2 MALIBO 训练集生成与元训练
```bash
# 1. 生成 16 工况网格数据集 (约 50 秒完成)
python -m src.malibo.dataset_generator

# 2. 执行神经网络元训练 (GPU 加速，约 3 秒完成)
python train_malibo.py --epochs 600
```

### 4.3 全量 10 随机种子基准对比测试
在统一工况（$V_\infty = 9.0$ m/s, $\Omega = 10.0$ rpm, 300 次评估预算）下运行测试：
```bash
python benchmark.py --max-evals 300
```

### 4.4 导出高清出版图表
```bash
python plot_results.py
```
图表将保存于 `figures/` 目录，涵盖：
- `blade_convergence_curves.png / .pdf`：平均收敛曲线对比与标准误差带；
- `blade_bar_comparison.png / .pdf`：10 个随机种子性能对比柱状图；
- `blade_boxplot_comparison.png / .pdf`：箱线图统计分布与抖动散点；
- `blade_shapes_comparison.png / .pdf`：3D 展向弦长、扭角分布及核心站位剖面对比。
