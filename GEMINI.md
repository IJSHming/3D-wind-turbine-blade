# 3D-Wind-Turbine-Blade 科研实验与学术规范准则 (Workspace Rules)

本规范为 **NREL Phase VI (Sequence H) 3D 风力机叶片**高维气动外形优化项目定制，严格作为算法设计、代码编写、数据生成与论文发表的不可逾越红线。

## 1. 绝对科研诚信与零静默降级原则 (Absolute Research Integrity & Zero Silent Degradation)
1. **严禁静默降级（No Silent Heuristic Fallback）**：
   - 凡涉及“LLM-Meta”、“大模型元优化”的方法，必须依赖真实的外部模型推理（如通过本地 SGLang 服务调用 `Qwen/Qwen3.8-27B-FP8`）。
   - 严禁在 LLM 连接失败或解析异常时，静默退化走人工硬编码的启发式字典（如包含精心挑选的几何偏移量），亦严禁在 BEM / NeuralFoil 求解失败时静默返回常数值或假极线；
   - 一旦 LLM 服务离线、超时或解析失败，必须立刻抛出明确异常或显式中止任务，杜绝“以人工启发式冒充大模型智能”。
2. **严禁后验数据窥探（Zero Data-Snooping / Oracle Leaks）**：
   - 先验提示词与系统 Prompt 必须仅包含风力机气动学公理与设计边界，严禁将已知最优解的具体数值或坐标反向硬塞给大模型作为“先验”。
3. **严格可行域隔离与零违规报告**：
   - 任何违反几何平滑约束或推力安全上限（$T \le 1.15 \times T_{\text{baseline}}$）的构型均必须在评估函数中返回严格负值（$< 0.0$），严禁微弱违规解以正分混入最优可行解统计。

## 2. 实验协议严格对齐原则 (Rigorous Experimental Protocol Alignment)
1. **初始化预算与时序严格对齐**：
   - 所有对比算法（TuRBO、LLM-Meta-TuRBO、MALIBO）必须使用完全一致的初始化预算（统一 $n_{\text{init}} = 15$，相同随机种子与归一空间对称初始化）；
   - 在 $n_{\text{init}}$ 阶段内，必须公平评估基准构型及相同的初始微扰散点，严禁任何算法在初始化阶段插队评估先验解或提前截断初始化预算。
2. **元控制干预时机合规**：
   - LLM-Meta 的宏观干预必须发生在初始化阶段完成之后（$\text{NFE} \ge 16$），依据真实历史轨迹和信赖域状态进行自适应决策。

## 3. 物理客观性与学术真实性 (Physical Objectivity & Scientific Realism)
1. **尊重 2D 翼型与 3D 叶片的客观物理差异**：
   - 严禁为了迎合 2D 翼型结论而人为扭曲 3D 叶片优化进程或篡改超参数公平性；
   - NREL Phase VI 为 S809 非对称翼型高度非线性双叶片系统（114 维 B 样条模态参数化），基准已具较合理扭角分布与气动效率（$C_p = 0.3637$），应以推力受限下弯度/弦长/扭角协同增益及其方差/早期收敛特性作为客观学术贡献。

## 4. 进程与算力安全分配 (Process Safety & Hardware Acceleration)
1. **多进程并发与 CUDA 解耦**：
   - NeuralFoil 与 BEM 评估统一在 CPU 多进程并发计算（采用 `spawn` 启动模式并锁定单线程 BLAS），杜绝 CUDA 上下文污染；
   - L40S GPU 专用保障本地大模型服务（SGLang）与 MALIBO 神经网络的高效显存推理（注意显存错峰管理，避免与 SGLang KV Cache 冲突）。

## 5. 参数化锁定与 NREL Phase VI 物理保真准则 (Parameterization Locking & Geometric Fidelity)
1. **统一锁定 B-Spline Modal ($D=114$)**：主干模型锁定为 6 控制站位 17 维闭口尾缘 CST（102 维，$t_{\text{TE}} = 0.0$）+ 6 阶三次 B 样条弦长增量（6 维）+ 6 阶三次 B 样条扭角增量（6 维），对应真实设计空间 $D=114$；
2. **基准比特级保真**：叶片在 $\mathbf{0}$ 增量与 S809 CST 基准向量 $x_0$ 处精确重构 NREL Phase VI (Sequence H) 气动段几何（`data/baseline_nrel_phase6/baseline_27_sections.npz`）；
3. **单步评估耗时红线**：单次叶片气动评估耗时必须严格控制在 $1.0\,\text{s}$ 红线内（实测 $\approx 0.10\,\text{s}$），依托向量化极线预计算消除重复计算；
4. **工业中段平滑与单调洗退准则**：弦长与扭角沿展向严格单调非增收缩/洗退，中段线性偏离与二阶差分保持工程可制造水平，杜绝波浪形震荡；
5. **对齐 NREL Phase VI 气动规范**：27 个展向微元覆盖升力主体段 $r \in [1.2575\,\text{m}, 5.0290\,\text{m}]$（$L=3.7715\,\text{m}$），30% 弦长（`0.30c`）作为变桨/扭转中心轴，空气工况 $V_\infty = 7.0\,\text{m/s}, \Omega = 72.0\,\text{rpm}, \theta_{p,75\%} = 4.815^\circ$（等效叶尖 $3.0^\circ$），$\rho = 1.225\,\text{kg/m}^3, \mu = 1.789 \times 10^{-5}\,\text{Pa}\cdot\text{s}$。
