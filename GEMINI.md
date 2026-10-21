# 3D-Wind-Turbine-Blade 科研实验与学术规范准则 (Workspace Rules)

本规范为 3D 风力机叶片高维气动外形优化项目定制，严格作为算法设计、代码编写、数据生成与论文发表的不可逾越红线。

## 1. 绝对科研诚信与零静默降级原则 (Absolute Research Integrity & Zero Silent Degradation)
1. **严禁静默降级（No Silent Heuristic Fallback）**：
   - 凡涉及“LLM-Meta”、“大模型元优化”的方法，必须依赖真实的外部模型推理（如通过本地服务或官方 API）。
   - 严禁在 LLM 连接失败或解析异常时，静默退化走人工硬编码的启发式字典（如包含精心挑选的几何偏移量）；
   - 一旦 LLM 服务离线、超时或解析失败，必须立刻抛出明确异常或显式中止任务，杜绝“以人工启发式冒充大模型智能”。
2. **严禁后验数据窥探（Zero Data-Snooping / Oracle Leaks）**：
   - 先验提示词与系统 Prompt 必须仅包含风力机气动学公理与设计边界，严禁将已知最优解的具体数值或坐标反向硬塞给大模型作为“先验”。

## 2. 实验协议严格对齐原则 (Rigorous Experimental Protocol Alignment)
1. **初始化预算与时序严格对齐**：
   - 所有对比算法（TuRBO、LLM-Meta-TuRBO、MALIBO）必须使用完全一致的初始化预算（统一 $n_{\text{init}} = 15$）；
   - 在 $n_{\text{init}}$ 阶段内，必须公平评估基准构型及相同的初始微扰散点，严禁任何算法在初始化阶段插队评估先验解或提前截断初始化预算。
2. **元控制干预时机合规**：
   - LLM-Meta 的宏观干预必须发生在初始化阶段完成之后（$NFE \ge 16$），依据真实历史轨迹和信赖域状态进行自适应决策。

## 3. 物理客观性与学术真实性 (Physical Objectivity & Scientific Realism)
1. **尊重 2D 翼型与 3D 叶片的客观物理差异**：
   - 严禁为了迎合 2D-Airfoil 上的特定结论而人为扭曲 3D-Blade 的优化进程；
   - 2D NACA 0012 为零弯度对称翼型，宏观先验易于破局；3D IEA 22MW 叶片为高度优化的 120 维强耦合非线性系统，不同算法展现出不同阶段特性具有深厚的物理必然性。应将此机理差异作为论文核心学术贡献，而非掩盖差异。

## 4. 进程与算力安全分配 (Process Safety & Hardware Acceleration)
1. **多进程并发与 CUDA 解耦**：
   - NeuralFoil 与 BEM 评估统一在 CPU 10 进程并发计算（采用 spawn 启动模式），杜绝 CUDA 上下文污染；
   - L40S GPU 专用保障本地大模型服务（SGLang / vLLM）与 MALIBO 神经网络的高效显存推理。
