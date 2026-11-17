"""MALIBO 架构、元训练、任务自适应与优化超参数配置模块：
严格对齐 Pan et al. (2024, Appendix G) 核心超参数，并适配 114 维 NREL Phase VI 3D 风力机叶片连续设计空间。
"""

from dataclasses import dataclass, field


@dataclass
class ResFFNConfig:
    """特征提取残差前馈网络 (ResFFN) 结构超参"""
    in_dim: int = 114                # 114 维 B-Spline Modal 叶片设计参数 (102 CST + 6 Chord + 6 Twist)
    hidden_dim: int = 128            # 隐层单元数
    num_hidden_layers: int = 4       # 4 个残差块
    feature_dim: int = 64            # 潜在特征空间维度 d = 64
    activation: str = "elu"


@dataclass
class TrainingConfig:
    """元训练 (Meta-Training) 超参数配置"""
    lr: float = 1e-3
    lr_decay_factor: float = 0.999
    weight_decay: float = 1e-4       # 骨干特征/均值头轻微正则，防止外推 Logit 爆炸
    batch_size: int = 256
    epochs: int = 1200
    early_stopping_patience: int = 80
    reg_lambda: float = 0.1          # 任务嵌入标准高斯先验正则权重
    gamma: float = 1.0 / 3.0         # 优质样本分位数 (前 1/3)
    seed: int = 42
    device: str = "cuda"


@dataclass
class AdaptationConfig:
    """目标任务在线自适应超参数"""
    lr: float = 1.0
    max_iter: int = 20
    tolerance_grad: float = 1e-6
    tolerance_change: float = 1e-8
    history_size: int = 50
    line_search_fn: str = "strong_wolfe"


@dataclass
class BoosterConfig:
    """残差梯度提升分类弱学习器配置 (Pan et al., 2024 Eq. 10 & Appendix G)"""
    n_estimators: int = 100
    learning_rate: float = 0.08
    subsample: float = 0.85
    min_samples_split: int = 2
    min_samples_leaf: int = 2
    max_depth: int = 3
    val_ratio: float = 0.25
    initial_thompson_steps: int = 15


@dataclass
class OptimizationConfig:
    """MALIBO 贝叶斯元优化运行配置 (对齐 D=114 高维信赖域与准蒙特卡洛采集)"""
    n_candidates: int = 5120         # 每代采集函数极大化 Sobol 候选点数 (Appendix G: 5,120)
    budget: int = 300                # 测试优化步数上限
    seed: int = 42
    n_init: int = 15                 # 初始采样点数 (严格与 TuRBO-1 / LLM-Meta 对齐)
    tr_len_init: float = 0.80        # 对齐 114 维 TuRBO-1 初始信赖域宽度，避免中段过早坍缩
    tr_len_restart: float = 0.60     # 重启尺度
    tr_len_min: float = 0.5**7
    tr_len_max: float = 1.6
    success_tolerance: int = 3
    failure_tolerance: int = 6       # 对齐 114 维 TuRBO-1 失败容忍阈值
    prob_perturb: float = 0.18       # 约 min(20/114, 1) = 0.175，避免高维全坐标同时扰动
    gamma_init: float = 1.0 / 3.0
    gamma_min: float = 0.08
    exploration_floor: float = 0.12


@dataclass
class MALIBOBladeConfig:
    """MALIBO 综合全局配置"""
    resffn: ResFFNConfig = field(default_factory=ResFFNConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)
    booster: BoosterConfig = field(default_factory=BoosterConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
