"""MALIBO 架构、元训练、任务自适应与优化超参数配置模块：
严格对齐原论文 Pan et al. (ICML 2024, Section 4, Algorithm 1 & Appendix G) 全部原始超参数，
不包含信赖域 (Trust Region)、稀疏坐标掩码、gamma 退火或额外探索方差地板等非原论文扩展。
"""

from dataclasses import dataclass, field


@dataclass
class ResFFNConfig:
    """特征提取残差前馈网络 (ResFFN) 结构超参 (Appendix G: 4 hidden layers x 64 units, ELU, d=50)"""
    in_dim: int = 63                 # 63 维 B-Spline Modal 叶片设计参数 (3*17 CST + 6 Chord + 6 Twist)
    hidden_dim: int = 64             # 原论文 Appendix G: 64 units per hidden layer
    num_hidden_layers: int = 4       # 原论文 Appendix G: 4 hidden layers (ResBlocks)
    feature_dim: int = 50            # 原论文 Appendix G: latent feature & task embedding dim d = 50
    activation: str = "elu"          # 原论文 Appendix G: ELU activation


@dataclass
class TrainingConfig:
    """元训练 (Meta-Training) 超参数配置 (Appendix G)"""
    lr: float = 1e-3                 # Adam lr = 1e-3 (no weight decay)
    lr_decay_factor: float = 0.999   # Exponential decay factor 0.999 per epoch
    batch_size: int = 256            # Batch size B = 256
    epochs: int = 2048               # Up to 2,048 epochs with early stopping
    early_stopping_patience: int = 80
    reg_lambda: float = 0.1          # Eq. (3) regularization factor lambda = 0.1
    gamma: float = 1.0 / 3.0         # Eq. (2) fixed quantile threshold gamma = 1/3
    seed: int = 42
    device: str = "cuda"


@dataclass
class AdaptationConfig:
    """目标任务在线拉普拉斯自适应 L-BFGS 超参数 (Appendix G)"""
    lr: float = 1.0
    max_iter: int = 20
    tolerance_grad: float = 1e-7
    tolerance_change: float = 1e-9
    history_size: int = 100
    line_search_fn: str = "strong_wolfe"


@dataclass
class BoosterConfig:
    """残差梯度提升分类器配置 (Pan et al., 2024 Eq. 10 & Appendix G, LFBO defaults)"""
    n_estimators: int = 100
    learning_rate: float = 0.1
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_depth: int = 3
    val_ratio: float = 0.30          # Appendix G: 70% train / 30% validation early stopping


@dataclass
class OptimizationConfig:
    """MALIBO 原始全局采集优化配置 (Algorithm 1 & Appendix G: 纯全局 5,120 随机采样，无信赖域)"""
    n_candidates: int = 5120         # Appendix G: 5,120 global random search samples in [0, 1]^D
    budget: int = 300                # 目标任务评估预算上限
    seed: int = 42
    n_init: int = 15                 # 初始设计点数 (与 TuRBO-1 / LLM-Meta-TuRBO 共享统一基线初始化协议)
    gamma: float = 1.0 / 3.0         # Appendix G: 固定 gamma = 1/3 (无退火)


@dataclass
class MALIBOBladeConfig:
    """MALIBO 综合全局配置"""
    resffn: ResFFNConfig = field(default_factory=ResFFNConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)
    booster: BoosterConfig = field(default_factory=BoosterConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
