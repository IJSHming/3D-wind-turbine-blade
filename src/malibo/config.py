"""MALIBO 架构、元训练、任务自适应与优化超参数配置模块：
严格对齐 Pan et al. (2024, Appendix G) 官方超参数，适配 60 维 3D 风力机叶片气动优化。
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ResFFNConfig:
    """特征提取残差前馈网络 (ResFFN) 结构超参"""
    in_dim: int = 60                 # 60 维叶片设计参数
    hidden_dim: int = 64             # 隐层单元数 (每层 64)
    num_hidden_layers: int = 4       # 4 个残差块
    feature_dim: int = 50            # 潜在特征空间维度 d = 50
    activation: str = "elu"


@dataclass
class TrainingConfig:
    """元训练 (Meta-Training) 超参数配置"""
    lr: float = 1e-3
    lr_decay_factor: float = 0.999
    batch_size: int = 256
    epochs: int = 1200
    early_stopping_patience: int = 60
    reg_lambda: float = 0.1          # 任务嵌入高斯先验正则项权重
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
    """残差梯度提升弱学习器配置"""
    n_estimators: int = 80
    learning_rate: float = 0.1
    min_samples_split: int = 2
    min_samples_leaf: int = 2
    max_depth: int = 3
    val_ratio: float = 0.25
    initial_thompson_steps: int = 15


@dataclass
class OptimizationConfig:
    """MALIBO 贝叶斯元优化运行配置"""
    n_candidates: int = 5120         # 每代采集函数极大化采样候选点数
    budget: int = 300                # 测试优化步数上限
    seed: int = 42
    n_init: int = 15                 # 初始采样点数
    tr_len_init: float = 0.40        # 初始信任域宽度
    tr_len_restart: float = 0.35
    tr_len_min: float = 0.5**7
    tr_len_max: float = 1.0
    success_tolerance: int = 3
    failure_tolerance: int = 5
    prob_perturb: float = 0.35
    gamma_init: float = 1.0 / 3.0
    gamma_min: float = 0.08
    exploration_floor: float = 0.15


@dataclass
class MALIBOBladeConfig:
    """MALIBO 综合全局配置"""
    resffn: ResFFNConfig = field(default_factory=ResFFNConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)
    booster: BoosterConfig = field(default_factory=BoosterConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
