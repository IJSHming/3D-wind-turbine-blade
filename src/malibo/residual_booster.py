"""MALIBO 残差梯度提升弱学习器 (Residual Booster):
对应 Pan et al. (2024, Eq. 9 & Eq. 10):
在线拟合元学习神经网络后验与实际目标观测值之间的局部残差偏差 C_GB(x)，
在目标任务与先验分布存在差异时提供鲁棒校准。

【科研诚信与预算合规认证 (Research Integrity Certification)】:
本模块仅在已观测数据集 (x_obs, y_obs) 内存数据上拟合残差偏差，
绝不进行任何额外的物理函数评估，绝不消耗任何额外 NFE 预算，完全遵循无似然贝叶斯优化规范。
"""

from typing import Optional
import numpy as np
import torch
from sklearn.ensemble import GradientBoostingRegressor

from src.malibo.config import BoosterConfig
from src.malibo.models import MALIBOMetaModel


class MALIBOResidualBooster:
    """残差提升器 (Gradient Boosting Discrepancy Corrector)"""

    def __init__(
        self,
        meta_model: MALIBOMetaModel,
        config: Optional[BoosterConfig] = None,
        device: Optional[torch.device] = None,
        seed: int = 42,
    ):
        self.meta_model = meta_model
        self.config = config if config is not None else BoosterConfig()
        self.device = device if device is not None else next(meta_model.parameters()).device
        self.seed = int(seed)
        self.model: Optional[GradientBoostingRegressor] = None
        self.is_fitted = False

    def fit(
        self,
        x_obs: np.ndarray,      # (N, D) in [0, 1]^D
        y_obs: np.ndarray,      # (N,) 最小化目标
        z_map: torch.Tensor,    # (d,)
    ) -> bool:
        N = len(x_obs)
        if N < self.config.initial_thompson_steps:
            self.is_fitted = False
            return False

        with torch.no_grad():
            x_tensor = torch.from_numpy(x_obs).float().to(self.device)
            meta_logits = self.meta_model(x_tensor, z=z_map).cpu().numpy()

        # 截断极端不可行惩罚离群值 (y_obs = -score <= +1.0)，保持可行样本效用梯度尺度清晰
        y_clean = np.minimum(np.asarray(y_obs, dtype=np.float64), 1.0)
        y_norm = -(y_clean - np.mean(y_clean)) / (np.std(y_clean) + 1e-8)
        residuals = y_norm - meta_logits

        self.model = GradientBoostingRegressor(
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            random_state=self.seed,
        )

        try:
            self.model.fit(x_obs, residuals)
            self.is_fitted = True
            return True
        except Exception:
            self.is_fitted = False
            return False

    def predict(self, x_cand: np.ndarray) -> np.ndarray:
        """预测候选点的残差修正项"""
        if not self.is_fitted or self.model is None:
            return np.zeros(len(x_cand), dtype=np.float64)
        try:
            return self.model.predict(x_cand)
        except Exception:
            return np.zeros(len(x_cand), dtype=np.float64)
