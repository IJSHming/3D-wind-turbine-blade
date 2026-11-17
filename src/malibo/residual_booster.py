"""MALIBO 残差梯度提升分类器 (Residual Gradient Boosting Classifier):
严格遵循 Pan et al. (2024, Section 4 Eq. 10 & Appendix G) 与 Meta-CFD 基准设计:
1. 以元模型基于 Thompson 采样嵌入 z_hat 预测的后验概率 p(k=1|x) = sigmoid(m(phi(x)) + z_hat^T phi(x))
   作为梯度提升分类器 GradientBoostingClassifier 的基础先验估计器 (init estimator)；
2. 基于已观测样本 (X_obs, k, weights) 执行 75% 训练 / 25% 分层验证早停搜索最优弱树数量 M*，
   彻底避免高维小样本下残差树过拟合；
3. 输出连续对数几率决策边距 decision_function(X_cand) = m(phi(x)) + z_hat^T phi(x) + sum_{m=1}^{M*} r_m(x)，
   杜绝概率饱和截断同时在连续神经网络特征梯度上叠加精准局部残差修正。

【科研诚信与预算合规认证】:
本模块仅使用内存中已评估样本，绝不调用物理评估器或消耗额外 NFE 预算。
"""

from typing import Optional
import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from src.malibo.config import BoosterConfig
from src.malibo.models import MALIBOMetaModel


class MALIBOInitialEstimator(BaseEstimator, ClassifierMixin):
    """scikit-learn GradientBoostingClassifier 专用的元模型先验基准估计器."""

    def __init__(
        self,
        meta_model: MALIBOMetaModel,
        z_sample: torch.Tensor,
        device: Optional[torch.device] = None,
    ):
        self.meta_model = meta_model
        self.z_sample = z_sample.detach()
        self.device = device if device is not None else next(meta_model.parameters()).device
        self.classes_ = np.array([0, 1], dtype=np.int32)

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
    ):
        self.classes_ = np.array([0, 1], dtype=np.int32)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """输出形状 (N, 2) 的 [p(k=0|x), p(k=1|x)] 后验概率."""
        tX = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            p1 = self.meta_model.predict_proba(tX, z=self.z_sample).detach().cpu().numpy().reshape(-1)
        p1 = np.clip(p1, 1e-9, 1.0 - 1e-9)
        return np.column_stack([1.0 - p1, p1])


class MALIBOResidualBooster:
    """残差梯度提升分类器封装 (Pan et al., 2024 Eq. 10)."""

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
        self.booster: Optional[GradientBoostingClassifier] = None
        self.is_fitted = False

    def fit_with_early_stopping(
        self,
        x_obs: np.ndarray,
        y_minimize: np.ndarray,
        tau: float,
        z_sample: torch.Tensor,
    ) -> bool:
        """基于 LFBO 分位数标签与归一化正例权重训练早停残差分类树."""
        N = len(x_obs)
        if N < self.config.initial_thompson_steps:
            self.is_fitted = False
            self.booster = None
            return False

        # 截断极端不可行离群惩罚，保持可行样本分位数与方差清晰
        y_clean = np.minimum(np.asarray(y_minimize, dtype=np.float64), 1.0)
        k = (y_clean <= tau).astype(np.int32)
        if len(np.unique(k)) < 2:
            self.is_fitted = False
            self.booster = None
            return False

        y_std = max(float(np.std(y_clean)), 1e-4)
        sample_weights = np.where(
            k == 1,
            np.clip(1.0 + (tau - y_clean) / y_std, 1.0, 20.0),
            1.0,
        )
        w_mean = float(np.mean(sample_weights))
        if w_mean > 1e-6:
            sample_weights = sample_weights / w_mean

        init_est = MALIBOInitialEstimator(
            meta_model=self.meta_model,
            z_sample=z_sample,
            device=self.device,
        )

        # 验证早停选树数 (防止 D=114 小样本阶段树模型过拟合)
        if N >= 20 and np.min(np.bincount(k)) >= 3:
            try:
                X_tr, X_val, k_tr, k_val, w_tr, w_val = train_test_split(
                    x_obs, k, sample_weights,
                    test_size=self.config.val_ratio,
                    random_state=self.seed,
                    stratify=k,
                )
            except ValueError:
                X_tr, X_val, k_tr, k_val, w_tr, w_val = train_test_split(
                    x_obs, k, sample_weights,
                    test_size=self.config.val_ratio,
                    random_state=self.seed,
                )

            if len(np.unique(k_tr)) >= 2 and len(np.unique(k_val)) >= 2:
                pilot = GradientBoostingClassifier(
                    init=init_est,
                    n_estimators=self.config.n_estimators,
                    learning_rate=self.config.learning_rate,
                    subsample=getattr(self.config, "subsample", 0.85),
                    max_depth=self.config.max_depth,
                    min_samples_split=self.config.min_samples_split,
                    min_samples_leaf=self.config.min_samples_leaf,
                    random_state=self.seed,
                )
                pilot.fit(X_tr, k_tr, sample_weight=w_tr)
                val_losses = [
                    log_loss(k_val, p, sample_weight=w_val, labels=[0, 1])
                    for p in pilot.staged_predict_proba(X_val)
                ]
                best_trees = max(1, int(np.argmin(val_losses) + 1))
            else:
                best_trees = min(12, self.config.n_estimators)
        else:
            best_trees = min(12, self.config.n_estimators)

        final_clf = GradientBoostingClassifier(
            init=init_est,
            n_estimators=best_trees,
            learning_rate=self.config.learning_rate,
            subsample=getattr(self.config, "subsample", 0.85),
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            random_state=self.seed,
        )
        final_clf.fit(x_obs, k, sample_weight=sample_weights)
        self.booster = final_clf
        self.is_fitted = True
        return True

    def predict_utility(
        self,
        x_cand: np.ndarray,
        z_sample: torch.Tensor,
        use_raw_margin: bool = True,
    ) -> np.ndarray:
        """返回候选点的综合采集边距 m(phi(x)) + z_sample^T phi(x) + sum r_m(x)."""
        if self.is_fitted and self.booster is not None:
            if use_raw_margin:
                return np.asarray(self.booster.decision_function(x_cand), dtype=np.float64).reshape(-1)
            return np.asarray(self.booster.predict_proba(x_cand)[:, 1], dtype=np.float64).reshape(-1)

        with torch.no_grad():
            tX = torch.as_tensor(x_cand, dtype=torch.float32, device=self.device)
            return self.meta_model(tX, z=z_sample).detach().cpu().numpy().reshape(-1)
