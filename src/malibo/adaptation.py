"""MALIBO 在线目标任务拉普拉斯自适应器 (LaplaceTaskAdapter):
严格对齐 Pan et al. (2024, Section 4 & Algorithm 1):
1. 在线根据当前已观测样本集 D_t = {(x_i, y_i)} 求解任务嵌入的最大后验估计 z_MAP (Eq. 7);
2. 求解二阶 Hessian 矩阵以构建拉普拉斯高斯后验 q(z) = N(z_MAP, Sigma_z);
3. 执行 Thompson 采样抽取候选任务嵌入 z_hat ~ q(z)，提供探索与利用的贝叶斯动态平衡。
"""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from src.malibo.config import AdaptationConfig
from src.malibo.models import MALIBOMetaModel


class LaplaceTaskAdapter:
    """拉普拉斯在线任务自适应与汤普森采样器"""

    def __init__(
        self,
        meta_model: MALIBOMetaModel,
        config: Optional[AdaptationConfig] = None,
        gamma: float = 1.0 / 3.0,
        device: Optional[torch.device] = None,
    ):
        self.meta_model = meta_model
        self.config = config if config is not None else AdaptationConfig()
        self.gamma = float(gamma)
        self.device = device if device is not None else next(meta_model.parameters()).device
        self.feature_dim = meta_model.feature_dim

        # 固化预训练骨干网络权重，自适应阶段只更新 z
        self.meta_model.eval()
        for p in self.meta_model.parameters():
            p.requires_grad = False

    def adapt_and_sample(
        self,
        x_obs: torch.Tensor,       # (N, D) in [0, 1]^D
        y_obs: torch.Tensor,       # (N,) 最小化目标 y = -Power
        gamma_current: Optional[float] = None,
        exploration_floor: float = 0.15,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """在线求解 z_MAP 与协方差，并返回 Thompson 采样向量 z_hat。
        
        返回:
            (z_hat, z_map, sigma_z)
        """
        N = len(x_obs)
        if N < 3:
            # 样本过少时退回高斯先验
            z_hat = torch.randn(self.feature_dim, device=self.device) * 0.5
            z_map = torch.zeros(self.feature_dim, device=self.device)
            return z_hat, z_map, torch.eye(self.feature_dim, device=self.device)

        gamma_use = gamma_current if gamma_current is not None else self.gamma
        y_np = y_obs.detach().cpu().numpy()
        tau_val = float(np.quantile(y_np, gamma_use))

        labels = (y_obs <= tau_val).float().to(self.device)
        weights = torch.clamp(tau_val - y_obs, min=0.0).to(self.device)
        w_mean = weights.mean()
        if w_mean > 1e-6:
            weights = weights / w_mean

        with torch.no_grad():
            phi = self.meta_model.extract_features(x_obs.to(self.device))  # (N, d)
            m = self.meta_model.mean_head(phi).squeeze(-1)                  # (N,)

        # 待优化的目标任务嵌入向量 z
        z = torch.zeros(self.feature_dim, device=self.device, requires_grad=True)
        opt = torch.optim.LBFGS(
            [z],
            lr=self.config.lr,
            max_iter=self.config.max_iter,
            tolerance_grad=self.config.tolerance_grad,
            tolerance_change=self.config.tolerance_change,
            history_size=self.config.history_size,
            line_search_fn=self.config.line_search_fn,
        )

        def closure():
            opt.zero_grad()
            interaction = torch.matmul(phi, z)
            logits = m + interaction
            bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            loss = torch.mean(weights * bce) + 0.5 * torch.sum(z**2)
            loss.backward()
            return loss

        try:
            opt.step(closure)
        except Exception:
            pass

        z_map = z.detach()

        # 计算拉普拉斯 Hessian 矩阵: H = sum(w_i * p_i * (1 - p_i) * phi_i * phi_i^T) + I
        with torch.no_grad():
            logits = m + torch.matmul(phi, z_map)
            probs = torch.sigmoid(logits)
            v = weights * probs * (1.0 - probs) + 1e-4  # (N,)
            # (d, N) * (N,) @ (N, d) -> (d, d)
            weighted_phi = phi.T * v.unsqueeze(0)
            H = torch.matmul(weighted_phi, phi) + torch.eye(self.feature_dim, device=self.device)

            # 协方差 Sigma = H^-1
            try:
                Sigma = torch.inverse(H)
            except Exception:
                Sigma = torch.eye(self.feature_dim, device=self.device) * 0.1

            # 加上探索方差地板，防止后验过快坍缩
            Sigma += exploration_floor * torch.eye(self.feature_dim, device=self.device)

            # Thompson 采样: z_hat ~ N(z_map, Sigma)
            L = torch.linalg.cholesky(Sigma)
            eps = torch.randn(self.feature_dim, device=self.device)
            z_hat = z_map + torch.matmul(L, eps)

        return z_hat, z_map, Sigma
