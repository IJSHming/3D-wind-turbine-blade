"""MALIBO 在线目标任务拉普拉斯自适应器 (src/malibo/adaptation.py):
严格对齐原论文 Pan et al. (ICML 2024, Section 4, Eq. 6, Eq. 7, Eq. 9 & Appendix G):
1. Eq. (6): 固定元学习骨干 g_omega = (phi, m), 用 L-BFGS (lr=1, max_iter=20, tol_grad=1e-7,
   tol_change=1e-9, history_size=100, strong_wolfe) 最小化:
   L_MALIBO(z) = 0.5 * z^T z + sum_{n=1}^N [ w_n * softplus(-a_n) + softplus(a_n) ], w_n = max(tau - y_n, 0);
2. Eq. (7): 计算精确负 Hessian 精度矩阵:
   Sigma_N^{-1} = I_d + sum_{n=1}^N (w_n + 1) * k_hat_n * (1 - k_hat_n) * Phi_n * Phi_n^T;
3. Eq. (9): 执行纯净拉普拉斯 Thompson 采样 z_hat ~ N(z_MAP, Sigma_N) (固定 gamma = 1/3, 无人工方差地板)。
"""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from src.malibo.config import AdaptationConfig
from src.malibo.models import MALIBOMetaModel


class LaplaceTaskAdapter:
    """拉普拉斯在线任务自适应与汤普森采样器 (Strictly Paper-Faithful Eq. 6-9)"""

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

        # 固化预训练骨干网络权重，自适应阶段仅优化目标任务嵌入 z
        self.meta_model.eval()
        for p in self.meta_model.parameters():
            p.requires_grad = False

    def adapt_and_sample(
        self,
        x_obs: torch.Tensor,          # (N, D) in [0, 1]^D
        y_obs_std: torch.Tensor,      # (N,) 标准化最小化目标 y_std
        tau_std: float,               # gamma=1/3 分位数阈值 (标准化尺度)
        rng: Optional[np.random.RandomState] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """在线求解 Eq. (6) z_MAP、Eq. (7) 精确精度矩阵与 Eq. (9) Thompson 采样 z_hat.

        返回:
            (z_hat, z_map, L_precision)
        """
        d = self.feature_dim
        if len(x_obs) == 0:
            eps_np = rng.randn(d) if rng is not None else np.random.randn(d)
            z_hat = torch.as_tensor(eps_np, dtype=torch.float32, device=self.device)
            z_map = torch.zeros(d, dtype=torch.float32, device=self.device)
            return z_hat, z_map, torch.eye(d, dtype=torch.float32, device=self.device)

        x_obs = x_obs.to(self.device)
        y_obs_std = y_obs_std.to(self.device)
        w = torch.clamp(float(tau_std) - y_obs_std, min=0.0)

        with torch.no_grad():
            phi = self.meta_model.extract_features(x_obs)            # (N, d)
            m = self.meta_model.mean_head(phi).squeeze(-1)           # (N,)

        # Eq. (6): L-BFGS 求解 z_MAP (Appendix G 超参数)
        z = torch.zeros(d, dtype=torch.float32, device=self.device, requires_grad=True)
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
            a = m + torch.matmul(phi, z)
            nll = torch.sum(w * F.softplus(-a) + F.softplus(a))
            prior = 0.5 * torch.sum(z ** 2)
            loss = prior + nll
            loss.backward()
            return loss

        opt.step(closure)
        z_map = z.detach()

        # Eq. (7): Sigma_N^{-1} = I_d + sum_n (w_n + 1) * k_hat_n * (1 - k_hat_n) * phi_n phi_n^T
        with torch.no_grad():
            a_map = m + torch.matmul(phi, z_map)
            k_hat = torch.sigmoid(a_map)
            weights_laplace = (w + 1.0) * k_hat * (1.0 - k_hat)  # (N,)

            weighted_phi = phi * weights_laplace.unsqueeze(-1)
            prec = torch.eye(d, dtype=torch.float32, device=self.device) + (weighted_phi.T @ phi)

            prec_64 = prec.to(torch.float64)
            prec_64 = 0.5 * (prec_64 + prec_64.T)
            diag_scale = torch.clamp(torch.diag(prec_64), min=1.0)
            jitter = torch.diag(diag_scale * 1e-6)

            try:
                L_64 = torch.linalg.cholesky(prec_64 + jitter)
            except torch.linalg.LinAlgError:
                eigvals, eigvecs = torch.linalg.eigh(prec_64)
                eigvals = torch.clamp(eigvals, min=1e-5)
                prec_psd = eigvecs @ torch.diag(eigvals) @ eigvecs.T
                L_64 = torch.linalg.cholesky(prec_psd)

            L_prec = L_64.to(torch.float32)

            # Eq. (9): z_hat ~ N(z_MAP, Sigma_N), 其中 L @ L^T = Sigma_N^{-1} => delta = L^{-T} eps
            eps_np = rng.randn(d, 1) if rng is not None else np.random.randn(d, 1)
            eps = torch.as_tensor(eps_np, dtype=torch.float32, device=self.device)
            delta = torch.linalg.solve_triangular(L_prec.T, eps, upper=True).squeeze(-1)
            z_hat = z_map + delta

        return z_hat, z_map, L_prec
