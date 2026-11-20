"""MALIBO 神经网络模型架构实现 (strictly aligned with Pan et al., ICML 2024 Figure 2 & Appendix G):
- ResFFN: 4 个残差全连接块，每层 64 单元，ELU 激活，无 LayerNorm，映射 D 维输入至 d=50 维特征空间 phi(x)；
- MeanHead: 任务无关先验均值预测网络 m(Phi): Linear(50, 50) -> ELU -> Linear(50, 1)；
- Task Embedding: 各工况任务特异性嵌入向量集合 {z_t} in R^50，遵循标准正态先验 p(Z) = N(0, I)；
- 联合输出分类 Logit: a = m(phi(x)) + z_t^T phi(x)。
"""

from typing import Optional, Union
import torch
import torch.nn as nn

from src.malibo.config import ResFFNConfig


class ResBlock(nn.Module):
    """标准残差全连接块 (ELU 激活)"""

    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.fc1(x))
        out = self.fc2(out)
        return self.act(out + residual)


class ResFFN(nn.Module):
    """残差前馈特征映射网络 phi(x): R^D -> R^50 (Appendix G: 4 hidden layers x 64 units, ELU)"""

    def __init__(self, config: Optional[ResFFNConfig] = None):
        super().__init__()
        if config is None:
            config = ResFFNConfig()

        self.in_dim = config.in_dim
        self.hidden_dim = config.hidden_dim
        self.feature_dim = config.feature_dim

        self.input_proj = nn.Linear(self.in_dim, self.hidden_dim)
        self.act = nn.ELU()
        self.res_blocks = nn.ModuleList(
            [ResBlock(self.hidden_dim) for _ in range(config.num_hidden_layers)]
        )
        self.output_proj = nn.Linear(self.hidden_dim, self.feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.input_proj(x))
        for block in self.res_blocks:
            h = block(h)
        return self.output_proj(h)


class MeanHead(nn.Module):
    """任务无关均值预测网络 m(Phi): R^50 -> R (Appendix G: 50 hidden units, ELU)"""

    def __init__(self, feature_dim: int = 50, hidden_dim: int = 50):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.act = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(phi))
        return self.fc2(h)


class MALIBOMetaModel(nn.Module):
    """MALIBO 完整多任务元学习模型 (Figure 2)"""

    def __init__(
        self,
        num_tasks: int,
        config: Optional[ResFFNConfig] = None,
    ):
        super().__init__()
        if config is None:
            config = ResFFNConfig()

        self.num_tasks = num_tasks
        self.feature_dim = config.feature_dim
        self.phi_net = ResFFN(config)
        # Appendix G: mean prediction layer m(.) uses a fully connected layer with 50 units
        self.mean_head = MeanHead(feature_dim=self.feature_dim, hidden_dim=self.feature_dim)

        # 任务嵌入矩阵 (T, d)，通过 Eq. (4) KS + Cov 正则项匹配 p(Z) = N(0, I_d)
        self.task_embeddings = nn.Parameter(
            torch.randn(num_tasks, self.feature_dim) * 0.1
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """输入设计参数 x -> 输出潜在特征 phi(x)"""
        return self.phi_net(x)

    def predict_mean(self, x: torch.Tensor) -> torch.Tensor:
        """预测任务无关先验效用 Logit m(phi(x))"""
        phi = self.extract_features(x)
        return self.mean_head(phi)

    def predict_proba(
        self,
        x: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """计算分类后验概率 p(k=1|x) = sigmoid(m(phi(x)) + z^T phi(x))"""
        logits = self.forward(x, z=z)
        return torch.sigmoid(logits)

    def get_task_embedding(self, task_idx: int) -> torch.Tensor:
        return self.task_embeddings[task_idx]

    def forward(
        self,
        x: torch.Tensor,
        task_idx: Optional[Union[int, torch.Tensor]] = None,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """计算分类 Logit a = m(phi(x)) + z^T phi(x)"""
        phi = self.extract_features(x)
        m = self.mean_head(phi).squeeze(-1)

        if z is not None:
            if z.dim() == 1:
                interaction = torch.matmul(phi, z)
            else:
                interaction = torch.sum(phi * z, dim=-1)
            return m + interaction

        if task_idx is not None:
            if isinstance(task_idx, int):
                z_t = self.task_embeddings[task_idx]
                interaction = torch.matmul(phi, z_t)
            else:
                z_t = self.task_embeddings[task_idx]
                interaction = torch.sum(phi * z_t, dim=-1)
            return m + interaction

        return m
