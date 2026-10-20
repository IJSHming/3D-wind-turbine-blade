"""MALIBO 多任务元学习数据集封装：
基于 Pan et al. (2024, Eq. 2 - 4) 设计：
- 接收打包的 16 工况任务数据；
- 依据 gamma 分位数 (默认 1/3) 自动计算各工况阈值 tau 与权重 w = max(tau - y, 0)；
- 支持 Meta-Batch 批量采样与任务索引解耦。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class MALIBOTaskTensor:
    """单一工况任务张量封装"""
    task_id: int
    task_name: str
    wind_speed: float
    rpm: float
    x_unit: torch.Tensor          # (N, 60)
    theta: torch.Tensor           # (N, 60)
    y_penalized: torch.Tensor     # (N,)
    y_minimize: torch.Tensor      # (N,)
    tau: float                    # gamma 分位数阈值
    labels: torch.Tensor          # (N,) 1(y <= tau)
    weights: torch.Tensor         # (N,) max(tau - y, 0)
    num_samples: int


class MALIBOBladeMetaDataset(Dataset):
    """MALIBO 3D 叶片多任务元学习数据集"""

    def __init__(
        self,
        pt_path: Union[str, Path],
        gamma: float = 1.0 / 3.0,
        device: torch.device = torch.device("cpu"),
    ):
        self.pt_path = Path(pt_path)
        self.gamma = float(gamma)
        self.device = device
        self.tasks: List[MALIBOTaskTensor] = []

        self._flat_x: Optional[torch.Tensor] = None
        self._flat_task_idx: Optional[torch.Tensor] = None
        self._flat_labels: Optional[torch.Tensor] = None
        self._flat_weights: Optional[torch.Tensor] = None

        self._load_and_process()

    def _load_and_process(self):
        data = torch.load(self.pt_path, map_location="cpu", weights_only=False)
        raw_tasks = data["tasks"]

        all_x = []
        all_task_ids = []
        all_labels = []
        all_weights = []

        for t in raw_tasks:
            t_id = int(t["task_id"])
            x_u = t["X_unit"].float()
            y_min = t["Y_minimize"].float()
            y_pen = t["Y_penalized"].float()
            th = t["Theta"].float()

            # gamma 分位数 (y_min 越小越好)
            tau_val = float(np.quantile(y_min.numpy(), self.gamma))
            labels = (y_min <= tau_val).float()
            weights = torch.clamp(tau_val - y_min, min=0.0)

            # 归一化权重以稳定训练
            w_mean = weights.mean()
            if w_mean > 1e-6:
                weights = weights / w_mean

            task_obj = MALIBOTaskTensor(
                task_id=t_id,
                task_name=t["task_name"],
                wind_speed=float(t["wind_speed"]),
                rpm=float(t["rpm"]),
                x_unit=x_u,
                theta=th,
                y_penalized=y_pen,
                y_minimize=y_min,
                tau=tau_val,
                labels=labels,
                weights=weights,
                num_samples=len(y_min),
            )
            self.tasks.append(task_obj)

            all_x.append(x_u)
            all_task_ids.append(torch.full((len(x_u),), t_id, dtype=torch.long))
            all_labels.append(labels)
            all_weights.append(weights)

        self._flat_x = torch.cat(all_x, dim=0)
        self._flat_task_idx = torch.cat(all_task_ids, dim=0)
        self._flat_labels = torch.cat(all_labels, dim=0)
        self._flat_weights = torch.cat(all_weights, dim=0)

    def __len__(self) -> int:
        return len(self._flat_x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self._flat_x[idx],
            self._flat_task_idx[idx],
            self._flat_labels[idx],
            self._flat_weights[idx],
        )

    @property
    def num_tasks(self) -> int:
        return len(self.tasks)
