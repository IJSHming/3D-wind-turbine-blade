"""MALIBO 多任务元学习数据集封装 (src/malibo/dataset.py):
严格对齐原论文 Pan et al. (ICML 2024, Eq. 2 - Eq. 4):
- 对每个离线任务，截断极端不可行离群惩罚 (y_min <= 1.0) 后执行任务内零均值单位方差标准化 (保持 EI 与分位数严格不变性)，
  消除不同风速工况 (4.5 ~ 9.5 m/s) 间功率绝对量级差异；
- 依据固定分位数 gamma = 1/3 计算阈值 tau = quantile(y_std, 1/3) 与精准 LFBO 效用权重 w = max(tau - y_std, 0)。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union
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
    x_unit: torch.Tensor          # (N, D)
    theta: torch.Tensor           # (N, D)
    y_penalized: torch.Tensor     # (N,)
    y_minimize: torch.Tensor      # (N,)
    tau: float                    # gamma 分位数阈值 (标准化尺度)
    labels: torch.Tensor          # (N,) 1(y_std <= tau)
    weights: torch.Tensor         # (N,) max(tau - y_std, 0) (Eq. 2)
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

            # 截断极端不可行惩罚离群值 (score <= -0.25 -> y_min <= +1.0)，再按任务内标准化对齐跨风速尺度
            y_clean = torch.clamp(y_min, max=1.0)
            y_mean = float(y_clean.mean().item())
            y_std_val = float(y_clean.std().item()) if len(y_clean) > 1 else 1.0
            y_std_val = max(y_std_val, 1e-4)
            y_norm = (y_clean - y_mean) / y_std_val

            # 严格按 Eq. (2): tau = Phi^-1(gamma), k = 1(y <= tau), w = max(tau - y, 0)
            tau_val = float(np.quantile(y_norm.numpy(), self.gamma))
            labels = (y_norm <= tau_val).float()
            weights = torch.clamp(tau_val - y_norm, min=0.0)

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
