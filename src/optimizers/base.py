"""3D 风力机叶片黑盒优化器基类定义：
统一管理 300 次 NFE 评估预算、物理与气动约束校验、归一化搜索空间映射与收敛轨迹跟踪。
"""

from abc import ABC, abstractmethod
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.blade_geometry import get_baseline_blade_params, get_blade_bounds
from src.evaluator import WindTurbineBladeEvaluator


class BaseBladeOptimizer(ABC):
    """3D 风机叶片黑盒优化算法的基础抽象类。"""

    def __init__(
        self,
        evaluator: WindTurbineBladeEvaluator,
        x0: Optional[np.ndarray] = None,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        max_evals: int = 300,
        seed: int = 42,
    ):
        self.evaluator = evaluator
        self.x0 = get_baseline_blade_params() if x0 is None else np.asarray(x0, dtype=np.float64)
        if bounds is None:
            self.lower_bounds, self.upper_bounds = get_blade_bounds(self.x0)
        else:
            self.lower_bounds, self.upper_bounds = bounds

        self.dim = len(self.x0)
        self.dim_range = self.upper_bounds - self.lower_bounds

        self.max_evals = int(max_evals)
        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)

        # 优化进程指标
        self.eval_count = 0
        self.best_x = self.x0.copy()
        self.best_score = -float("inf")
        self.initial_score = 0.0

        # 收敛历史
        self.history_best: List[float] = []
        self.history_evaluated_scores: List[float] = []
        self.history_feasible: List[bool] = []
        self.stagnation_counter = 0

    def to_unit(self, x: np.ndarray) -> np.ndarray:
        """从真实物理尺度映射至 [0, 1]^D 归一化空间"""
        return np.clip((x - self.lower_bounds) / (self.dim_range + 1e-12), 0.0, 1.0)

    def from_unit(self, u: np.ndarray) -> np.ndarray:
        """从 [0, 1]^D 归一化空间还原为真实物理尺度"""
        return np.clip(self.lower_bounds + u * self.dim_range, self.lower_bounds, self.upper_bounds)

    def evaluate(self, x: np.ndarray) -> float:
        """执行单次评估，受 max_evals 严格保护"""
        if self.eval_count >= self.max_evals:
            return self.best_score

        x_clipped = np.clip(x, self.lower_bounds, self.upper_bounds)
        score = self.evaluator.evaluate(x_clipped)
        self.eval_count += 1

        is_feasible = (score > 0.0)
        self.history_feasible.append(is_feasible)
        self.history_evaluated_scores.append(score)

        if is_feasible and score > self.best_score:
            self.best_score = score
            self.best_x = x_clipped.copy()
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1

        self.history_best.append(self.best_score)
        return score

    def evaluate_initial(self) -> float:
        """评估标准基准叶片构型 x0"""
        score = self.evaluate(self.x0)
        self.initial_score = score
        self.best_score = score
        self.best_x = self.x0.copy()
        return score

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """运行完整优化流程，返回统计与轨迹结果"""
        pass
