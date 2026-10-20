"""MALIBO 多任务元学习训练器：
严格执行 Pan et al. (2024, Eq. 3 & Eq. 4) 损失优化：
L(omega, {z_t}) = - sum w_i [ k_i log sigma(a_i) + (1 - k_i) log(1 - sigma(a_i)) ] + 0.5 * lambda * sum ||z_t||^2.
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from src.malibo.config import TrainingConfig
from src.malibo.dataset import MALIBOBladeMetaDataset
from src.malibo.models import MALIBOMetaModel


class MALIBOTrainer:
    """MALIBO 元模型训练器"""

    def __init__(
        self,
        model: MALIBOMetaModel,
        dataset: MALIBOBladeMetaDataset,
        config: Optional[TrainingConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config if config is not None else TrainingConfig()
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() and self.config.device == "cuda" else "cpu"
            )
        else:
            self.device = device

        self.model = model.to(self.device)
        self.dataset = dataset

        # 80/20 划分训练集与早停验证集
        total_len = len(self.dataset)
        val_len = int(total_len * 0.20)
        train_len = total_len - val_len

        generator = torch.Generator().manual_seed(self.config.seed)
        self.train_ds, self.val_ds = random_split(
            self.dataset, [train_len, val_len], generator=generator
        )

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=self.config.lr_decay_factor
        )

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """带效用加权的二分类交叉熵损失 + 任务嵌入高斯先验正则化"""
        # 加权二元交叉熵
        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        weighted_bce = torch.mean(weights * bce)

        # 任务嵌入的高斯先验正则项: 0.5 * lambda * sum(||z_t||^2)
        z_norm_sq = torch.sum(self.model.task_embeddings**2)
        reg = 0.5 * self.config.reg_lambda * z_norm_sq

        return weighted_bce + reg

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for x, task_idx, labels, weights in self.train_loader:
            x = x.to(self.device)
            task_idx = task_idx.to(self.device)
            labels = labels.to(self.device)
            weights = weights.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(x, task_idx=task_idx)
            loss = self.compute_loss(logits, labels, weights)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        self.scheduler.step()
        return total_loss / max(1, n_batches)

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for x, task_idx, labels, weights in self.val_loader:
            x = x.to(self.device)
            task_idx = task_idx.to(self.device)
            labels = labels.to(self.device)
            weights = weights.to(self.device)

            logits = self.model(x, task_idx=task_idx)
            loss = self.compute_loss(logits, labels, weights)
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(1, n_batches)

    def train(
        self,
        epochs: Optional[int] = None,
        save_path: Optional[str] = "weights/malibo_checkpoint.pt",
        verbose_every: int = 50,
    ) -> Dict[str, Any]:
        """执行完整训练循环"""
        max_epochs = epochs if epochs is not None else self.config.epochs
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None
        t0 = time.time()

        print(f"=== 开始 MALIBO 元训练 (Device: {self.device}, Epochs: {max_epochs}) ===")

        for ep in range(1, max_epochs + 1):
            train_loss = self.train_epoch()
            val_loss = self.evaluate()

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            if ep % verbose_every == 0 or ep == 1 or patience_counter >= self.config.early_stopping_patience:
                lr_curr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"  [Epoch {ep:4d}/{max_epochs}] Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} (Best: {best_val_loss:.4f}) | LR: {lr_curr:.6f}",
                    flush=True,
                )

            if patience_counter >= self.config.early_stopping_patience:
                print(f"  --> 触发早停机制 (Patience={self.config.early_stopping_patience})，提前终止。")
                break

        elapsed = time.time() - t0
        print(f"=== 训练完成，最佳 Val Loss: {best_val_loss:.4f}，耗时: {elapsed:.2f}s ===")

        if best_state is not None:
            self.model.load_state_dict(best_state)

        if save_path:
            save_file = Path(save_path)
            save_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "num_tasks": self.model.num_tasks,
                    "feature_dim": self.model.feature_dim,
                    "best_val_loss": best_val_loss,
                    "in_dim": self.dataset.tasks[0].x_unit.shape[-1],
                },
                save_file,
            )
            print(f"模型权重已保存至: {save_file}")

        return {
            "best_val_loss": best_val_loss,
            "elapsed_seconds": elapsed,
        }
