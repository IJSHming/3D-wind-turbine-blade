"""MALIBO 多任务元学习训练器 (src/malibo/trainer.py):
严格对齐原论文 Pan et al. (ICML 2024, Eq. 2, Eq. 3, Eq. 4 & Appendix C / G):
- 分类效用损失: LFBOLoss = E[ max(tau - y, 0) * softplus(-a) + softplus(a) ] (Eq. 2);
- 潜在空间正则: R({z_t}; N(0, I)) = lambda_KS * L_KS + lambda_Cov * L_Cov (Eq. 4 & Eq. 27-29),
  总目标 L_meta = L_LFBO + lambda * R (lambda = 0.1, Eq. 3);
- 优化器与调度: Adam (lr=1e-3, 无 weight_decay), ExponentialLR (gamma=0.999/epoch),
  batch_size=256, 最多 2,048 epochs 带验证集早停。
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional
import torch
from torch.utils.data import DataLoader, random_split

from src.malibo.config import TrainingConfig
from src.malibo.dataset import MALIBOBladeMetaDataset
from src.malibo.losses import LFBOLoss, LatentSpaceRegularizer
from src.malibo.models import MALIBOMetaModel


class MALIBOTrainer:
    """MALIBO 元模型训练器 (严格对齐原论文 Eq. 2-4 & Appendix G)"""

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

        # Eq. (2) LFBO 损失与 Eq. (4) KS + Cov 高斯先验正则化器
        self.lfbo_loss = LFBOLoss()
        self.latent_reg = LatentSpaceRegularizer(feature_dim=self.model.feature_dim).to(self.device)
        self.latent_reg.calibrate_coefficients(self.model.num_tasks, self.device, num_samples=200)

        # Appendix G: Adam (lr=1e-3, 无 weight_decay) + ExponentialLR(0.999)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=self.config.lr_decay_factor
        )

    def compute_loss(
        self,
        logits: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Eq. (3): L_meta = L_LFBO + lambda * R({z_t}; N(0, I))."""
        lfbo = self.lfbo_loss(logits, weights)
        reg_total, _, _ = self.latent_reg(self.model.task_embeddings)
        return lfbo + self.config.reg_lambda * reg_total

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for x, task_idx, _labels, weights in self.train_loader:
            x = x.to(self.device)
            task_idx = task_idx.to(self.device)
            weights = weights.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(x, task_idx=task_idx)
            loss = self.compute_loss(logits, weights)
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

        for x, task_idx, _labels, weights in self.val_loader:
            x = x.to(self.device)
            task_idx = task_idx.to(self.device)
            weights = weights.to(self.device)

            logits = self.model(x, task_idx=task_idx)
            loss = self.compute_loss(logits, weights)
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(1, n_batches)

    def train(
        self,
        epochs: Optional[int] = None,
        save_path: Optional[str] = "weights/malibo_checkpoint.pt",
        verbose_every: int = 50,
    ) -> Dict[str, Any]:
        """执行完整元训练循环 (Appendix G)"""
        max_epochs = epochs if epochs is not None else self.config.epochs
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None
        t0 = time.time()

        print(
            f"=== 开始 MALIBO 元训练 (Device: {self.device}, Epochs: {max_epochs}, "
            f"lambda_KS={self.latent_reg.lambda_ks:.4f}, lambda_Cov={self.latent_reg.lambda_cov:.4f}) ==="
        )

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
                with torch.no_grad():
                    reg_tot, ks_l, cov_l = self.latent_reg(self.model.task_embeddings)
                print(
                    f"  [Epoch {ep:4d}/{max_epochs}] Train: {train_loss:.4f} | "
                    f"Val: {val_loss:.4f} (Best: {best_val_loss:.4f}) | "
                    f"R={reg_tot.item():.4f} (KS={ks_l.item():.4f}, Cov={cov_l.item():.4f}) | LR: {lr_curr:.6f}",
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
                    "hidden_dim": self.model.phi_net.hidden_dim,
                    "num_hidden_layers": len(self.model.phi_net.res_blocks),
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
