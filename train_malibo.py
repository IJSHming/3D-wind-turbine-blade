"""MALIBO 神经网络元训练命令行入口脚本 (train_malibo.py):
严格遵循原论文 Pan et al. (ICML 2024, Appendix G) 全部默认结构与训练超参:
- ResFFN: 4 hidden layers x 64 units, ELU, feature_dim = 50;
- MeanHead: 50 units, ELU;
- Loss: Eq. (2) LFBO + 0.1 * Eq. (4) KS + Covariance Gaussian prior regularizer;
- Optimizer: Adam (lr=1e-3), ExponentialLR(0.999), batch_size=256, epochs=2048 with early stopping.
"""

import argparse
from pathlib import Path
import torch

from src.malibo.config import ResFFNConfig, TrainingConfig
from src.malibo.dataset import MALIBOBladeMetaDataset
from src.malibo.models import MALIBOMetaModel
from src.malibo.trainer import MALIBOTrainer


def main():
    parser = argparse.ArgumentParser(description="训练严格对齐原论文的 MALIBO 3D 叶片元优化神经网络")
    parser.add_argument(
        "--dataset",
        type=str,
        default="malibo_dataset/malibo_blade_train_dataset.pt",
        help="MALIBO 训练集 .pt 文件路径",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2048,
        help="最大训练轮数 (Appendix G 默认: 2048)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Adam 初始学习率 (Appendix G 默认: 1e-3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="批大小 (Appendix G 默认: 256)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="weights/malibo_checkpoint.pt",
        help="模型权重导出路径",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="控制台日志打印间隔 (轮数)",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"未找到训练集: {dataset_path}！请先生成网格数据集。")

    print(f"正在载入 MALIBO 数据集: {dataset_path} ...")
    dataset = MALIBOBladeMetaDataset(dataset_path)
    in_dim = dataset.tasks[0].x_unit.shape[1]
    print(f"载入成功！工况总数: {dataset.num_tasks}, 样本总数: {len(dataset)}, 输入特征维度: {in_dim}")

    # 严格对齐 Appendix G: 4 hidden layers x 64 units, feature_dim = 50, ELU
    resffn_cfg = ResFFNConfig(
        in_dim=in_dim,
        hidden_dim=64,
        num_hidden_layers=4,
        feature_dim=50,
    )
    meta_model = MALIBOMetaModel(num_tasks=dataset.num_tasks, config=resffn_cfg)
    n_params = sum(p.numel() for p in meta_model.parameters() if p.requires_grad)
    print(f"模型参数量 (含 task_embeddings): {n_params:,}")

    train_cfg = TrainingConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )

    trainer = MALIBOTrainer(
        model=meta_model,
        dataset=dataset,
        config=train_cfg,
    )

    res = trainer.train(
        epochs=args.epochs,
        save_path=args.output,
        verbose_every=args.log_interval,
    )

    with torch.no_grad():
        z_emb = meta_model.task_embeddings.detach().cpu()
        z_std = float(z_emb.std().item())
        z_mean_abs = float(z_emb.mean(dim=0).abs().mean().item())
    print(f"MALIBO 元训练完成！最佳验证损失: {res['best_val_loss']:.4f}, z_t 统计: std={z_std:.4f}, |mean|={z_mean_abs:.4f}")


if __name__ == "__main__":
    main()
