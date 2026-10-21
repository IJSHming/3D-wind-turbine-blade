"""MALIBO 神经网络元训练命令行入口脚本 (train_malibo.py):
基于生成的 16 工况网格数据集 (malibo_dataset/malibo_blade_train_dataset.pt)，
训练 60 维叶片特征映射网络 ResFFN、任务无关均值网络 MeanHead 与多工况任务嵌入集合 {z_t}。
模型权重默认保存至 weights/malibo_checkpoint.pt。
"""

import argparse
from pathlib import Path
import sys
import torch

from src.malibo.config import ResFFNConfig, TrainingConfig
from src.malibo.dataset import MALIBOBladeMetaDataset
from src.malibo.models import MALIBOMetaModel
from src.malibo.trainer import MALIBOTrainer


def main():
    parser = argparse.ArgumentParser(description="训练 MALIBO 3D 叶片元优化神经网络")
    parser.add_argument(
        "--dataset",
        type=str,
        default="malibo_dataset/malibo_blade_train_dataset.pt",
        help="MALIBO 训练集 .pt 文件路径",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=600,
        help="最大训练轮数 (默认: 600)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Adam 初始学习率 (默认: 1e-3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="批大小 (默认: 256)",
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
    print(f"载入成功！工况总数: {dataset.num_tasks}, 样本总数: {len(dataset)}")

    resffn_cfg = ResFFNConfig(in_dim=120, feature_dim=64)
    meta_model = MALIBOMetaModel(num_tasks=dataset.num_tasks, config=resffn_cfg)

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

    print(f"MALIBO 元训练完成！最佳验证损失: {res['best_val_loss']:.4f}")


if __name__ == "__main__":
    main()
