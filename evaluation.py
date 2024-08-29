# Code for evaluating on test dataset
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dino_finetune import (
    DINOV2EncoderLoRA,
    get_dataloader_for_evaluation,
)

def evaluate(
    dino_lora: nn.Module,
    test_loader: DataLoader,
    metrics: dict,
) -> None:
    test_acc = 0.0

    dino_lora.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.float().cuda()
            labels = labels.long().cuda()

            logits = dino_lora(images)

            predicted_classes = torch.argmax(logits, dim=1)
            correct_predictions = (predicted_classes == labels).sum().item()
            acc = correct_predictions / len(labels) * 100
            test_acc += acc

    metrics["test_acc"].append(test_acc / len(test_loader))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment Configuration")
    parser.add_argument(
        "--r",
        type=int,
        default=3,
        help="loRA rank parameter r",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="small",
        help="DINOv2 backbone parameter [small, base, large, giant]",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        default=True,
        help="Use Low-Rank Adaptation (LoRA) to finetune",
    )
    parser.add_argument(
        "--use_fpn",
        action="store_true",
        default=False,
        help="True -> Use the FPN decoder for finetuning, False -> Classification(Linear) Head(decoder) for fine-tuning",
    )
    parser.add_argument(
        "--img_dim",
        type=int,
        nargs=2,
        default=(490, 490),
        help="Images will be resized into dimensions (height width)",
    )
    parser.add_argument(
        "--lora_weights",
        type=str,
        default=None,
        help="Load the LoRA weights from file location",
    )
    # Training parameters
    parser.add_argument(
        "--dataset",
        type=str,
        default="custom",
        help="The dataset to finetune on, `custom` ",
    )
    parser.add_argument(
        "--custom_dataset_path",
        type=str,
        default="/home/work/jmpark/DGU_Medical/dataset_initial",
        help="The path to custom dataset to finetune on",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Finetuning batch size",
    )
    config = parser.parse_args()

    # All backbone sizes and configurations
    backbones = {
        "small": "vits14_reg",
        "base": "vitb14_reg",
        "large": "vitl14_reg",
        "giant": "vitg14_reg",
    }
    embedding_dims = {
        "small": 384,
        "base": 768,
        "large": 1024,
        "giant": 1536,
    }
    config.emb_dim = embedding_dims[config.size]

    # Dataset
    dataset_classes = {
        "voc": 21,
        "ade20k": 150,
        "custom": 4,
    }
    config.n_classes = dataset_classes[config.dataset]

    encoder = torch.hub.load(
        repo_or_dir="facebookresearch/dinov2",
        model=f"dinov2_{backbones[config.size]}",
    ).cuda()

    dino_lora = DINOV2EncoderLoRA(
        encoder=encoder,
        r=config.r,
        emb_dim=config.emb_dim,
        img_dim=config.img_dim,
        n_classes=config.n_classes,
        use_lora=config.use_lora,
        use_fpn=config.use_fpn,
    ).cuda()
    
    dino_lora.load_parameters(config.lora_weights)

    metrics = {
        "test_acc": [],
    }

    test_loader = get_dataloader_for_evaluation(
        config.dataset, img_dim=config.img_dim, batch_size=config.batch_size
    )

    print("Calculating Accuracy...")
    evaluate(dino_lora, test_loader, metrics)
    print("Done!")
    print(f'Test Accuracy : {metrics["test_acc"]}%')
