import json
import logging
import argparse
import tqdm
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from dino_finetune import (
    DINOV2EncoderLoRA,
    get_dataloader,
    visualize_overlay,
)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def print_trainable_parameters(model):
    trainable_params = 0
    all_params = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        all_params += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(
        f"Trainable params: {trainable_params} || All params: {all_params} || "
        f"Trainable%: {100 * trainable_params / all_params:.2f}%"
    )
    return trainable_params, all_params

def validate_epoch(
    dino_lora: nn.Module,
    val_loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    metrics: dict,
) -> None:
    val_loss = 0.0
    val_acc = 0.0

    dino_lora.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.float().cuda()
            labels = labels.long().cuda()

            logits = dino_lora(images)
            loss = criterion(logits, labels)
            val_loss += loss.item()

            predicted_classes = torch.argmax(logits, dim=1)
            correct_predictions = (predicted_classes == labels).sum().item()
            acc = correct_predictions / len(labels) * 100
            val_acc += acc

    metrics["val_loss"].append(val_loss / len(val_loader))
    metrics["val_acc"].append(val_acc / len(val_loader))


def finetune_dino(config: argparse.Namespace, encoder: nn.Module):
    dino_lora = DINOV2EncoderLoRA(
        encoder=encoder,
        r=config.r,
        emb_dim=config.emb_dim,
        img_dim=config.img_dim,
        n_classes=config.n_classes,
        use_lora=config.use_lora,
        use_fpn=config.use_fpn,
    ).cuda()

    if config.lora_weights:
        dino_lora.load_parameters(config.lora_weights)
    
    # Verify Trainable Parameters
    print('DINOv2-LoRA Trainable Parameters')
    print_trainable_parameters(dino_lora)
    
    train_loader, val_loader = get_dataloader(
        config.dataset, img_dim=config.img_dim, batch_size=config.batch_size
    )

    # Finetuning for Classification
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.AdamW(dino_lora.parameters(), lr=config.lr)
    if config.use_amp:
        scaler = GradScaler()

    # Log training and validation metrics
    metrics = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(config.epochs):
        dino_lora.train()

        epoch_loss = 0
        epoch_correct = 0
        total_samples = 0

        # Progress bar for batches
        with tqdm.tqdm(total=len(train_loader), desc=f"Epoch {epoch}", unit='batch') as pbar:
            for images, labels in train_loader:
                images = images.float().cuda()
                labels = labels.long().cuda()
                optimizer.zero_grad()
                
                # fp16 -> mixed precision training
                if config.use_amp:
                    with autocast():
                        logits = dino_lora(images)
                        loss = criterion(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                # fp32 -> original training
                else:
                    logits = dino_lora(images)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()

                epoch_loss += loss.item()
                _, predicted_classes = torch.max(logits, 1)
                epoch_correct += (predicted_classes == labels).sum().item()
                total_samples += labels.size(0)

                # Update tqdm progress bar
                pbar.set_postfix(loss=loss.item(), accuracy=epoch_correct / total_samples)
                pbar.update(1)

        # Calculate and log epoch metrics
        avg_loss = epoch_loss / len(train_loader)
        accuracy = epoch_correct / total_samples

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_loss,
            "train_accuracy": accuracy
        })

        # Validate and save model at certain epochs
        if epoch % 5 == 0:
            predicted_classes = torch.argmax(logits, dim=1)
            validate_epoch(dino_lora, val_loader, criterion, metrics)
            dino_lora.save_parameters(f"output/{config.exp_name}_{epoch}.pt")

            if config.debug:
                # Visualize some of the batch and write to files when debugging
                visualize_overlay(
                    images, predicted_classes, config.n_classes, filename=f"viz_{epoch}"
                )

            # Log validation metrics to wandb
            wandb.log({
                "val_acc": metrics['val_acc'][-1],
                "val_loss": metrics['val_loss'][-1]
            })

            logging.info(
                f"Epoch: {epoch} - val acc: {metrics['val_acc'][-1]} "
                f"- val loss {metrics['val_loss'][-1]}"
            )

    # Final save and logging
    dino_lora.save_parameters(f"output/{config.exp_name}_last.pt")
    with open(f"output/{config.exp_name}_metrics.json", "w") as f:
        json.dump(metrics, f)

    # Log final metrics to wandb
    wandb.log({
        "final_train_loss": avg_loss,
        "final_train_accuracy": accuracy,
        "final_val_acc": metrics['val_acc'][-1],
        "final_val_loss": metrics['val_loss'][-1]
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment Configuration")
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument(
        "--exp_name",
        type=str,
        default="lora",
        help="Experiment name, will be used for wandb and checkpoint file logging too.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug by visualizing some of the outputs to file for a sanity check",
    )
    parser.add_argument(
        "--r",
        type=int,
        default=4,
        help="loRA rank parameter r",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="base",
        help="DINOv2 backbone parameter [small, base, large, giant]",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Use Low-Rank Adaptation (LoRA) to finetune",
    )
    parser.add_argument(
        "--use_fpn",
        action="store_true",
        default=False,
        help="True -> Use the FPN decoder for finetuning, False -> Classification(Linear) Head(decoder) for fine-tuning",
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="Use Mixed Precision Training (fp16), Default is fp32",
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
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Finetuning batch size",
    )
    config = parser.parse_args()
    
    # Set Manual Seed
    set_random_seed(config.seed)
    
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
        # "voc": 21,
        # "ade20k": 150,
        "custom": 4, # Write the number of classes of your custom dataset
    }
    config.n_classes = dataset_classes[config.dataset]

    encoder = torch.hub.load(
        repo_or_dir="facebookresearch/dinov2",
        model=f"dinov2_{backbones[config.size]}",
    ).cuda()

    wandb.init(project="dino_v2 FT", config=config)
    wandb.run.name = config.exp_name
    finetune_dino(config, encoder)
    wandb.finish()
