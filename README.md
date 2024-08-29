# lora-dinov2
A repo for fine-tuning DINOv2 using LoRA layers for multi-class classification

## Setup Conda Environment
I recommend using conda for setting up environments!

```bash
# Clone this repository into current path
git clone https://github.com/Jmipar-k/lora-dinov2.git

# Getting into the cloned repo
cd lora-dinov2

# Creating conda environment and installing necessary packages
conda env create -f environment.yml

# Activating created environment!
conda activate lora-dinov2
```

## Preparing Custom Dataset
The dataset should have a root folder, split folders, class folders, images.

The root folder("Your_Dataset") should have train, val, test(optional for training) respectively.

Each split(train, val, test) folder should have subdirectories named by index numbers of each class included in your custom dataset.

For example, my data folder looks like this.

```
CBNU_Medical/
├── train/
│   ├── 0/
│   │   └── image1.png
│   │   └── image2.png
│   │   └── image3.png
│   │   └── ...
│   ├── 1/
│   ├── 2/
│   ├── 3/
├── val/
│   ├── 0/
│   │   └── image1.png
│   │   └── image2.png
│   │   └── image3.png
│   │   └── ...
│   ├── 1/
│   ├── 2/
│   ├── 3/
└── test/
    ├── 0/
    │   └── image1.png
    │   └── image2.png
    │   └── image3.png
    │   └── ...
    ├── 1/
    ├── 2/
    ├── 3/
```

My root folder is /CBNU_Medical,

The splits are train/val/test and they have 4 sub directories (My Custom Dataset includes 4 classes)

The name of each directory will be the class label (0, 1, 2, 3).

They are composed of .png images.

train, val sets will be used for training, test will be used for evaluation.

## Training
### Hyperparameters
**--exp_name** : Name of experiment, this name will be used for trained weight saving or wandb logging(run name)

**--dataset** : Do not change this from "custom"

**--custom_dataset_path** : "path to the root directory of your custom dataset"

**--use_lora** : For using lora layers for fine-tuning (Storngly recommended to keep it on)

**--size** : Model size of the ViT encoder you would like to use (small, base, large, giant)

**--img_dim** : The height and weight(spatial size) input(training) images will be resized into (before getting pachified) **!MUST BE DIVISIBLE BY 14!**

**--epochs** : Number of epochs to be trained

**--batch_size** : Training batch size

**--lr** : learning rate

**--r** : Number of rank for LoRA layers

**--use_amp** : Turns on mixed precision training (amp, fp16)

### Training Commands

**Example of how to run fine-tuning on custom dataset (fp32)**

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --exp_name base_1e-4_50epochs_r6 --dataset custom --custom_dataset_path "path to the root directory of your custom dataset" --use_lora --size base --img_dim 490 490 --epochs 50 --batch_size 16 --lr 1e-4 -r 3
```

**Example of how to run fine-tuning on custom dataset with mixed precision training (amp, fp16)**

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --exp_name base_1e-4_50epochs_r6 --dataset custom --custom_dataset_path "path to the root directory of your custom dataset" --use_lora --size base --img_dim 490 490 --epochs 50 --batch_size 16 --lr 1e-4 -r 3 --use_amp
```

## Evaluation

**Example of how to run evaluation on custom dataset's test split**

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --dataset custom --custom_dataset_path "path to the root directory of your custom dataset" --lora_weights "path to the trained lora weights"--use_lora --size base --img_dim 490 490 --epochs 50 --batch_size 16 -r 3
```

## References
This code was heavily based on https://github.com/RobvanGastel/dinov2-finetune .
