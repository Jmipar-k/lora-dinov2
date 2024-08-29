import os
import cv2
import numpy as np
import torch
from typing import Optional
import albumentations as A

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, root: str, split: str = "train", transform: Optional[transforms.Compose] = None):
        """
        Args:
            root (str): Root directory of the dataset, e.g., 'dataset_initial/'.
            split (str): Which split to use ('train', 'val', 'test').
            transform (Optional[transforms.Compose]): Transformations to apply to the images.
        """
        self.root = root
        self.split = split
        self.transform = transform

        # Get list of all images and corresponding labels
        self.images, self.labels = self._load_data()

    def _load_data(self):
        images = []
        labels = []

        # The root directory structure should be like 'dataset/train/0/', 'dataset/train/1/', etc.
        split_dir = os.path.join(self.root, self.split)
        for label in os.listdir(split_dir):
            label_dir = os.path.join(split_dir, label)
            if os.path.isdir(label_dir):
                for img_name in os.listdir(label_dir):
                    img_path = os.path.join(label_dir, img_name)
                    images.append(img_path)
                    labels.append(int(label))  # The directory name is the label (0, 1, 2, 3)

        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int):
        # Load image
        img_path = self.images[index]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Get corresponding label
        label = self.labels[index]

        # Apply transformations
        if self.transform is not None:
            image = self.transform(image=image)["image"]

        # Convert image to a tensor and normalize it
        image = torch.from_numpy(np.moveaxis(image, -1, 0)).float() / 255.0

        return image, label

# Example usage:
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     # transforms.Resize((224, 224)),  # Resize images if needed
#     transforms.ToTensor(),
#     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
# ])

def get_dataloader(
    dataset_name: str,
    root_dir: str = "/home/work/jmpark/DGU_Medical/dataset_initial",
    img_dim: tuple[int, int] = (490, 490),
    batch_size: int = 16,
    corruption_severity: int = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Get the dataloaders for the custom dataset.

    Args:
        root_dir (str, optional): The root directory of the dataset. Defaults to "./dataset_initial".
        img_dim (tuple[int, int], optional): The input size of the images. Defaults to (490, 490).
        batch_size (int, optional): The batch size of the dataloader. Defaults to 6.
        corruption_severity (int, optional): The corruption severity level between 1 and 5. Defaults to None.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: The train/validation/test loader respectively.
    """
    assert dataset_name in ["custom"], "dataset name not in [custom]"
    transform = A.Compose([A.Resize(height=img_dim[0], width=img_dim[1])])

    if dataset_name == "custom":
        train_dataset = CustomDataset(
            root=root_dir,
            split="train",
            transform=transform,
        )

        if corruption_severity is not None:
            transform = get_corruption_transforms(img_dim, corruption_severity)

        val_dataset = CustomDataset(
            root=root_dir,
            split="val",
            transform=transform,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=16,
        persistent_workers=True,
        shuffle=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=16,
        shuffle=False,
        pin_memory=True)
    
    return train_loader, val_loader

def get_dataloader_for_evaluation(
    dataset_name: str,
    root_dir: str = "/home/work/jmpark/DGU_Medical/dataset_initial",
    img_dim: tuple[int, int] = (490, 490),
    batch_size: int = 16,
    corruption_severity: int = None,
) -> DataLoader:
    """Get the dataloaders for the custom dataset.

    Args:
        root_dir (str, optional): The root directory of the dataset. Defaults to "./dataset_initial".
        img_dim (tuple[int, int], optional): The input size of the images. Defaults to (490, 490).
        batch_size (int, optional): The batch size of the dataloader. Defaults to 6.
        corruption_severity (int, optional): The corruption severity level between 1 and 5. Defaults to None.

    Returns:
       DataLoader: The test loader
    """
    assert dataset_name in ["custom"], "dataset name not in [custom]"
    transform = A.Compose([A.Resize(height=img_dim[0], width=img_dim[1])])

    if dataset_name == "custom":
        if corruption_severity is not None:
            transform = get_corruption_transforms(img_dim, corruption_severity)

        test_dataset = CustomDataset(
            root=root_dir,
            split="test",
            transform=transform,
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=16,
        shuffle=False,
        pin_memory=True
    )
    
    return test_loader



