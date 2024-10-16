import os
import cv2
import numpy as np
import torch
from typing import Dict, Optional
import albumentations as A

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, root: str, split: str = "train", class_name: str = None, transform: Optional[transforms.Compose] = None):
        """
        Args:
            root (str): Root directory of the dataset, e.g., 'dataset_initial/'.
            split (str): Which split to use ('train', 'val', 'test').
            class_name (str, optional): Name of the specific class to load.
            transform (Optional[transforms.Compose]): Transformations to apply to the images.
        """
        self.root = root
        self.split = split
        self.class_name = class_name
        self.transform = transform

        # Get list of images and corresponding labels for a specific class
        self.images, self.labels = self._load_data()

    def _load_data(self):
        images = []
        labels = []

        # The root directory structure should be like 'dataset/train/0/', 'dataset/train/1/', etc.
        split_dir = os.path.join(self.root, self.split)
        if self.class_name is not None:
            label_dir = os.path.join(split_dir, self.class_name)
            if os.path.isdir(label_dir):
                for img_name in os.listdir(label_dir):
                    img_path = os.path.join(label_dir, img_name)
                    images.append(img_path)
                    labels.append(int(self.class_name))  # The directory name is the label (0, 1, 2, 3)
        else:
            # Handle the case when class_name is None
            # raise ValueError("class_name must be specified to load a specific class.")
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

def get_dataloaders_for_all_classes(
    root_dir: str = "/home/work/jmpark/DGU_Medical/dataset_initial",
    img_dim: tuple[int, int] = (490, 490),
    batch_size: int = 16,
    split: str = "test"
) -> Dict[int, DataLoader]:
    """Get dataloaders for each class in the custom dataset.

    Args:
        root_dir (str, optional): The root directory of the dataset. Defaults to "./dataset_initial".
        img_dim (tuple[int, int], optional): The input size of the images. Defaults to (490, 490).
        batch_size (int, optional): The batch size of the dataloader. Defaults to 16.
        corruption_severity (int, optional): The corruption severity level between 1 and 5. Defaults to None.
        split (str, optional): Dataset split ('train', 'val', 'test'). Defaults to 'test'.

    Returns:
        Dict[int, DataLoader]: A dictionary where keys are class labels and values are DataLoader objects.
    """
    transform = A.Compose([A.Resize(height=img_dim[0], width=img_dim[1])])

    # Dictionary to hold dataloaders for each class
    dataloaders = {}

    # Load dataloaders for each class
    split_dir = os.path.join(root_dir, split)
    for class_name in os.listdir(split_dir):
        label_dir = os.path.join(split_dir, class_name)
        if os.path.isdir(label_dir):
            class_label = int(class_name)
            class_dataset = CustomDataset(
                root=root_dir,
                split=split,
                class_name=class_name,
                transform=transform,
            )

            class_loader = DataLoader(
                class_dataset,
                batch_size=batch_size,
                num_workers=4,
                shuffle=False,
                pin_memory=True
            )

            dataloaders[class_label] = class_loader

    return dataloaders

def get_dataloader(
    dataset_name: str,
    root_dir: str = "/home/work/jmpark/DGU_Medical/dataset_initial",
    img_dim: tuple[int, int] = (490, 490),
    batch_size: int = 16,
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

        val_dataset = CustomDataset(
            root=root_dir,
            split="val",
            transform=transform,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=4,
        persistent_workers=True,
        shuffle=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=True)
    
    return train_loader, val_loader

# def get_dataloader_for_evaluation(
#     dataset_name: str,
#     root_dir: str = "/home/work/jmpark/DGU_Medical/dataset_initial",
#     img_dim: tuple[int, int] = (490, 490),
#     batch_size: int = 16,
# ) -> DataLoader:
#     """Get the dataloaders for the custom dataset.

#     Args:
#         root_dir (str, optional): The root directory of the dataset. Defaults to "./dataset_initial".
#         img_dim (tuple[int, int], optional): The input size of the images. Defaults to (490, 490).
#         batch_size (int, optional): The batch size of the dataloader. Defaults to 6.
#         corruption_severity (int, optional): The corruption severity level between 1 and 5. Defaults to None.

#     Returns:
#        DataLoader: The test loader
#     """
#     assert dataset_name in ["custom"], "dataset name not in [custom]"
#     transform = A.Compose([A.Resize(height=img_dim[0], width=img_dim[1])])

#     if dataset_name == "custom":
#         test_dataset = CustomDataset(
#             root=root_dir,
#             split="test",
#             transform=transform,
#         )

#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=batch_size,
#         num_workers=4,
#         shuffle=False,
#         pin_memory=True
#     )
    
#     return test_loader

# class CustomDatasetPerClass(Dataset):
#     def __init__(
#         self,
#         root: str,
#         split: str = "test",
#         class_label: Optional[int] = None,
#         transform: Optional[transforms.Compose] = None
#     ):
#         """
#         Args:
#             root (str): Root directory of the dataset, e.g., 'dataset_initial/'.
#             split (str): Which split to use ('train', 'val', 'test').
#             class_label (Optional[int]): Specific class label to load. If None, loads all classes.
#             transform (Optional[transforms.Compose]): Transformations to apply to the images.
#         """
#         self.root = root
#         self.split = split
#         self.class_label = class_label  # New parameter to specify class label
#         self.transform = transform

#         # Get list of images and corresponding labels
#         self.images, self.labels = self._load_data()

#     def _load_data(self):
#         images = []
#         labels = []

#         # The root directory structure should be like 'dataset/test/0/', 'dataset/test/1/', etc.
#         split_dir = os.path.join(self.root, self.split)

#         # Get the list of class directories
#         class_dirs = []
#         if self.class_label is not None:
#             # Load data only from the specified class
#             label_str = str(self.class_label)
#             label_dir = os.path.join(split_dir, label_str)
#             if os.path.isdir(label_dir):
#                 class_dirs.append((label_dir, self.class_label))
#             else:
#                 raise ValueError(f"Class label {self.class_label} does not exist in {split_dir}")
#         else:
#             # Load data from all classes
#             for label in os.listdir(split_dir):
#                 label_dir = os.path.join(split_dir, label)
#                 if os.path.isdir(label_dir):
#                     class_dirs.append((label_dir, int(label)))

#         # Load images and labels
#         for label_dir, label in class_dirs:
#             for img_name in os.listdir(label_dir):
#                 img_path = os.path.join(label_dir, img_name)
#                 images.append(img_path)
#                 labels.append(label)

#         return images, labels

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, index: int):
#         # Load image
#         img_path = self.images[index]
#         image = cv2.imread(img_path)
#         if image is None:
#             raise IOError(f"Cannot read image at {img_path}")
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

#         # Get corresponding label
#         label = self.labels[index]

#         # Apply transformations
#         if self.transform is not None:
#             # If using Albumentations transforms
#             transformed = self.transform(image=image)
#             image = transformed["image"]
#         else:
#             # If no transform, normalize the image
#             image = image / 255.0

#         # Move channels to the first dimension
#         image = np.transpose(image, (2, 0, 1))  # Convert from HWC to CHW
#         image = torch.from_numpy(image).float()

#         return image, label

# def get_dataloader_for_evaluation_each_class(
#     dataset_name: str,
#     root_dir: str = "/home/work/jmpark/DGU_Medical/dataset_initial",
#     img_dim: tuple = (490, 490),
#     batch_size: int = 16,
#     corruption_severity: Optional[int] = None,
# ) -> Dict[int, DataLoader]:
#     """Get dataloaders for each class separately for evaluation.

#     Args:
#         dataset_name (str): Name of the dataset (e.g., 'custom').
#         root_dir (str, optional): Root directory of the dataset. Defaults to "/home/work/jmpark/DGU_Medical/dataset_initial".
#         img_dim (tuple, optional): Image dimensions (height, width). Defaults to (490, 490).
#         batch_size (int, optional): Batch size for the dataloaders. Defaults to 16.
#         corruption_severity (Optional[int], optional): Corruption severity level between 1 and 5. Defaults to None.

#     Returns:
#         Dict[int, DataLoader]: A dictionary mapping class labels to their corresponding DataLoaders.
#     """
#     assert dataset_name in ["custom"], "Dataset name not in ['custom']"

#     # Define the transformation
#     transform = A.Compose([A.Resize(height=img_dim[0], width=img_dim[1])])

#     if corruption_severity is not None:
#         transform = get_corruption_transforms(img_dim, corruption_severity)

#     # Get the list of class labels by reading the directory names in the test split
#     split_dir = os.path.join(root_dir, "test")
#     class_labels = [
#         int(label) for label in os.listdir(split_dir)
#         if os.path.isdir(os.path.join(split_dir, label))
#     ]
#     class_labels.sort()  # Optional: sort the class labels

#     dataloaders_per_class = {}

#     for class_label in class_labels:
#         # Create dataset for the specific class
#         test_dataset = CustomDatasetPerClass(
#             root=root_dir,
#             split="test",
#             class_label=class_label,
#             transform=transform,
#         )

#         # Create DataLoader for the specific class
#         test_loader = DataLoader(
#             test_dataset,
#             batch_size=batch_size,
#             num_workers=4,
#             shuffle=False,
#             pin_memory=True
#         )

#         dataloaders_per_class[class_label] = test_loader

#     return dataloaders_per_class