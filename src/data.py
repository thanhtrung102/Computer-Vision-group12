"""
Data Loading Utilities for MNIST Dataset

This module provides functions for loading and preprocessing the MNIST dataset
with support for data augmentation and various transforms.
"""

import torch
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from typing import Tuple, Optional
import numpy as np


def get_transforms(augment: bool = False) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get train and test transforms for MNIST.

    Args:
        augment: Whether to apply data augmentation to training data

    Returns:
        Tuple of (train_transform, test_transform)
    """
    # Standard normalization for MNIST
    normalize = transforms.Normalize((0.1307,), (0.3081,))

    if augment:
        train_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, test_transform


def get_mnist_loaders(
    batch_size: int = 64,
    augment: bool = False,
    num_workers: int = 0,
    data_dir: str = './data',
    validation_split: Optional[float] = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Get MNIST data loaders.

    Args:
        batch_size: Batch size for training and testing
        augment: Whether to apply data augmentation
        num_workers: Number of workers for data loading
        data_dir: Directory to store/load MNIST data
        validation_split: If provided, split training data into train/val
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_loader, test_loader, val_loader or None)
    """
    train_transform, test_transform = get_transforms(augment)

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    val_loader = None

    if validation_split is not None:
        # Create train/validation split
        np.random.seed(seed)
        indices = np.random.permutation(len(train_dataset))
        split_idx = int(len(indices) * (1 - validation_split))

        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        # Create validation dataset with test transform (no augmentation)
        val_dataset = datasets.MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=test_transform,
        )

        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, test_loader, val_loader


def get_sample_images(data_loader: DataLoader, n_samples: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get sample images from a data loader.

    Args:
        data_loader: DataLoader to sample from
        n_samples: Number of samples to return

    Returns:
        Tuple of (images, labels)
    """
    images, labels = next(iter(data_loader))
    return images[:n_samples], labels[:n_samples]


def get_class_distribution(data_loader: DataLoader) -> dict:
    """
    Get class distribution from a data loader.

    Args:
        data_loader: DataLoader to analyze

    Returns:
        Dictionary mapping class labels to counts
    """
    distribution = {i: 0 for i in range(10)}
    for _, labels in data_loader:
        for label in labels:
            distribution[label.item()] += 1
    return distribution
