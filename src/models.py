"""
Neural Network Models for MNIST Classification

This module contains implementations of various CNN architectures:
- LeNet: Classic architecture by Yann LeCun (1998)
- AlexNet: Adapted version for MNIST (originally for ImageNet)
- SimpleCNN: Lightweight model for quick experiments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class LeNet(nn.Module):
    """
    LeNet-5 Architecture (adapted for MNIST)

    Original paper: "Gradient-Based Learning Applied to Document Recognition"
    by Yann LeCun et al. (1998)

    Architecture:
        Input (1x28x28) -> Conv1 -> Pool -> Conv2 -> Pool -> FC1 -> FC2 -> Output (10)

    Expected accuracy: ~96% on MNIST
    """

    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.5):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=dropout_rate)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before the final classification layer."""
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return x


class AlexNet(nn.Module):
    """
    AlexNet Architecture (adapted for MNIST)

    Original paper: "ImageNet Classification with Deep Convolutional Neural Networks"
    by Alex Krizhevsky et al. (2012)

    This is a simplified version adapted for 28x28 grayscale images.

    Expected accuracy: ~97% on MNIST
    """

    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.5):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(32 * 12 * 12, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(-1, 32 * 12 * 12)
        x = self.classifier(x)
        return x

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the last convolutional layer."""
        x = self.features(x)
        x = x.view(-1, 32 * 12 * 12)
        return x


class SimpleCNN(nn.Module):
    """
    Simple CNN for quick experiments and baseline comparisons.

    A lightweight model with fewer parameters for faster training.

    Expected accuracy: ~94% on MNIST
    """

    def __init__(self, num_classes: int = 10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def get_model(name: str, num_classes: int = 10, **kwargs) -> nn.Module:
    """
    Factory function to get a model by name.

    Args:
        name: Model name ('lenet', 'alexnet', 'simple')
        num_classes: Number of output classes
        **kwargs: Additional arguments passed to model constructor

    Returns:
        Initialized model
    """
    models = {
        'lenet': LeNet,
        'alexnet': AlexNet,
        'simple': SimpleCNN,
    }

    name = name.lower()
    if name not in models:
        raise ValueError(f"Unknown model: {name}. Available: {list(models.keys())}")

    return models[name](num_classes=num_classes, **kwargs)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count model parameters.

    Returns:
        Tuple of (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
