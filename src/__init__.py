"""
MNIST Digit Classification - Computer Vision Portfolio Project

A comprehensive comparison of machine learning approaches for handwritten digit recognition,
from classical methods (KNN, SVM) to deep learning architectures (LeNet, AlexNet).
"""

from .models import LeNet, AlexNet, SimpleCNN
from .data import get_mnist_loaders, get_transforms
from .train import Trainer
from .evaluate import Evaluator
from .visualize import plot_confusion_matrix, plot_training_history, plot_sample_predictions

__version__ = "1.0.0"
__author__ = "Computer Vision Group 12"

__all__ = [
    "LeNet",
    "AlexNet",
    "SimpleCNN",
    "get_mnist_loaders",
    "get_transforms",
    "Trainer",
    "Evaluator",
    "plot_confusion_matrix",
    "plot_training_history",
    "plot_sample_predictions",
]
