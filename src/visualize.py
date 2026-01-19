"""
Visualization Utilities for Model Analysis

This module provides functions for creating visualizations:
- Training history plots
- Confusion matrices
- Sample predictions
- Feature embeddings (t-SNE/UMAP)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def plot_training_history(
    history: Dict[str, List[float]],
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot training history (loss and accuracy curves).

    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        figsize: Figure size
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', color='blue')
    if 'val_loss' in history and history['val_loss']:
        axes[0].plot(history['val_loss'], label='Val Loss', color='orange')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train Acc', color='blue')
    if 'val_acc' in history and history['val_acc']:
        axes[1].plot(history['val_acc'], label='Val Acc', color='orange')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'Blues',
    normalize: bool = False,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a confusion matrix.

    Args:
        cm: Confusion matrix array
        class_names: Names for each class
        figsize: Figure size
        cmap: Colormap name
        normalize: Whether to normalize the matrix
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure
    """
    if class_names is None:
        class_names = [str(i) for i in range(len(cm))]

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_sample_predictions(
    images: np.ndarray,
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    n_samples: int = 10,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot sample predictions with images.

    Args:
        images: Array of images (N, C, H, W) or (N, H, W)
        true_labels: True labels
        pred_labels: Predicted labels
        probabilities: Prediction probabilities
        class_names: Names for each class
        n_samples: Number of samples to plot
        figsize: Figure size
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure
    """
    if class_names is None:
        class_names = [str(i) for i in range(10)]

    n_samples = min(n_samples, len(images))
    n_cols = min(5, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols

    if figsize is None:
        figsize = (3 * n_cols, 3.5 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).flatten()

    for i in range(n_samples):
        ax = axes[i]

        # Handle different image formats
        img = images[i]
        if len(img.shape) == 3:
            img = img.squeeze()

        ax.imshow(img, cmap='gray')
        ax.axis('off')

        # Color based on correctness
        color = 'green' if true_labels[i] == pred_labels[i] else 'red'

        title = f'True: {class_names[true_labels[i]]}\nPred: {class_names[pred_labels[i]]}'
        if probabilities is not None:
            title += f'\nConf: {probabilities[i, pred_labels[i]]:.2f}'

        ax.set_title(title, fontsize=10, color=color)

    # Hide empty subplots
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_misclassified_samples(
    misclassified: List[Dict],
    class_names: Optional[List[str]] = None,
    n_samples: int = 20,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot misclassified samples for error analysis.

    Args:
        misclassified: List of dictionaries from Evaluator.get_misclassified()
        class_names: Names for each class
        n_samples: Number of samples to plot
        figsize: Figure size
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure
    """
    if class_names is None:
        class_names = [str(i) for i in range(10)]

    n_samples = min(n_samples, len(misclassified))
    n_cols = min(5, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols

    if figsize is None:
        figsize = (3 * n_cols, 3.5 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).flatten()

    for i in range(n_samples):
        ax = axes[i]
        sample = misclassified[i]

        img = sample['image'].squeeze()
        ax.imshow(img, cmap='gray')
        ax.axis('off')

        title = (f"True: {class_names[sample['true_label']]}\n"
                f"Pred: {class_names[sample['predicted_label']]}\n"
                f"Conf: {sample['confidence']:.2f}")
        ax.set_title(title, fontsize=9, color='red')

    for i in range(n_samples, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Misclassified Samples', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_model_comparison(
    results: Dict[str, Dict],
    metrics: List[str] = ['accuracy'],
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot comparison of multiple models.

    Args:
        results: Dictionary mapping model names to their metrics
        metrics: List of metrics to plot
        figsize: Figure size
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure
    """
    model_names = list(results.keys())
    n_metrics = len(metrics)

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(model_names))
    width = 0.8 / n_metrics

    colors = plt.cm.Set2(np.linspace(0, 1, n_metrics))

    for i, metric in enumerate(metrics):
        values = [results[name].get(metric, 0) for name in model_names]
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=metric.replace('_', ' ').title(), color=colors[i])

        # Add value labels
        for bar, val in zip(bars, values):
            ax.annotate(f'{val:.1f}%' if 'acc' in metric.lower() else f'{val:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 3),
                       textcoords='offset points',
                       ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


@torch.no_grad()
def plot_feature_embeddings(
    model: nn.Module,
    data_loader: DataLoader,
    method: str = 'tsne',
    n_samples: int = 1000,
    device: Optional[torch.device] = None,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot 2D feature embeddings using t-SNE or UMAP.

    Args:
        model: Model with get_features() method
        data_loader: Data loader
        method: 'tsne' or 'umap'
        n_samples: Number of samples to visualize
        device: Device for inference
        figsize: Figure size
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model.eval()

    features = []
    labels = []

    for data, target in data_loader:
        if len(features) * data.size(0) >= n_samples:
            break
        data = data.to(device)
        feat = model.get_features(data)
        features.append(feat.cpu().numpy())
        labels.extend(target.numpy())

    features = np.vstack(features)[:n_samples]
    labels = np.array(labels)[:n_samples]

    # Dimensionality reduction
    if method.lower() == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    elif method.lower() == 'umap':
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
        except ImportError:
            raise ImportError("UMAP not installed. Run: pip install umap-learn")
    else:
        raise ValueError(f"Unknown method: {method}. Use 'tsne' or 'umap'")

    embeddings = reducer.fit_transform(features)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    scatter = ax.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        c=labels,
        cmap='tab10',
        alpha=0.6,
        s=10,
    )

    cbar = plt.colorbar(scatter, ax=ax, ticks=range(10))
    cbar.set_label('Digit')

    ax.set_xlabel(f'{method.upper()} Component 1')
    ax.set_ylabel(f'{method.upper()} Component 2')
    ax.set_title(f'Feature Embeddings ({method.upper()})')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
