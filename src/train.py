"""
Training Utilities for Neural Network Models

This module provides a Trainer class for training PyTorch models with
support for early stopping, learning rate scheduling, and model checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Callable
import time
from pathlib import Path
from tqdm import tqdm


class Trainer:
    """
    A flexible trainer for PyTorch models.

    Features:
        - Training and validation loops
        - Early stopping
        - Learning rate scheduling
        - Model checkpointing
        - Training history tracking
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module = nn.CrossEntropyLoss(),
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the trainer.

        Args:
            model: PyTorch model to train
            optimizer: Optimizer for training
            criterion: Loss function
            scheduler: Optional learning rate scheduler
            device: Device to use for training (auto-detected if None)
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': [],
        }

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)
        for data, target in pbar:
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * data.size(0)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total

        return {'loss': epoch_loss, 'acc': epoch_acc}

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model on a dataset."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        for data, target in data_loader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = self.criterion(output, target)

            running_loss += loss.item() * data.size(0)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        return {
            'loss': running_loss / total,
            'acc': 100. * correct / total,
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        early_stopping_patience: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of epochs to train
            early_stopping_patience: Stop if val_loss doesn't improve for N epochs
            checkpoint_path: Path to save best model checkpoint
            verbose: Whether to print progress

        Returns:
            Training history dictionary
        """
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            # Training
            train_metrics = self.train_epoch(train_loader, epoch)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['acc'])

            # Validation
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_acc'].append(val_metrics['acc'])

            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['lr'].append(current_lr)

            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'] if val_loader else train_metrics['loss'])
                else:
                    self.scheduler.step()

            # Logging
            if verbose:
                elapsed = time.time() - start_time
                msg = f"Epoch {epoch}/{epochs} ({elapsed:.1f}s) - "
                msg += f"train_loss: {train_metrics['loss']:.4f}, train_acc: {train_metrics['acc']:.2f}%"
                if val_loader is not None:
                    msg += f" - val_loss: {val_metrics['loss']:.4f}, val_acc: {val_metrics['acc']:.2f}%"
                print(msg)

            # Early stopping and checkpointing
            if val_loader is not None:
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    patience_counter = 0
                    if checkpoint_path:
                        self.save_checkpoint(checkpoint_path)
                else:
                    patience_counter += 1
                    if early_stopping_patience and patience_counter >= early_stopping_patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch}")
                        break

        return self.history

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)


def create_optimizer(
    model: nn.Module,
    optimizer_name: str = 'adam',
    lr: float = 0.001,
    weight_decay: float = 0.0,
    **kwargs
) -> optim.Optimizer:
    """
    Factory function to create an optimizer.

    Args:
        model: Model to optimize
        optimizer_name: Name of optimizer ('adam', 'sgd', 'adamw')
        lr: Learning rate
        weight_decay: Weight decay (L2 regularization)
        **kwargs: Additional optimizer arguments

    Returns:
        Configured optimizer
    """
    optimizers = {
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'sgd': optim.SGD,
        'rmsprop': optim.RMSprop,
    }

    optimizer_name = optimizer_name.lower()
    if optimizer_name not in optimizers:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return optimizers[optimizer_name](
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        **kwargs
    )
