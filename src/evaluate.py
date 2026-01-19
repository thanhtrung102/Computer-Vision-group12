"""
Evaluation Utilities for Model Assessment

This module provides comprehensive evaluation tools including:
- Accuracy metrics
- Confusion matrix generation
- Per-class performance analysis
- Inference time benchmarking
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)
import time


class Evaluator:
    """
    Comprehensive model evaluator for classification tasks.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        class_names: Optional[List[str]] = None,
    ):
        """
        Initialize the evaluator.

        Args:
            model: Trained PyTorch model
            device: Device for inference
            class_names: Names for each class (default: digits 0-9)
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model.eval()
        self.class_names = class_names or [str(i) for i in range(10)]

    @torch.no_grad()
    def get_predictions(
        self,
        data_loader: DataLoader,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get all predictions for a dataset.

        Returns:
            Tuple of (predictions, true_labels, probabilities)
        """
        all_preds = []
        all_labels = []
        all_probs = []

        for data, target in data_loader:
            data = data.to(self.device)
            output = self.model(data)
            probs = torch.softmax(output, dim=1)
            preds = output.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.numpy())
            all_probs.extend(probs.cpu().numpy())

        return np.array(all_preds), np.array(all_labels), np.array(all_probs)

    def evaluate(self, data_loader: DataLoader) -> Dict:
        """
        Comprehensive evaluation on a dataset.

        Returns:
            Dictionary containing all metrics
        """
        preds, labels, probs = self.get_predictions(data_loader)

        # Basic accuracy
        accuracy = (preds == labels).mean() * 100

        # Confusion matrix
        cm = confusion_matrix(labels, preds)

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, preds, average=None, zero_division=0
        )

        # Macro and weighted averages
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            labels, preds, average='macro', zero_division=0
        )
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            labels, preds, average='weighted', zero_division=0
        )

        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'per_class': {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': support,
            },
            'macro': {
                'precision': macro_precision,
                'recall': macro_recall,
                'f1': macro_f1,
            },
            'weighted': {
                'precision': weighted_precision,
                'recall': weighted_recall,
                'f1': weighted_f1,
            },
            'predictions': preds,
            'labels': labels,
            'probabilities': probs,
        }

    def get_misclassified(
        self,
        data_loader: DataLoader,
        n_samples: Optional[int] = None,
    ) -> List[Dict]:
        """
        Get misclassified samples for error analysis.

        Returns:
            List of dictionaries containing misclassified samples info
        """
        misclassified = []

        for data, target in data_loader:
            data_device = data.to(self.device)
            with torch.no_grad():
                output = self.model(data_device)
                probs = torch.softmax(output, dim=1)
                preds = output.argmax(dim=1).cpu()

            for i in range(len(target)):
                if preds[i] != target[i]:
                    misclassified.append({
                        'image': data[i].numpy(),
                        'true_label': target[i].item(),
                        'predicted_label': preds[i].item(),
                        'confidence': probs[i, preds[i]].item(),
                        'probabilities': probs[i].cpu().numpy(),
                    })

                    if n_samples and len(misclassified) >= n_samples:
                        return misclassified

        return misclassified

    def benchmark_inference(
        self,
        data_loader: DataLoader,
        n_runs: int = 3,
    ) -> Dict[str, float]:
        """
        Benchmark inference time.

        Returns:
            Dictionary with timing statistics
        """
        times = []
        total_samples = 0

        for _ in range(n_runs):
            start = time.time()
            for data, _ in data_loader:
                data = data.to(self.device)
                with torch.no_grad():
                    _ = self.model(data)
                total_samples += data.size(0)
            times.append(time.time() - start)

        avg_time = np.mean(times)
        samples_per_run = total_samples // n_runs

        return {
            'total_time': avg_time,
            'samples_per_second': samples_per_run / avg_time,
            'ms_per_sample': (avg_time / samples_per_run) * 1000,
            'n_samples': samples_per_run,
        }

    def print_report(self, metrics: Dict) -> None:
        """Print a formatted evaluation report."""
        print("=" * 60)
        print("MODEL EVALUATION REPORT")
        print("=" * 60)
        print(f"\nOverall Accuracy: {metrics['accuracy']:.2f}%")
        print(f"\nMacro Averages:")
        print(f"  Precision: {metrics['macro']['precision']:.4f}")
        print(f"  Recall:    {metrics['macro']['recall']:.4f}")
        print(f"  F1-Score:  {metrics['macro']['f1']:.4f}")
        print(f"\nWeighted Averages:")
        print(f"  Precision: {metrics['weighted']['precision']:.4f}")
        print(f"  Recall:    {metrics['weighted']['recall']:.4f}")
        print(f"  F1-Score:  {metrics['weighted']['f1']:.4f}")
        print("\nPer-Class Performance:")
        print("-" * 60)
        print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 60)
        for i, name in enumerate(self.class_names):
            print(f"{name:<10} {metrics['per_class']['precision'][i]:<12.4f} "
                  f"{metrics['per_class']['recall'][i]:<12.4f} "
                  f"{metrics['per_class']['f1'][i]:<12.4f} "
                  f"{int(metrics['per_class']['support'][i]):<10}")
        print("=" * 60)
