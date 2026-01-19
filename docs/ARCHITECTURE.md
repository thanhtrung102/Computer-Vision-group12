# Architecture & Design Decisions

This document explains the architectural choices and design decisions made in this project.

## Table of Contents

- [Project Overview](#project-overview)
- [Model Architectures](#model-architectures)
- [Design Principles](#design-principles)
- [Trade-offs & Decisions](#trade-offs--decisions)
- [Lessons Learned](#lessons-learned)

---

## Project Overview

### Goal

Create a comprehensive comparison of machine learning approaches for handwritten digit classification, demonstrating the evolution from classical methods to deep learning.

### Scope

- **Dataset**: MNIST (28x28 grayscale images, 60K train / 10K test)
- **Task**: 10-class classification (digits 0-9)
- **Methods**: KNN, SVM, SPM, LeNet, AlexNet

---

## Model Architectures

### Classical Methods

#### K-Nearest Neighbors (KNN)

**Why KNN?**
- Simple baseline with no training required
- Demonstrates importance of distance metrics
- Interpretable predictions

**Design Choices:**
- Tested k values: 1, 3, 5, 7
- Best performance at k=3
- Euclidean > Manhattan for pixel-space similarity

**Limitations:**
- Slow inference (O(n) per prediction)
- No learned features
- Sensitive to noise

#### Support Vector Machine (SVM)

**Why SVM?**
- Strong theoretical foundations
- Effective for high-dimensional data
- Kernel trick enables non-linear boundaries

**Design Choices:**
- RBF kernel (best for image data)
- Grid search for C and gamma
- One-vs-Rest for multiclass

#### Spatial Pyramid Matching (SPM)

**Why SPM?**
- Bridge between hand-crafted and learned features
- Captures spatial structure
- Classic computer vision approach

**Design Choices:**
- Dense SIFT at multiple scales
- 3-level pyramid (1x1, 2x2, 4x4)
- 200 visual words via K-means

**Limitations:**
- Computationally expensive
- SIFT designed for natural images, not digits
- Feature engineering bottleneck

### Deep Learning

#### LeNet-5

**Architecture:**
```
Input (1x28x28)
    ↓
Conv2d(1→10, 5x5) + ReLU + MaxPool(2x2)
    ↓
Conv2d(10→20, 5x5) + Dropout + ReLU + MaxPool(2x2)
    ↓
Flatten (320)
    ↓
Linear(320→50) + ReLU + Dropout
    ↓
Linear(50→10) + LogSoftmax
```

**Why LeNet?**
- Historical significance (pioneered CNNs)
- Simple yet effective
- Fast training and inference

**Modifications from Original:**
- Added dropout for regularization
- Using ReLU instead of tanh
- Batch normalization option

#### AlexNet (Adapted)

**Architecture:**
```
Input (1x28x28)
    ↓
Conv2d(1→32, 5x5) + ReLU
    ↓
Conv2d(32→64, 3x3) + ReLU + MaxPool(2x2)
    ↓
Conv2d(64→96, 3x3) + ReLU
    ↓
Conv2d(96→64, 3x3) + ReLU
    ↓
Conv2d(64→32, 3x3) + ReLU + MaxPool(2x2)
    ↓
Flatten (32x12x12 = 4608)
    ↓
Linear(4608→2048) + ReLU + Dropout
    ↓
Linear(2048→1024) + ReLU + Dropout
    ↓
Linear(1024→10)
```

**Adaptations for MNIST:**
- Reduced channels (ImageNet uses 3 channels)
- Smaller kernel sizes
- Fewer parameters overall

---

## Design Principles

### 1. Modularity

Each component is self-contained and reusable:

```python
# Data loading
from src.data import get_mnist_loaders

# Models
from src.models import LeNet, AlexNet

# Training
from src.train import Trainer

# Evaluation
from src.evaluate import Evaluator
```

### 2. Reproducibility

- Fixed random seeds
- Documented hyperparameters
- Version-controlled dependencies

### 3. Extensibility

Easy to add new models:

```python
from src.models import get_model

# Add to models.py registry
model = get_model('your_new_model')
```

### 4. Visualization-First

Every experiment produces visual outputs:
- Training curves
- Confusion matrices
- Sample predictions
- Feature embeddings

---

## Trade-offs & Decisions

### Why Not Use Pre-trained Models?

**Decision:** Train from scratch

**Reasoning:**
- MNIST is small enough for full training
- Educational value in seeing training dynamics
- Pre-trained models are for different domains (ImageNet)

### Data Augmentation

**Decision:** Optional, off by default

**Reasoning:**
- MNIST is "solved" without augmentation
- Augmentation shows marginal improvements
- Keeps baselines comparable to literature

### Batch Size Selection

**Decision:** 64 for deep learning, full dataset for classical

**Reasoning:**
- 64 balances memory and gradient noise
- Classical methods often work better with full data
- Consistent with common practice

### Learning Rate

**Decision:** Adam with 0.001, SGD with 0.01

**Reasoning:**
- Adam converges faster, good default
- SGD used in original papers
- Both achieve similar final accuracy

---

## Lessons Learned

### 1. Feature Learning > Feature Engineering

SPM with carefully designed SIFT features achieved 65% accuracy. A simple CNN with learned features achieved 94%+ with less effort.

**Takeaway:** Let the network learn features when you have enough data.

### 2. Architecture Matters Less Than Expected

LeNet (44K params) vs AlexNet (6.4M params) differ by <1% accuracy.

**Takeaway:** For simple tasks, simpler models often suffice. Don't over-engineer.

### 3. Distance Metrics Are Important

Euclidean vs Manhattan made a 1% difference in KNN.

**Takeaway:** Small choices compound. Test assumptions.

### 4. Regularization Is Essential

Without dropout, models quickly overfit MNIST.

**Takeaway:** Always use appropriate regularization, even on "easy" datasets.

---

## Future Improvements

1. **Modern Architectures**: ResNet, Vision Transformer
2. **Data Augmentation Study**: Systematic comparison
3. **Ensemble Methods**: Combine predictions from multiple models
4. **Adversarial Robustness**: Test against perturbations
5. **Interpretability**: Grad-CAM, attention visualization

---

## References

1. LeCun, Y., et al. (1998). "Gradient-based learning applied to document recognition."
2. Krizhevsky, A., et al. (2012). "ImageNet classification with deep convolutional neural networks."
3. Lazebnik, S., et al. (2006). "Beyond bags of features: Spatial pyramid matching."
