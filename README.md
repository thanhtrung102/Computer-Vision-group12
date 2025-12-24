# Computer Vision: MNIST Digit Classification

A comprehensive computer vision project implementing and comparing multiple classical machine learning and deep learning approaches for handwritten digit recognition using the MNIST dataset.

## Overview

This repository explores different methodologies for digit classification, ranging from traditional machine learning techniques to modern deep learning architectures. The project serves as an educational resource demonstrating the evolution of computer vision approaches and their comparative performance.

## Dataset

**MNIST (Modified National Institute of Standards and Technology)**
- 28x28 grayscale images of handwritten digits (0-9)
- Training set: 60,000 samples (10,000 for classical methods)
- Test set: 10,000 samples (2,000 for classical methods)
- 10 classes (digits 0-9)

## Implemented Approaches

### Deep Learning Models

#### 1. AlexNet (`AlexNet.ipynb`)
A CNN inspired by the AlexNet architecture, adapted for MNIST digit classification.

**Architecture:**
- 5 convolutional layers (32 → 64 → 96 → 64 → 32 filters)
- ReLU activations with max pooling
- 3 fully connected layers (2048 → 1024 → 10)
- Dropout regularization

**Training Configuration:**
- Optimizer: SGD (learning rate: 0.01)
- Loss: Cross-entropy
- Batch size: 32
- Epochs: 14

**Performance:**
- Best accuracy: **96.85%** (Epoch 4)
- Rapid convergence (>96% by epoch 3)

#### 2. LeNet (`LeNet.ipynb`)
Implementation of the classic LeNet-5 architecture.

**Architecture:**
- 2 convolutional layers (1 → 10 → 20 filters)
- Max pooling after each conv layer
- Dropout2d for regularization
- Fully connected layers: 320 → 50 → 10

**Training Configuration:**
- Optimizer: SGD (learning rate: 0.0001)
- Loss: Cross-entropy
- Batch size: 32
- Epochs: 14

**Performance:**
- High accuracy comparable to AlexNet
- More compact and efficient architecture

### Classical Machine Learning Approaches

#### 3. K-Nearest Neighbors - Euclidean Distance (`knn_Euclidean.ipynb`)
KNN classifier using Euclidean distance metric (L2 norm).

**Configuration:**
- Tested k values: [1, 2, 3, 4, 5, 6, 7, 8, 9]
- Best k: 3
- Dataset: 10,000 training / 2,000 test samples

**Performance:**
- Accuracy: **92.32%**
- Includes confusion matrix visualization

#### 4. K-Nearest Neighbors - Manhattan Distance (`knn_Manhattan.ipynb`)
KNN classifier using Manhattan distance metric (L1 norm).

**Configuration:**
- Tested k values: [1, 2, 3, 4, 5, 6, 7, 8, 9]
- Best k: 3
- Dataset: 10,000 training / 2,000 test samples

**Performance:**
- Accuracy: **91.26%**
- Demonstrates impact of distance metric choice

#### 5. Spatial Pyramid Matching (`Spatial Pyramid Matching.ipynb`)
Feature-based approach using Dense SIFT descriptors with SVM classifier.

**Feature Extraction:**
- Dense SIFT (Scale-Invariant Feature Transform)
- Step size: 4 pixels
- Vocabulary size: 100 clusters (K-means)

**Spatial Pyramid:**
- Pyramid levels: 0 and 1
- Weighted encoding (0.5 per level)
- Multi-scale spatial representation

**Classifier:**
- SVM with RBF kernel
- Hyperparameter tuning via 5-fold CV
- Best params: C=1.0, gamma=100.0

**Performance:**
- Accuracy: **65.18%**
- Per-class precision/recall analysis
- Best class: Digit 1 (0.87 precision, 0.95 recall)
- Most challenging: Digit 8 (0.45 precision, 0.50 recall)

#### 6. Support Vector Machine (`svm.ipynb`)
Standard SVM classifier with RBF kernel.

**Configuration:**
- C: 0.5
- Gamma: 0.05
- Normalized features [0, 1]
- Dataset: 10,000 training / 2,000 test samples

**Performance:**
- Confusion matrix analysis
- Traditional baseline approach

## Performance Comparison

| Method | Dataset Size (Train/Test) | Accuracy | Type |
|--------|---------------------------|----------|------|
| **AlexNet** | 60K / 10K | **96.85%** | Deep Learning |
| **LeNet** | 60K / 10K | **~96%** | Deep Learning |
| **KNN (Euclidean)** | 10K / 2K | **92.32%** | Distance-based |
| **KNN (Manhattan)** | 10K / 2K | **91.26%** | Distance-based |
| **SVM** | 10K / 2K | **~90%** | Kernel-based |
| **Spatial Pyramid Matching** | 10K / 2K | **65.18%** | Feature-based |

## Key Insights

1. **Deep learning superiority**: CNNs (AlexNet, LeNet) achieve highest accuracy (96%+) with learned hierarchical features
2. **Distance metrics matter**: Euclidean distance slightly outperforms Manhattan in KNN (92.32% vs 91.26%)
3. **Feature engineering complexity**: Hand-crafted features (Dense SIFT + SPM) require careful tuning but offer interpretability
4. **Computational trade-offs**: Classical methods use smaller datasets due to computational constraints
5. **Convergence speed**: Deep models reach high performance within 3-4 epochs

## Dependencies

### Deep Learning
```bash
pip install torch torchvision
```

### Classical ML & Computer Vision
```bash
pip install scikit-learn opencv-python scipy numpy matplotlib
```

### Complete Requirements
- PyTorch & torchvision (deep learning)
- scikit-learn (KNN, SVM, clustering, metrics)
- OpenCV (cv2) (image processing, SIFT)
- NumPy (numerical operations)
- SciPy (scientific computing)
- Matplotlib (visualization)
- Pickle (model serialization)

## Usage

Each notebook is self-contained and can be run independently:

```bash
# For Jupyter Notebook
jupyter notebook <notebook_name>.ipynb

# For Jupyter Lab
jupyter lab <notebook_name>.ipynb

# For Google Colab
# Upload the notebook and run all cells
```

**Note:** The notebooks were created using Google Colab and can be directly opened there.

## Project Structure

```
.
├── AlexNet.ipynb                      # AlexNet CNN implementation
├── LeNet.ipynb                        # LeNet-5 CNN implementation
├── knn_Euclidean.ipynb               # KNN with Euclidean distance
├── knn_Manhattan.ipynb               # KNN with Manhattan distance
├── Spatial Pyramid Matching.ipynb    # Dense SIFT + SPM + SVM
├── svm.ipynb                         # Standard SVM classifier
└── README.md                         # This file
```

## Outputs & Artifacts

- **Trained models**: PyTorch models saved during training
- **Codebook**: `spm_lv1_codebook.pkl` (SPM vocabulary)
- **Visualizations**:
  - Training curves
  - Confusion matrices (normalized and raw)
  - Accuracy vs. hyperparameter plots

## Future Improvements

- Implement data augmentation for deep learning models
- Add modern architectures (ResNet, VGG, EfficientNet)
- Ensemble methods combining multiple classifiers
- Transfer learning experiments
- Real-time digit recognition application
- Extended evaluation metrics (F1-score, ROC curves)
- Cross-validation for deep learning models

## Contributors

Group 12 - Computer Vision Project

## License

This project is for educational purposes.

## Acknowledgments

- MNIST dataset: Yann LeCun, Corinna Cortes, Christopher J.C. Burges
- PyTorch framework
- scikit-learn library
- Google Colab for computational resources
