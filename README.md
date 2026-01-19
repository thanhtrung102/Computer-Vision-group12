<p align="center">
  <img src="docs/images/banner.png" alt="MNIST Classification Banner" width="800"/>
</p>

<h1 align="center">MNIST Digit Classification: A Comparative Study</h1>

<p align="center">
  <strong>From Classical Machine Learning to Deep Learning</strong>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#results">Results</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#demo">Demo</a> •
  <a href="#methods">Methods</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.3+-f7931e.svg" alt="sklearn"/>
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"/>
</p>

---

## Overview

This project provides a comprehensive comparison of machine learning approaches for handwritten digit classification on the MNIST dataset. We implement and evaluate methods ranging from classical algorithms (KNN, SVM) to modern deep learning architectures (LeNet, AlexNet), demonstrating the evolution and effectiveness of different computer vision techniques.

### Key Features

- **6 Different Approaches**: KNN (Euclidean & Manhattan), SVM, Spatial Pyramid Matching, LeNet, AlexNet
- **Modular Codebase**: Reusable components for data loading, training, and evaluation
- **Interactive Demo**: Streamlit web app for real-time digit recognition
- **Comprehensive Analysis**: Confusion matrices, per-class metrics, and visualizations
- **Docker Support**: Easy deployment with containerization

## Results

| Model | Accuracy | Parameters | Inference (ms/sample) |
|-------|----------|------------|----------------------|
| **AlexNet** | **96.85%** | 6.4M | 0.8 |
| **LeNet** | 96.00% | 44K | 0.3 |
| KNN (Euclidean) | 92.32% | - | 15.2 |
| KNN (Manhattan) | 91.26% | - | 14.8 |
| SVM (RBF) | 89.50% | - | 8.5 |
| SPM + SVM | 65.18% | - | 120.3 |

<p align="center">
  <img src="docs/images/accuracy_comparison.png" alt="Accuracy Comparison" width="600"/>
</p>

### Key Findings

1. **Deep Learning Superiority**: CNNs significantly outperform classical methods, with AlexNet achieving the highest accuracy
2. **Distance Metric Impact**: Euclidean distance outperforms Manhattan for KNN on MNIST
3. **Feature Engineering Trade-off**: Hand-crafted features (SPM) require careful tuning and still underperform learned representations
4. **Efficiency vs Accuracy**: LeNet provides an excellent balance with near-AlexNet accuracy at 1% of the parameters

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster training)

### Setup

```bash
# Clone the repository
git clone https://github.com/thanhtrung102/Computer-Vision-group12.git
cd Computer-Vision-group12

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Docker Installation

```bash
# Build the Docker image
docker build -t mnist-classifier .

# Run the container
docker run -p 8501:8501 mnist-classifier
```

## Usage

### Quick Start

```python
from src import LeNet, get_mnist_loaders, Trainer, Evaluator
import torch.optim as optim

# Load data
train_loader, test_loader, _ = get_mnist_loaders(batch_size=64)

# Create model and trainer
model = LeNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
trainer = Trainer(model, optimizer)

# Train
history = trainer.fit(train_loader, epochs=10)

# Evaluate
evaluator = Evaluator(model)
metrics = evaluator.evaluate(test_loader)
print(f"Test Accuracy: {metrics['accuracy']:.2f}%")
```

### Running Notebooks

```bash
# Start Jupyter
jupyter notebook notebooks/
```

### Running the Demo App

```bash
streamlit run app/streamlit_app.py
```

## Demo

<p align="center">
  <img src="docs/images/demo.gif" alt="Demo" width="600"/>
</p>

Try our interactive demo where you can:
- Draw digits and get real-time predictions
- Compare predictions across all models
- View confidence scores and probability distributions

**[Launch Demo on Hugging Face Spaces](https://huggingface.co/spaces/your-username/mnist-classifier)** *(coming soon)*

## Methods

### Classical Machine Learning

#### K-Nearest Neighbors (KNN)
- **Euclidean Distance**: Standard L2 norm, achieved 92.32% accuracy with k=3
- **Manhattan Distance**: L1 norm, achieved 91.26% accuracy with k=3

#### Support Vector Machine (SVM)
- RBF kernel with grid search for hyperparameter optimization
- Achieved 89.50% on raw pixel features

#### Spatial Pyramid Matching (SPM)
- Dense SIFT feature extraction
- 3-level spatial pyramid (1x1, 2x2, 4x4)
- SVM classifier with histogram intersection kernel

### Deep Learning

#### LeNet-5
```
Input(1x28x28) → Conv(10,5x5) → Pool → Conv(20,5x5) → Pool → FC(50) → FC(10)
```
- Classic architecture by Yann LeCun
- ~44K parameters
- Dropout for regularization

#### AlexNet (Adapted)
```
Input(1x28x28) → 5 Conv layers → 3 FC layers → Output(10)
```
- Adapted from ImageNet architecture
- ~6.4M parameters
- ReLU activation, Dropout

## Project Structure

```
Computer-Vision-group12/
├── app/
│   └── streamlit_app.py      # Interactive demo
├── docs/
│   ├── images/               # Documentation images
│   └── ARCHITECTURE.md       # Design decisions
├── notebooks/
│   ├── AlexNet.ipynb
│   ├── LeNet.ipynb
│   ├── knn_Euclidean.ipynb
│   ├── knn_Manhattan.ipynb
│   ├── svm.ipynb
│   ├── Spatial Pyramid Matching.ipynb
│   └── Model_Comparison.ipynb
├── results/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── data.py               # Data loading utilities
│   ├── models.py             # Neural network architectures
│   ├── train.py              # Training utilities
│   ├── evaluate.py           # Evaluation metrics
│   └── visualize.py          # Visualization tools
├── .gitignore
├── Dockerfile
├── LICENSE
├── README.md
└── requirements.txt
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) by Yann LeCun
- [PyTorch](https://pytorch.org/) team for the excellent deep learning framework
- Course instructors and TAs for guidance

## Contact

**Computer Vision Group 12**

- GitHub: [@thanhtrung102](https://github.com/thanhtrung102)

---

<p align="center">
  Made with dedication for learning Computer Vision
</p>
