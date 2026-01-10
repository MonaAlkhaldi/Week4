# Deep Learning Foundations

This repository combines multiple assignments and experiments focused on **machine learning and deep learning fundamentals**, with an emphasis on **image classification using PyTorch**.  
The work reflects a progressive learning journey—from core neural network concepts to building and training **CNN models on real image datasets**.

---

##  C1M1 — Machine Learning & Neural Network Fundamentals  
**File:** `C1M1_Assignment.ipynb`

### Overview
This assignment introduces the foundational concepts of machine learning and neural networks. It focuses on **how models learn**, what parameters represent, and why certain model behaviors (underfitting vs. overfitting) occur.

### Topics Covered
- Machine learning problem formulation
- Parameters vs. hyperparameters
- Epochs, batches, and iterations
- High-level neural network intuition

### Key Outcome
Established a solid theoretical foundation required for understand

---
## C1M2 — PyTorch Workflow & Model Building  
**File:** `C1M2_Assignment.ipynb`

### Overview
This assignment transitions from theory to implementation using **PyTorch**, focusing on constructing and training neural networks from scratch.

### Topics Covered
- PyTorch tensors, datasets, and dataloaders
- Building models using `nn.Module` and `nn.Sequential`
- Forward pass and backpropagation
- Loss functions (e.g., `CrossEntropyLoss`)
- Optimizers (SGD, Adam)
- Training loops
- Device management (CPU vs. GPU)

### Key Outcome
Developed the ability to implement and train neural networks using PyTorch.

---

##  C1M4 — Training Loops, Evaluation & Debugging  
**File:** `C1M4_Assignment.ipynb`

### Overview
This assignment focuses on **training stability, evaluation correctness, and debugging performance issues** in deep learning workflows.

### Topics Covered
- Custom training and validation loops
- Batch-level and epoch-level logging
- Accuracy and loss tracking
- Using `torch.no_grad()` for evaluation
- Identifying slow data loading bottlenecks
- Debugging GPU/CPU performance issues

### Key Outcome
Gained practical experience in managing training pipelines and diagnosing performance problems.

---
## Assignment 3 — Plant Species Classification Using CNNs  
**File:** `Assignment3.ipynb`

### Overview
This capstone-style assignment applies all previous concepts to a **real-world image classification problem**: recognizing plant species from images using a **Convolutional Neural Network (CNN)**.

### Topics Covered
- Loading image datasets from directory structures
- Data preprocessing and normalization
- Data augmentation techniques:
  - Random resized crop
  - Horizontal flipping
- Class balance analysis
- Designing a custom CNN architecture
- Training and evaluating CNN models

### Key Outcome
Built an end-to-end **CNN-based image classification system** for plant species recognition.

---


## Assignment 4 — Overcoming Overfitting: Building a Robust CNN  
**File:** `Assignment4.ipynb`

### Overview
This final assignment addresses **overfitting in CNNs** by transforming an initially overfitting model into a **robust and generalizable architecture** using professional deep learning techniques.

### Key Improvements
- Stronger **data augmentation**
- Modular CNN design using reusable blocks
- **Batch Normalization** for stable training
- **Dropout** and **Weight Decay** for regularization

### Outcome
Built a well-regularized CNN with improved generalization and cleaner, scalable PyTorch architecture.
---
---
# Bone Fracture Detection using ResNet (Transfer Learning)

This project focuses on detecting bone fractures from X-ray images using deep learning.  
A pretrained **ResNet** model is fine-tuned to classify X-ray images into:

- `fractured`
- `normal`

The goal is to build a robust medical-image classifier using transfer learning and partial fine-tuning.

---

## Dataset

The dataset used in this project is:

**Bone Fracture X-Ray Dataset**  
https://www.kaggle.com/datasets/usman44m/bone-fracture-x-ray-dataset

It contains labeled X-ray images of bones, organized into two classes:
- Fractured bones
- Normal (non-fractured) bones

Number of samples:

| Split | Total Images |
|------|--------------|
| Train | **(9248)** |
| Test  | **(508)** |
| val   | **(831)** |

## Model & Approach

- Backbone: **ResNet (pretrained on ImageNet)**
- Strategy: **Transfer Learning with Partial Fine-Tuning**
- Froze all pretrained layers.
- Unfroze:
   - The last ResNet block (`layer4`)
   - The classification head (`fc`)
- Trained using Adam optimizer with a small learning rate.

  ## Results & Evaluation

| Metric Type        | Dataset Used        | Accuracy |
|--------------------|---------------------|----------|
| Validation Accuracy | Validation set (during training) | ~99% |
| Test Accuracy       | Unseen test set     | 98.6% |

