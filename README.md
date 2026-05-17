# VGGNet16-BloodCell-Classification

> Deep Learning Assignment | Shri Mata Vaishno Devi University | May 2026

Automated classification of white blood cells using **VGGNet-16 with Transfer Learning** on the Blood Cell Images (BCCD) dataset.

---

## Team

| Name | Roll No |
|------|---------|
| Abhishek Kumar Kaith | 23BCS005 |
| Kashav Thappa | 23BCS040 |
| Akshit Kumar | 23BCS011 |

---

## Results

| Metric | Score |
|--------|-------|
| Test Accuracy | 91.63% |
| F1 Score (Macro) | 0.9160 |
| F1 Score (Weighted) | 0.9158 |
| AUC-ROC (Macro) | 0.9857 |
| Best Val Accuracy | 91.96% |

---

## Dataset

- **Name:** Blood Cell Images (BCCD)
- **Source:** [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/blood-cells)
- **Size:** 12,500 microscopic blood cell images
- **Classes:** Eosinophil, Lymphocyte, Monocyte, Neutrophil
- **Split:** 70% Train / 15% Val / 15% Test

---

## Model Architecture

- **Base:** VGGNet-16 (pretrained on ImageNet)
- **Strategy:** Transfer Learning — frozen conv layers, custom 4-class classifier head
- **Classifier:** Linear(25088→4096) → ReLU → Dropout(0.5) → Linear(4096→1024) → ReLU → Dropout(0.4) → Linear(1024→4)
- **Optimizer:** Adam (lr=1e-4, weight_decay=1e-4)
- **Scheduler:** Cosine Annealing (T_max=15)
- **Loss:** Cross-Entropy with Label Smoothing (ε=0.1)
- **Epochs:** 15

---

## Repository Structure

```
VGGNet16-BloodCell-Classification/
├── VGGNet16_BCCD_Assignment.ipynb   # Full training pipeline (Colab)
├── README.md                         # This file
└── report/
    └── VGGNet16_BCCD_Report.docx    # IEEE-format report
```

---

## Model Weights

The trained model weights (`vgg16_bccd_final.pth`) are hosted on Google Drive due to GitHub's file size limit:

**[Download Model Weights (.pth) — Google Drive](https://drive.google.com/file/d/1RnjvnqNRFw2v0e3KDAjh2UG_yuAk13ai/view?usp=drive_link)**


---

## How to Run

### 1. Open in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/VGGNet16-BloodCell-Classification/blob/main/VGGNet16_BCCD_Assignment.ipynb)

> Replace `YOUR_USERNAME` with your GitHub username.

### 2. Set up Kaggle API
```python
import os
os.environ['KAGGLE_USERNAME'] = 'your_kaggle_username'
os.environ['KAGGLE_KEY'] = 'your_kaggle_api_key'
```

### 3. Run all cells
Go to **Runtime → Run All** and follow the prompts.

---

## Requirements

```
torch
torchvision
scikit-learn
matplotlib
seaborn
Pillow
tqdm
kaggle
```

---

## Figures

| Training Curves | Confusion Matrix |
|---|---|
| ![training](figures/training_curves.png) | ![cm](figures/confusion_matrix.png) |

| ROC Curves | Sample Predictions |
|---|---|
| ![roc](figures/roc_curves.png) | ![pred](figures/predictions.png) |

---

## References

1. Simonyan, K. & Zisserman, A. (2015). Very Deep Convolutional Networks for Large-Scale Image Recognition. ICLR.
2. Pan, S. J. & Yang, Q. (2010). A Survey on Transfer Learning. IEEE TKDE.
3. Mooney, P. T. (2018). Blood Cell Images. Kaggle.
