# Alzheimer's Disease Detection

A comprehensive deep learning repository focusing on multi-class classification for Alzheimer's Disease stage detection from Magnetic Resonance Imaging (MRI) scans. This repository explores and compares custom architectures and advanced pre-trained models across two benchmark datasets, utilizing both **PyTorch** and **TensorFlow/Keras** frameworks.

---

## 📊 Dataset Profiles

### 1. Alzheimer MRI Disease Classification Dataset (Tracks 1A, 1B)

- **Source:** [Kaggle - Alzheimer MRI Disease Classification Dataset](https://www.kaggle.com/datasets/borhanitrash/alzheimer-mri-disease-classification-dataset/data)
- **Classes:** 4-class classification tracking the clinical progression of cognitive impairment:
  1. `NonDemented`
  2. `VeryMildDemented`
  3. `MildDemented`
  4. `ModerateDemented`
- **Characteristics:** Standard grayscale structural MRI slices processed to uniform sizes, exhibiting class imbalance.

### 2. OASIS Alzheimer's Detection Dataset (Tracks 2A, 2B, 2C)

- **Source:** [Kaggle - Images OASIS](https://www.kaggle.com/datasets/ninadaithal/imagesoasis)
- **Classes:** 4-class categorization aligned with clinical standards derived from the Open Access Series of Imaging Studies (OASIS):
  1. `NonDemented`
  2. `VeryMildDemented`
  3. `MildDemented`
  4. `ModerateDemented`
- **Characteristics:** High-resolution multi-modal neurological skull-stripped/registered cross-sectional scans across a diverse demographic population.

---

## 🧠 Model Architectures & Framework Diversity

### 🛠️ Track 1A: Custom Baseline CNN (PyTorch)

- **Architecture:** \* `Conv2D` (1 $\to$ 32 filters, 3x3, Padding=1) $\to$ `ReLU` $\to$ `MaxPool2D` (2x2) $\to$ `BatchNorm2D` (32)
  - `Conv2D` (32 $\to$ 64 filters, 3x3, Padding=1) $\to$ `ReLU` $\to$ `MaxPool2D` (2x2) $\to$ `Flatten`
  - `Linear` (64 _ 32 _ 32 $\to$ 128) $\to$ `ReLU` $\to$ `Linear` (128 $\to$ 4)
- **Focus:** Establishes an empirical baseline. Incorporates a secondary training execution utilizing torchvision `v2` transforms (random rotations, affine transformations) over 15 epochs to evaluate validation performance resilience under synthetic data expansion.

### 📱 Track 1B: MobileNetV2 Transfer Learning (PyTorch)

- **Architecture:** Pre-trained `models.mobilenet_v2` serving as a frozen feature extractor (`param.requires_grad = False`).
- **Classifier Top:** Customized with a dedicated linear block: `Dropout(p=0.2)` $\to$ `Linear(last_channel, 4)`.
- **Focus:** Optimized for highly efficient, parameter-light mobile/edge deployments using inverted residual blocks.

### ⚙️ Track 2A: Deep Sequential CNN (TensorFlow / Keras)

- **Architecture:** A deeply stacked sequential pipeline:
  - 2x `Conv2D` (32 filters, 2x2) $\to$ `BatchNormalization` $\to$ `MaxPooling2D` $\to$ `Dropout(0.25)`
  - 2x `Conv2D` (64 filters, 2x2) $\to$ `BatchNormalization` $\to$ `MaxPooling2D` $\to$ `Dropout(0.25)`
  - 2x `Conv2D` (128 filters, 2x2) $\to$ `BatchNormalization` $\to$ `MaxPooling2D` $\to$ `Dropout(0.25)`
  - `Flatten` $\to$ `Dense(512, relu)` $\to$ `BatchNormalization` $\to$ `Dropout(0.5)` $\to$ `Dense(4, softmax)`
- **Focus:** Implements an expressive, deeply supervised network using early stopping regularization based on validation loss.

### 👁️ Track 2B: Vision Transformer - ViT-B/16 (PyTorch)

- **Architecture:** Standard `models.vit_b_16` utilizing multi-head self-attention mechanisms over $16\times16$ non-overlapping pixel patches.
- **Focus:** Transcends classic convolutional inductive biases, learning global pixel correlations using a customized linear head tailored to the 4 clinical states.

### ⚡ Track 2C: ResNet-18 Residual Learning (PyTorch)

- **Architecture:** Fine-tuned `models.resnet18` utilizing residual skip-connections to mitigate vanishing gradient phenomena during backpropagation.
- **Focus:** Provides robust convergence performance using pre-trained ImageNet-1k weights.

---

## 🎛️ Training & Hyperparameter Matrix

| Component / Parameter | Track 1A (CNN)       | Track 1B (MobileNetV2) | Track 2A (Keras CNN)          | Track 2B (ViT-B/16) | Track 2C (ResNet-18) |
| :-------------------- | :------------------- | :--------------------- | :---------------------------- | :------------------ | :------------------- |
| **Framework**         | PyTorch              | PyTorch                | TensorFlow / Keras            | PyTorch             | PyTorch              |
| **Optimizer**         | AdamW                | AdamW                  | Adam                          | SGD / Adam          | Adam / SGD           |
| **Learning Rate**     | 0.001                | 0.001                  | Dynamic                       | 0.001 (StepLR)      | 0.001 (StepLR)       |
| **Batch Size**        | 32                   | 32                     | 128                           | 32 / 64             | 32 / 64              |
| **Epochs**            | 10 (Base) / 15 (Aug) | 15                     | 15 (Early Stopping)           | 15                  | 15                   |
| **Loss Function**     | CrossEntropyLoss     | CrossEntropyLoss       | SparseCategoricalCrossentropy | CrossEntropyLoss    | CrossEntropyLoss     |

---

## 🛠️ Installation & Setup

1. **Clone the Repository:**

   ```bash
   git clone [https://github.com/yourusername/Alzheimer-Disease-Detection.git](https://github.com/yourusername/Alzheimer-Disease-Detection.git)
   cd Alzheimer-Disease-Detection

   ```

2. **Environment Configuration:**
   Create a virtual environment and install the required dependencies:

   ```bash
   pip install torch torchvision tensorflow keras openpyxl pandas numpy opencv-python scikit-learn matplotlib seaborn tqdm pillow
   ```
