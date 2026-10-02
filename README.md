# 🍅 Tomato Leaf AI — Disease Detection & Diagnostic System

An end-to-end, deep-learning-powered plant pathology system and interactive Streamlit web application for identifying foliar diseases on tomato plants.

---

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Dataset Structure & Zero-Leakage Split](#dataset-structure--zero-leakage-split)
- [Model Architectures](#model-architectures)
  - [Model 1: Custom Deep CNN](#model-1-custom-deep-cnn)
  - [Model 2: MobileNetV2 Transfer Learning](#model-2-mobilenetv2-transfer-learning)
- [Project Folder Structure](#project-folder-structure)
- [Windows Setup & Installation](#windows-setup--installation)
- [Training & Artifact Generation](#training--artifact-generation)
- [Running the Streamlit Application](#running-the-streamlit-application)
- [Prediction & Explainability Workflow (Grad-CAM)](#prediction--explainability-workflow-grad-cam)
- [Performance & Evaluation](#performance--evaluation)
- [Field Limitations & Critical Notice](#field-limitations--critical-notice)

---

## 🌟 Project Overview
Tomato crops suffer from destructive bacterial, fungal, and viral infections that significantly reduce agricultural yields and farm profitability. Identifying diseases at the earliest symptom onset allows growers to act quickly and apply targeted, eco-friendly treatments rather than indiscriminate chemical sprays.

This project delivers a complete, locally executable deep learning system:
1. **Dynamic Dataset Discovery & Audit**: Automatically detects image datasets across different folder layouts, checks image integrity, and identifies duplicate files.
2. **Scientifically Valid Grouped Stratification**: Extracts unique source-leaf UUIDs and partitions splits using `StratifiedGroupKFold` to guarantee zero data leakage between training and validation.
3. **Dual Model Comparison**: Trains and benchmarks a Custom Deep CNN (with Grad-CAM explainability hooks) against an ImageNet-pretrained MobileNetV2 network, selecting the superior model based strictly on validation accuracy.
4. **Interactive Streamlit Web Dashboard**: A dark-themed agricultural AI dashboard that reads real exported artifacts, offers real-time leaf image diagnosis, plots confidence distributions, and overlays Grad-CAM attention heatmaps.

---

## 🌿 Recognised Tomato Leaf Conditions (11 Classes)
1. **Bacterial Spot** (`Xanthomonas campestris pv. vesicatoria`)
2. **Early Blight** (`Alternaria solani`)
3. **Late Blight** (`Phytophthora infestans`)
4. **Leaf Mold** (`Passalora fulva`)
5. **Powdery Mildew** (`Oidium neolycopersici` / `Leveillula taurica`)
6. **Septoria Leaf Spot** (`Septoria lycopersici`)
7. **Spider Mites / Two-spotted Spider Mite** (`Tetranychus urticae`)
8. **Target Spot** (`Corynespora cassiicola`)
9. **Tomato Mosaic Virus** (ToMV)
10. **Tomato Yellow Leaf Curl Virus** (TYLCV)
11. **Healthy Leaf** (No pathogenic symptoms)

---

## 📊 Dataset Structure & Zero-Leakage Split

The system dynamically detects any of the following directory structures:
```
train/                     valid/ (or val/)
  ├── class_1/               ├── class_1/
  ├── class_2/               ├── class_2/
  └── ...                    └── ...
```
OR:
```
tomato/
  ├── train/
  └── val/
```

### Data Audit & Leakage Prevention
- **Image Integrity**: Scans all images with PIL to detect corrupted or unreadable files.
- **Exact Duplicate Detection**: Computes file hashes to remove byte-identical duplicate images within training and across splits.
- **Source Leaf Grouping**: In PlantVillage, multiple augmented photos originate from the exact same physical leaf (`<uuid>___...`). The pipeline extracts this UUID and enforces `StratifiedGroupKFold`:
  $$\text{Train Groups} \cap \text{Validation Groups} = \emptyset$$
  $$\text{Train Groups} \cap \text{Test Groups} = \emptyset$$
  $$\text{Validation Groups} \cap \text{Test Groups} = \emptyset$$
  This ensures that augmented look-alikes never artificially inflate validation or test scores.

---

## 🧠 Model Architectures

### Model 1: Custom Deep CNN
- **Input Resolution**: $128 \times 128 \times 3$ RGB tensor
- **Preprocessing**: `Rescaling(1./255)` layer built directly into the computational graph
- **Feature Extractor**:
  - `Conv2D(32, 3x3)` $\rightarrow$ `BatchNormalization` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPooling2D(2x2)`
  - `Conv2D(64, 3x3)` $\rightarrow$ `BatchNormalization` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPooling2D(2x2)`
  - `Conv2D(128, 3x3)` $\rightarrow$ `BatchNormalization` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPooling2D(2x2)`
  - `Conv2D(256, 3x3)` $\rightarrow$ `BatchNormalization` $\rightarrow$ `ReLU(name="last_conv_act")`
- **Classification Head**:
  - `GlobalAveragePooling2D()`
  - `Dense(128, activation='relu')`
  - `Dropout(0.5)`
  - `Dense(NUM_CLASSES, activation='softmax')`
- **Explainability**: The final convolutional activation layer is explicitly named `last_conv_act` for Gradient-weighted Class Activation Mapping (Grad-CAM).

### Model 2: MobileNetV2 Transfer Learning
- **Backbone**: MobileNetV2 pretrained on ImageNet
- **Stage 1 (Head Training)**: Frozen backbone, training top classification head with `Adam(1e-3)`
- **Stage 2 (Fine-Tuning)**: Unfreezing top backbone layers (keeping `BatchNormalization` layers frozen) with low learning rate `Adam(1e-5)`

---

## 📁 Project Folder Structure

```
TomatoLeafAI/
│
├── app.py                             # Full-featured Streamlit AI web application
├── train_and_export.py                # Standalone training, audit & export pipeline
├── crop_disease_detection_cnn.ipynb   # Complete Jupyter Notebook (all Kaggle paths removed)
├── requirements.txt                   # Pinned Python package dependencies
├── README.md                          # Documentation & instructions
│
├── train/                             # Training image subdirectories per class
│   ├── Bacterial_spot/
│   ├── Early_blight/
│   └── ...
│
├── valid/                             # Validation / Held-out test image subdirectories
│   ├── Bacterial_spot/
│   ├── Early_blight/
│   └── ...
│
└── project/
    ├── model/
    │   ├── model.keras                # Exported trained Keras model
    │   ├── class_names.json           # Class mapping and Grad-CAM layer metadata
    │   └── metrics.json               # Genuine evaluation metrics and training curves
    │
    └── data/
        ├── dataset.csv                # Complete audited dataset manifest
        └── audit.json                 # Image counts, dimensions, and duplicate statistics
```

---

## 💻 Windows Setup & Installation

Open **PowerShell** or **Command Prompt** in the project directory:

### 1. Create and Activate a Virtual Environment
```powershell
python -m venv venv
venv\Scripts\activate
```

### 2. Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Verify Environment
```powershell
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import streamlit as st; print('Streamlit:', st.__version__)"
```

---

## 🚀 Training & Artifact Generation

### Option A: Via Python Command Line (Fast & Automated)
Run the automated pipeline script:
```powershell
python train_and_export.py
```
This script will:
1. Scan and audit the images and write `data/audit.json` and `data/dataset.csv`.
2. Perform the leak-free `StratifiedGroupKFold` split.
3. Train the Custom CNN and MobileNetV2 models.
4. Select the best model based on validation accuracy.
5. Evaluate on the untouched held-out test set.
6. Export `model.keras`, `class_names.json`, and `metrics.json`.
7. Verify inference and Grad-CAM on a sample test leaf.

### Option B: Via Jupyter Notebook
```powershell
jupyter notebook
```
Open `crop_disease_detection_cnn.ipynb` and select **Cell $\rightarrow$ Run All**.

---

## 🌐 Running the Streamlit Application

Start the web application with a single command:
```powershell
streamlit run app.py
```
Open your web browser and navigate to:
```
http://localhost:8501
```

### Streamlit Navigation Pages:
- 🏠 **Home**: Overview metrics (Total Images, Classes, Test Accuracy, Macro F1), Problem statement, Approach, and detected conditions list.
- 📊 **Dataset**: Interactive Plotly class distribution bar chart, split distribution donut chart, manifest data preview, and full audit logs.
- 🔍 **Prediction**: Upload any leaf image (`.jpg`, `.jpeg`, `.png`), view real-time disease diagnosis, model confidence percentage, Top-3 probability bars, and Grad-CAM attention heatmap overlay.
- 📈 **Model Performance**: Interactive confusion matrix heatmap, per-class precision/recall/F1 metrics, epoch training and validation curves, and model comparison.
- ℹ️ **About**: Architectural diagrams, training methodology, Grad-CAM explanation, limitations, and tech stack.

---

## 🔍 Prediction & Grad-CAM Workflow

```
USER UPLOADS IMAGE (JPG / PNG)
              ↓
    PIL Image (RGB Conversion)
              ↓
  Bilinear Resize to 128 x 128
              ↓
    Rescaling Layer (1 / 255)
              ↓
    Convolutional Feature Extraction
              ↓
  last_conv_act Feature Map Tap
              ↓
    Global Average Pooling + Dense Head
              ↓
  Class Probabilities & Disease Classification
        ┌─────┴────────────────┐
        ↓                      ↓
  Predicted Class      Grad-CAM Heatmap
  + Confidence (%)     Overlaid on Leaf Image
```

---

## ⚠️ Field Limitations & Critical Notice

The underlying PlantVillage dataset was acquired in controlled greenhouse and laboratory conditions (individual leaf, plain neutral background, steady illumination). 

When evaluating photos taken in open-air fields, predictions may be impacted by:
- Uneven outdoor sunlight, harsh shadows, and reflections
- Complex backgrounds (soil, weeds, hands, plant stakes)
- Leaves showing multiple simultaneous infections or secondary nutritional chlorosis
- Motion blur or camera lens distortion

**Intended Use**: This tool is an agronomical decision-support aid and should be combined with professional visual field scouting.
