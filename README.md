# 🌿 LeafCare: Plant Leaf Health Classification Using CNN

LeafCare is a deep learning project aimed at classifying plant leaf images as either healthy or diseased. By applying Convolutional Neural Networks (CNN) and various preprocessing techniques, this system supports agricultural productivity through early detection and sustainable practices.

---

## 📝 Abstract

Modern agriculture faces serious challenges including diseases that reduce crop yield. LeafCare leverages machine learning to provide a fast, cost-effective, and accurate solution for identifying diseased leaves. This improves yield, reduces pesticide use, and contributes to ecological balance.

---

## 🛠️ Tech Stack

Language: Python

IDE: Jupyter Notebook, PyCharm

Libraries Used: TensorFlow, NumPy, Pandas, Scikit-learn, Pillow (PIL), Matplotlib, OpenCV

---

## 📂 Dataset Information

Sources:

  1. Kaggle: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset/data
  2. Mendeley: https://data.mendeley.com/datasets/hb74ynkjcn/1
     
Total Images: ~4500

Classes: Healthy (2278) and Diseased (2225)

Plant Types: Mango, Guava, Lemon, Basil, Pomegranate, Poplar, and others

---

## 🔁 Project Workflow

1. Data Loading & Preprocessing
2. Data Augmentation
3. CNN Model Training
4. Model Evaluation
5. Prediction & Results Display

---

## 🧠 CNN Architecture

-  Conv2D Layers: Three layers (32, 64, 128 filters) with ReLU activation
-  MaxPooling2D: After each Conv layer
-  Flatten Layer
-  Dense Layer: 128 neurons + Dropout(0.5)
-  Output Layer: Sigmoid activation (1 neuron for binary classification)

---

## ⚙️ Training Configuration

-  Loss Function: Binary Crossentropy
-  Optimizer: Adam
-  Metric: Accuracy
-  Epochs: 50
-  Augmentation Techniques: Rotation, Zoom, Shear, Horizontal Flip, etc.

---

## 🔍 Prediction Flow

-  Load image from external folder
-  Resize to 128x128
-  Normalize pixel values to [0, 1]
-  Run prediction using trained CNN
-  Classify as "Healthy" or "Diseased" based on score threshold (0.5)

---

## 📈 Results & Observations

-  High accuracy achieved with augmented datasets
-  Robust generalization on unseen external images
-  Realistic prediction for random test images from the internet

---

## 🔮 Future Scope

-  Real-time mobile/web leaf scanner
-  Multi-class disease classification
-  Grad-CAM visual explanations
-  Integration into sustainable agriculture toolkits

--- 

## 📚 References

Machine Learning for Plant Disease Detection – IEEE: https://ieeexplore.ieee.org/document/8698004
Kaggle Plant Pathology 2021 - FGVC8: https://www.kaggle.com/competitions/plant-pathology-2021-fgvc8

---
