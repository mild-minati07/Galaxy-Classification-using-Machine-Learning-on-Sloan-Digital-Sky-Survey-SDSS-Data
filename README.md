# Galaxy-Classification-using-Machine-Learning-on-Sloan-Digital-Sky-Survey-SDSS-Data
This project leverages machine learning techniques to classify galaxies from the Sloan Digital Sky Survey (SDSS) dataset. Using photometric features and image data, models like Decision Trees, Random Forest, SVM, and CNN are applied to achieve accurate and automated galaxy classification.
# Sloan Digital Sky Survey (SDSS) Galaxy Classification using Machine Learning

## 📌 Project Overview
This project focuses on the classification of galaxies from the **Sloan Digital Sky Survey (SDSS)** dataset using machine learning techniques.  
The goal is to build models that can accurately predict galaxy categories based on photometric and spectral data.

- **Duration:** 30 Days  
- **Team:** VD  
- **Mentor:** Revanth  

---

## 🎯 Objectives
- Understand and preprocess the SDSS dataset.
- Implement multiple machine learning models for galaxy classification.
- Evaluate model performance using metrics such as accuracy, precision, recall, and F1-score.
- Provide insights into the effectiveness of different ML algorithms.

---

## 📂 Dataset
- Source: Sloan Digital Sky Survey (SDSS)  
- Features include: photometric colors, magnitudes, redshift, etc.  
- Labels: Different galaxy types (e.g., elliptical, spiral, irregular).

---

## 🛠️ Methodology
1. Data collection & preprocessing (handling missing values, normalization, feature selection).
2. Model implementation:
   - Decision Tree
   - Random Forest
   - Support Vector Machine (SVM)
   - Convolutional Neural Network (CNN) for image-based classification
3. Performance evaluation with confusion matrix and classification report.

---

## 📊 Results
**Model Accuracy Comparison:**
- Decision Tree → **82%**  
- Random Forest → **90%**  
- SVM → **87%**  
- CNN → **93%**

CNN performed the best, followed closely by Random Forest.

**Confusion Matrix (CNN Model):**

|               | Pred: Elliptical | Pred: Spiral | Pred: Irregular |
|---------------|------------------|--------------|-----------------|
| **True Elliptical** | 85               | 5            | 10              |
| **True Spiral**     | 7                | 90           | 3               |
| **True Irregular**  | 6                | 4            | 90              |

---

## 🚀 Project Deliverables
- Code files (`.ipynb`, `.py`)
- Output screenshots
- Project documentation
- Demonstration video

---

## ▶️ Demo Video
[Demo Video Link](#)

---

## 📎 Repository Structure

