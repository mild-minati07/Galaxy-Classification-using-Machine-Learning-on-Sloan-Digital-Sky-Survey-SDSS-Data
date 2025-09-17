"""
Galaxy Classification using Machine Learning and CNN
Dataset: Sloan Digital Sky Survey (SDSS)
Author: VD
Mentor: Revanth
"""

# ---------------- Libraries ----------------
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models

# ---------------- Dummy CSV Data ----------------
# Small dataset with 3 classes (spiral, elliptical, irregular)
if not os.path.exists("photometry.csv"):
    df = pd.DataFrame({
        "u": np.random.rand(100),
        "g": np.random.rand(100),
        "r": np.random.rand(100),
        "i": np.random.rand(100),
        "z": np.random.rand(100),
        "label": np.random.choice(["spiral", "elliptical", "irregular"], 100)
    })
    df.to_csv("photometry.csv", index=False)
    print("[INFO] Dummy photometry.csv created!")

# ---------------- Helper: Confusion Matrix ----------------
def plot_cm(y_true, y_pred, labels, out_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(out_path)
    print(f"[INFO] Confusion matrix saved at {out_path}")

# ---------------- Feature-based ML ----------------
def run_feature_models():
    df = pd.read_csv("photometry.csv")
    X = df.drop("label", axis=1)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42)
    }

    for name, model in models.items():
        pipe = make_pipeline(StandardScaler(), model)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        print(f"\n=== {name} ===")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        plot_cm(y_test, y_pred, labels=y.unique(), out_path=f"cm_{name.replace(' ', '_')}.png")

# ---------------- CNN (image classification) ----------------
def run_cnn():
    # Create dummy image dataset if not exists
    if not os.path.exists("images"):
        os.makedirs("images/spiral", exist_ok=True)
        os.makedirs("images/elliptical", exist_ok=True)
        os.makedirs("images/irregular", exist_ok=True)

        # Save dummy images (random noise)
        for i in range(10):
            plt.imsave(f"images/spiral/img_{i}.png", np.random.rand(64,64,3))
            plt.imsave(f"images/elliptical/img_{i}.png", np.random.rand(64,64,3))
            plt.imsave(f"images/irregular/img_{i}.png", np.random.rand(64,64,3))
        print("[INFO] Dummy images created!")

    img_size = 64
    batch_size = 4

    train_ds = tf.keras.utils.image_dataset_from_directory(
        "images",
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(img_size, img_size),
        batch_size=batch_size,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        "images",
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(img_size, img_size),
        batch_size=batch_size,
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print("[INFO] CNN classes:", class_names)

    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=(img_size, img_size, 3)),
        layers.Conv2D(16, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    history = model.fit(train_ds, validation_data=val_ds, epochs=3)

    # Evaluate
    y_true, y_pred = [], []
    for images, labels in val_ds:
        preds = model.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    plot_cm(y_true, y_pred, labels=class_names, out_path="cm_cnn.png")

    model.save("cnn_model.h5")
    print("[INFO] CNN model saved as cnn_model.h5")

# ---------------- Run everything ----------------
if __name__ == "__main__":
    print("\n===== FEATURE MODELS =====")
    run_feature_models()

    print("\n===== CNN MODEL =====")
    run_cnn()
