

```markdown
# 🩺 Skin Disease Detection & Classification with CNN (ISIC 2019)

A deep learning pipeline for **automated skin lesion detection and classification** using the **ISIC 2019 dataset**.  
This repository is designed with **research reproducibility**, **scalability**, and **deployment readiness** in mind.  

---

## 🔍 Overview
- **Objective**: Build a reliable CNN model to classify dermoscopic images into **8 lesion categories**.  
- **Why**: Early and accurate detection of skin cancer can significantly improve patient outcomes.  
- **Approach**: Custom **Convolutional Neural Network (CNN)** enhanced with **Dropout**, **L2 Regularization**, and **callbacks** (EarlyStopping, ReduceLROnPlateau).  
- **Performance**: Achieved **96.5% accuracy** on the test dataset.  

---

## 📂 Repository Structure
```

├── train\_model.py             # Train CNN, apply callbacks, save best model
├── evaluate\_model.py          # Evaluate trained model (metrics + confusion matrix)
├── predict.py                 # Predict class of new lesion images
├── sdd-reseach-cnn.ipynb      # End-to-end notebook with code, outputs & visuals
├── ImranIsicModle\_02.h5       # Pretrained CNN model (best weights)
├── training\_history.pkl       # Saved training/validation curves
├── X\_data.npy / y\_labels.npy  # Preprocessed training data & labels
├── X\_test.npy / y\_test.npy    # Preprocessed test data & labels
├── requirements.txt           # Dependency list
└── README.md                  # Documentation (this file)

````

---

## ⚙️ Setup & Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/skin-lesion-classification.git
   cd skin-lesion-classification
````

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### 1️⃣ Train the Model

Train the CNN and save best model + history:

```bash
python train_model.py
```

* Outputs:

  * `ImranIsicModle_02.h5` (trained weights)
  * `training_history.pkl` (accuracy/loss curves)

---

### 2️⃣ Evaluate the Model

Run full evaluation on test set:

```bash
python evaluate_model.py
```

* Reports: **accuracy, loss, classification report**
* Plots: **confusion matrix**

---

### 3️⃣ Predict on New Images

Classify unseen skin lesion images:

```bash
python predict.py --image path/to/image.jpg
```

Example output:

```
Predicted Class: MEL (Confidence: 92.1%)
```

---

## 📊 Results

* **Validation Accuracy**: \~96.5%
* **Test Accuracy**: 96.53%
* **Test Loss**: 0.33

✔️ Model demonstrates **strong generalization** across lesion categories.
✔️ Properly handles **imbalanced dataset** via augmentation and preprocessing.

---

## 🔮 Future Directions

* **Transfer Learning**: Experiment with **EfficientNet, ResNet, ViT** for higher accuracy.
* **Explainable AI**: Integrate **Grad-CAM / SHAP** to visualize model decision-making.
* **Deployment**: Export model with TensorFlow Lite or ONNX for mobile/web applications.
* **Clinical Validation**: Extend dataset and validate with dermatologist-labeled images.

---

## 🙌 Credits

* **Dataset**: [ISIC 2019 Challenge](https://challenge.isic-archive.com/landing/2019/)
* **Core Libraries**: TensorFlow, Keras, NumPy, scikit-learn, Matplotlib, OpenCV, imbalanced-learn
* **Author**: *Imran Roshan*

---
