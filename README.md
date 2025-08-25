
```
# 🩺 Skin Disease Detection & Classification using CNN (ISIC 2019)

This project implements a **Convolutional Neural Network (CNN)** to detect and classify skin lesions using the **ISIC 2019 dataset**.  
It is structured for **research reproducibility**, with separate scripts for training, evaluation, and prediction.  

---

## 🔍 Project Overview
- **Goal**: Classify skin lesions into 8 categories from dermoscopic images.  
- **Dataset**: ISIC 2019 Challenge (~25k images).  
- **Approach**: Custom CNN architecture with Conv2D, Dropout, L2 regularization, and Softmax output.  
- **Training Optimization**: EarlyStopping and ReduceLROnPlateau callbacks.  
- **Result**: Achieved ~96.5% accuracy on test data.  

---

## 📂 Repository Structure
```

├── train\_model.py             # Train the CNN and save best model
├── evaluate\_model.py          # Evaluate model on test dataset
├── predict.py                 # Run predictions on new lesion images
├── sdd-reseach-cnn.ipynb      # Full pipeline in Jupyter Notebook (with outputs)
├── ImranIsicModle\_02.h5       # Saved trained CNN model
├── training\_history.pkl       # Training history (accuracy/loss curves)
├── X\_data.npy / y\_labels.npy  # Preprocessed training data & labels
├── X\_test.npy / y\_test.npy    # Preprocessed test data & labels
├── requirements.txt           # List of dependencies
└── README.md                  # Documentation

````

---

## ⚙️ Installation
1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/skin-lesion-classification.git
   cd skin-lesion-classification
````

2. Install requirements:

   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### 1️⃣ Train the Model

```bash
python train_model.py
```

* Saves the trained model as `ImranIsicModle_02.h5`
* Saves training history as `training_history.pkl`

---

### 2️⃣ Evaluate the Model

```bash
python evaluate_model.py
```

* Loads the saved model
* Prints **accuracy, loss, classification report**
* Displays **confusion matrix**

---

### 3️⃣ Predict on New Images

```bash
python predict.py --image path/to/image.jpg
```

* Preprocesses image (resize + normalize)
* Outputs **predicted class** and **confidence score**

Example:

```
Predicted Class: MEL (Confidence: 92%)
```

---

## 📊 Results

* **Validation Accuracy**: \~96.5%
* **Test Accuracy**: 96.53%
* **Test Loss**: 0.33

The model generalizes well across lesion categories.

---

## 🔮 Future Improvements

* Apply **Transfer Learning** (ResNet, EfficientNet, Vision Transformers)
* Integrate **Explainable AI (Grad-CAM)** for visual reasoning
* Extend dataset with **data augmentation** and semi-supervised learning

---

## 🙌 Credits

* **Dataset**: [ISIC 2019 Challenge](https://challenge.isic-archive.com/landing/2019/)
* **Libraries**: TensorFlow, Keras, scikit-learn, OpenCV, imbalanced-learn
* **Author**: Imran Roshan


Do you also want me to **add code snippets from your `.py` files** (like the `train`, `evaluate`, and `predict` functions) inside the README so users see the workflow immediately, or keep it clean like this?
```
