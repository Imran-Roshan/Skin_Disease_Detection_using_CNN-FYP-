# ============================
# train_model.py
# ============================
# Train the ISIC-2019 CNN with strict reproducibility, class balancing, and Nadam + LR scheduling.
# Saves:
#   - artifacts/Imran_Roshan_Isic_Modle.h5
#   - artifacts/training_history.pkl
#   - artifacts/label_classes.npy
#   - artifacts/X_train.npy, artifacts/y_train.npy
#   - artifacts/X_test.npy,  artifacts/y_test.npy
#
# Usage:
#   python train_model.py --dataset_path /path/to/ISIC_2019 --output_dir artifacts
# Dataset folder must contain class subfolders: MEL, VASC, SCC, DF, NV, BKL, BCC, AK
# and optionally the ISIC_2019_Training_GroundTruth.csv (not required to run).

import os
import sys
import random
import argparse
import pickle
from collections import Counter

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from imblearn.over_sampling import RandomOverSampler


SEED = 42
CLASSES = ['MEL', 'VASC', 'SCC', 'DF', 'NV', 'BKL', 'BCC', 'AK']
IMG_SIZE = (28, 28)  # keep identical to your notebook
CHANNELS = 3


def set_reproducible():
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')  # single GPU for determinism
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print("Using only one GPU:", gpus[0])
        except RuntimeError as e:
            print(e)
    print("Python:", sys.version)
    print("TensorFlow:", tf.__version__)
    print("GPU devices:", tf.config.list_physical_devices('GPU'))


def load_and_resize_images(image_paths, target_size=IMG_SIZE):
    images = []
    for p in image_paths:
        img = load_img(p, target_size=target_size)  # RGB by default
        images.append(img_to_array(img))
    return np.array(images, dtype=np.float32)  # NOTE: no normalization to match your notebook


def load_dataset(dataset_path: str):
    all_images, all_labels = [], []
    for cname in CLASSES:
        cdir = os.path.join(dataset_path, cname)
        if not os.path.isdir(cdir):
            raise FileNotFoundError(f"Expected class folder not found: {cdir}")
        files = [os.path.join(cdir, f) for f in os.listdir(cdir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not files:
            raise RuntimeError(f"No images found in {cdir}")
        imgs = load_and_resize_images(files, target_size=IMG_SIZE)
        all_images.append(imgs)
        all_labels.extend([cname] * len(imgs))
        print(f"Loaded {len(imgs):6d} images from {cname}")

    X_all = np.concatenate(all_images, axis=0)
    y_all = np.array(all_labels)
    print("Shape of all images:", X_all.shape)
    print("Shape of all labels:", y_all.shape)
    return X_all, y_all


def balance_with_ros(X: np.ndarray, y_one_hot: np.ndarray):
    X_flat = X.reshape(X.shape[0], -1)
    ros = RandomOverSampler(random_state=SEED)
    X_res, y_res = ros.fit_resample(X_flat, y_one_hot)
    print("Class distribution after oversampling:", Counter(np.argmax(y_res, axis=1)))
    return X_res, y_res


def build_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], CHANNELS)):
    wd = 0.01  # L2 like your notebook
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(wd), input_shape=input_shape),
        MaxPooling2D(), BatchNormalization(),

        Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(wd)),
        MaxPooling2D(), BatchNormalization(),

        Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(wd)),
        MaxPooling2D(), BatchNormalization(),

        Flatten(),
        Dropout(0.5),

        Dense(256, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(wd)),
        BatchNormalization(),

        Dense(128, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(wd)),
        BatchNormalization(),

        Dense(64, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(wd)),
        BatchNormalization(),

        Dense(len(CLASSES), activation='softmax')
    ])
    optimizer = Nadam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    parser = argparse.ArgumentParser(description="Train CNN on ISIC-2019 (balanced, reproducible).")
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Root path containing class folders: MEL, VASC, SCC, DF, NV, BKL, BCC, AK')
    parser.add_argument('--output_dir', type=str, default='artifacts', help='Where to save model & artifacts')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_reproducible()

    # 1) Load raw images
    X_all, y_all = load_dataset(args.dataset_path)

    # 2) Encode labels -> one-hot
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_all)
    y_one_hot = to_categorical(y_encoded, num_classes=len(CLASSES))
    np.save(os.path.join(args.output_dir, 'label_classes.npy'), le.classes_)
    print("Saved label classes:", le.classes_)

    # 3) Balance with RandomOverSampler
    X_res, y_res = balance_with_ros(X_all, y_one_hot)

    # 4) Split train/test (no stratify, to match notebook)
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=SEED
    )

    # 5) Reshape back to images
    X_train = X_train.reshape((-1, IMG_SIZE[0], IMG_SIZE[1], CHANNELS)).astype(np.float32)
    X_test = X_test.reshape((-1, IMG_SIZE[0], IMG_SIZE[1], CHANNELS)).astype(np.float32)

    # 6) Build & train model
    model = build_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], CHANNELS))
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-5, verbose=1),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # 7) Save artifacts
    model_path = os.path.join(args.output_dir, 'Imran_Roshan_Isic_Modle.h5')  # keeping original name
    model.save(model_path)
    print("Saved model to:", model_path)

    with open(os.path.join(args.output_dir, 'training_history.pkl'), 'wb') as f:
        pickle.dump(history.history, f)
    print("Saved training history (pickle).")

    np.save(os.path.join(args.output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(args.output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(args.output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(args.output_dir, 'y_test.npy'), y_test)
    print("Saved train/test arrays.")

    # 8) Quick test metrics
    loss, acc = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Accuracy: {acc * 100:.2f}%")
    print(f"Test Loss: {loss:.4f}")


if __name__ == "__main__":
    main()
