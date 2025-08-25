# ============================
# predict.py
# ============================
# Predict class for one or more dermoscopic images using the trained model.
#
# Usage:
#   python predict.py --model artifacts/Imran_Roshan_Isic_Modle.h5 \
#                     --classes artifacts/label_classes.npy \
#                     --images /path/to/image1.jpg /path/to/image2.png
#
# Notes:
# - Preprocessing exactly mirrors training: resize to (28,28), RGB, NO normalization.

import os
import argparse
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model


IMG_SIZE = (28, 28)
CHANNELS = 3


def preprocess_image(path: str):
    img = load_img(path, target_size=IMG_SIZE)  # RGB by default
    arr = img_to_array(img).astype(np.float32)  # no scaling to match training
    return arr


def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def main():
    parser = argparse.ArgumentParser(description="Predict ISIC lesion class for input images.")
    parser.add_argument('--model', type=str, required=True, help='Path to trained .h5 model')
    parser.add_argument('--classes', type=str, required=True, help='Path to label_classes.npy')
    parser.add_argument('--images', type=str, nargs='+', required=True, help='Image file paths')
    parser.add_argument('--topk', type=int, default=3, help='Show top-K predictions')
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not os.path.isfile(args.classes):
        raise FileNotFoundError(f"Classes file not found: {args.classes}")

    classes = np.load(args.classes)
    model = load_model(args.model)

    batch = []
    valid_paths = []
    for p in args.images:
        if not os.path.isfile(p):
            print(f"Warning: file not found, skipping: {p}")
            continue
        batch.append(preprocess_image(p))
        valid_paths.append(p)

    if not batch:
        raise RuntimeError("No valid images to predict.")
    X = np.stack(batch, axis=0)  # (N, 28, 28, 3)

    logits = model.predict(X, verbose=0)  # already softmax from model
    probs = logits  # model has softmax final activation

    for i, p in enumerate(valid_paths):
        topk_idx = np.argsort(probs[i])[::-1][:args.topk]
        print(f"\nImage: {p}")
        for r, idx in enumerate(topk_idx, start=1):
            print(f"  Top-{r}: {classes[idx]}  |  prob={probs[i][idx]:.4f}")


if __name__ == "__main__":
    main()
