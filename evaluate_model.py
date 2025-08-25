# ============================
# evaluate_model.py
# ============================
# Evaluate a trained model: test metrics, classification report, confusion matrix,
# and render training curves if training_history.pkl is available.
#
# Usage:
#   python evaluate_model.py --artifacts_dir artifacts --show_plots --save_plots

import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model


def plot_training_curves(history_dict: dict, save_dir: str = None, show_plots: bool = True):
    tr_acc = history_dict.get('accuracy', [])
    tr_loss = history_dict.get('loss', [])
    val_acc = history_dict.get('val_accuracy', [])
    val_loss = history_dict.get('val_loss', [])

    if not tr_acc or not tr_loss:
        print("History does not contain accuracy/loss keys; skipping curve plots.")
        return

    epochs = np.arange(1, len(tr_acc) + 1)

    # Plot Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, tr_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    # Highlight best val loss epoch
    idx_loss = int(np.argmin(val_loss))
    plt.scatter(epochs[idx_loss], val_loss[idx_loss], s=120, label=f'Best Epoch = {idx_loss + 1}')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    if save_dir:
        pth = os.path.join(save_dir, 'training_loss.png')
        plt.savefig(pth, bbox_inches='tight'); print("Saved:", pth)
    if show_plots:
        plt.show()
    plt.close()

    # Plot Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, tr_acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    idx_acc = int(np.argmax(val_acc))
    plt.scatter(epochs[idx_acc], val_acc[idx_acc], s=120, label=f'Best Epoch = {idx_acc + 1}')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
    if save_dir:
        pth = os.path.join(save_dir, 'training_accuracy.png')
        plt.savefig(pth, bbox_inches='tight'); print("Saved:", pth)
    if show_plots:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained ISIC model.")
    parser.add_argument('--artifacts_dir', type=str, default='artifacts', help='Directory containing saved artifacts')
    parser.add_argument('--model_path', type=str, default=None, help='Optional explicit model path (.h5)')
    parser.add_argument('--show_plots', action='store_true', help='Show plots interactively')
    parser.add_argument('--save_plots', action='store_true', help='Save plots to artifacts_dir')
    args = parser.parse_args()

    model_path = args.model_path or os.path.join(args.artifacts_dir, 'Imran_Roshan_Isic_Modle.h5')
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Load arrays
    X_test_path = os.path.join(args.artifacts_dir, 'X_test.npy')
    y_test_path = os.path.join(args.artifacts_dir, 'y_test.npy')
    if not (os.path.isfile(X_test_path) and os.path.isfile(y_test_path)):
        raise FileNotFoundError("X_test.npy / y_test.npy not found in artifacts_dir. Run train_model.py first.")

    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)
    print("Loaded X_test:", X_test.shape, "y_test:", y_test.shape)

    # Load classes
    label_classes_path = os.path.join(args.artifacts_dir, 'label_classes.npy')
    if not os.path.isfile(label_classes_path):
        raise FileNotFoundError("label_classes.npy not found. Run train_model.py to save it.")
    classes = np.load(label_classes_path)
    print("Classes:", classes.tolist())

    # Load model
    model = load_model(model_path)

    # Evaluate
    loss, acc = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Accuracy: {acc * 100:.2f}%")
    print(f"Test Loss: {loss:.4f}")

    # Predictions -> reports
    y_true = np.argmax(y_test, axis=1)
    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=classes))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap='Blues', xticks_rotation='vertical')
    if args.save_plots:
        cm_path = os.path.join(args.artifacts_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, bbox_inches='tight'); print("Saved:", cm_path)
    if args.show_plots:
        plt.show()
    plt.close()

    # Training curves (optional)
    hist_path = os.path.join(args.artifacts_dir, 'training_history.pkl')
    if os.path.isfile(hist_path):
        with open(hist_path, 'rb') as f:
            history_dict = pickle.load(f)
        plot_training_curves(
            history_dict,
            save_dir=args.artifacts_dir if args.save_plots else None,
            show_plots=args.show_plots
        )
    else:
        print("training_history.pkl not found; skipping curve plots.")


if __name__ == "__main__":
    main()
