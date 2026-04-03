"""
Train a CNN on Fruits-360 (or any folder layout: Training/ and Test/ with class subfolders).

Usage:
  python main.py --train-dir ./data/Training --test-dir ./data/Test --epochs 10

Dataset: https://www.kaggle.com/datasets/moltean/fruits
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

CLASSES = [
    "Apple",
    "Banana",
    "Cherry",
    "Lemon",
    "Onion",
    "Plum",
    "Potato",
    "Strawberry",
    "Tomato",
    "Walnut",
]


def build_model(input_shape: tuple[int, int, int] = (128, 128, 3), num_classes: int = 10) -> Sequential:
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(4096, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Fruits-360 CNN trainer")
    parser.add_argument("--train-dir", type=str, required=True, help="Path to Training folder")
    parser.add_argument("--test-dir", type=str, required=True, help="Path to Test folder")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="fruits_model_weights",
        help="Prefix path for ModelCheckpoint weights",
    )
    parser.add_argument("--save-model", type=str, default="fruits_saved_model", help="Keras SavedModel directory")
    args = parser.parse_args()

    train_dir = os.path.abspath(args.train_dir)
    test_dir = os.path.abspath(args.test_dir)
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    igen = ImageDataGenerator(rescale=1.0 / 255.0)
    train_gen = igen.flow_from_directory(
        directory=train_dir,
        target_size=(128, 128),
        classes=CLASSES,
        class_mode="sparse",
        batch_size=args.batch_size,
    )
    test_gen = igen.flow_from_directory(
        directory=test_dir,
        target_size=(128, 128),
        classes=CLASSES,
        class_mode="sparse",
        batch_size=args.batch_size,
    )

    model = build_model(num_classes=len(CLASSES))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    cp_callback = ModelCheckpoint(
        args.checkpoint,
        save_weights_only=True,
        verbose=1,
    )

    history = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=test_gen,
        callbacks=[cp_callback],
    )

    model.save(args.save_model)
    print(f"Saved model to {args.save_model}")

    hist = history.history
    plt.figure()
    plt.plot(hist["accuracy"], label="train accuracy")
    plt.plot(hist["val_accuracy"], label="val accuracy")
    plt.title("Accuracy")
    plt.xlabel("epoch")
    plt.legend()
    plt.savefig("accuracy_curve.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(hist["loss"], label="train loss")
    plt.plot(hist["val_loss"], label="val loss")
    plt.title("Loss")
    plt.xlabel("epoch")
    plt.legend()
    plt.savefig("loss_curve.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
