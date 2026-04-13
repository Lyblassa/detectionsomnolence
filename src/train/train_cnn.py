# src/train/train_cnn.py
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from src.config import EPOCHS, BATCH_SIZE, AUGMENT, BEST_MODEL_PATH
from src.data.dataset import train_val_split
from src.models.cnn_eye import build_model


def make_ds(X, y, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if augment:
        def aug(x, y):
            x = tf.image.random_brightness(x, 0.1)
            x = tf.image.random_contrast(x, 0.9, 1.1)
            x = tf.image.random_flip_left_right(x)
            return x, y
        ds = ds.map(aug, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.shuffle(2048).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def main():
    # Chargement / split
    X_train, X_val, y_train, y_val = train_val_split()

    # Assure un type entier pour sklearn / TF
    y_train = y_train.astype(np.int64)
    y_val = y_val.astype(np.int64)

    # --- Class weights (CORRIGÉ) ---
    # sklearn exige un np.ndarray pour 'classes'
    classes = np.unique(y_train)          # e.g. array([0, 1])
    # Cas pathologique: si une seule classe présente (improbable mais safe)
    if classes.size < 2:
        classes = np.array([0, 1], dtype=np.int64)

    cw = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train
    )
    class_weights = {int(c): float(w) for c, w in zip(classes, cw)}

    print("[INFO] classes:", classes, "class_weights:", class_weights)

    # Modèle
    model = build_model()
    model.summary()

    # Callbacks
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(BEST_MODEL_PATH),
        save_best_only=True,
        monitor='val_accuracy',
        mode='max'
    )
    es = tf.keras.callbacks.EarlyStopping(
        patience=4,
        restore_best_weights=True,
        monitor='val_accuracy',
        mode='max'
    )

    # Datasets
    train_ds = make_ds(X_train, y_train, augment=AUGMENT)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)

    # Entraînement
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[ckpt, es],
        class_weight=class_weights
    )

    print(f"\nEntraînement terminé. Modèle sauvegardé: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    # Optionnel: réduire le bruit TF dans la console
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()