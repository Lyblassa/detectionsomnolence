import tensorflow as tf
from tensorflow.keras import layers, models
from src.config import IMG_SIZE, LEARNING_RATE


def build_model():
    h, w = IMG_SIZE
    inp = layers.Input(shape=(h, w, 1))

    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inp)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model