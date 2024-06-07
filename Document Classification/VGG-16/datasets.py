import os

import keras

DATA_PATH = os.path.join('..', '..', 'Daten_sorted')

IMG_HEIGHT = 2245
IMG_WIDTH = 1692
IMG_HEIGHT_REDUCED = int(IMG_HEIGHT / 2)
IMG_WIDTH_REDUCED = int(IMG_WIDTH / 2)


def get(scale=False, data_path=DATA_PATH):
    training, validation = keras.utils.image_dataset_from_directory(
        directory=data_path,
        label_mode='categorical',
        batch_size=8,  # Adjust to avoid OOM
        image_size=(IMG_HEIGHT_REDUCED, IMG_WIDTH_REDUCED),
        seed=42,
        subset='both',
        validation_split=0.2,
    )
    if scale:
        training = training.map(lambda x, y: (x / 255.0, y))
        validation = validation.map(lambda x, y: (x / 255.0, y))
    return training, validation
