import os

import keras

IMG_WIDTH = 1692
IMG_HEIGHT = 2245
IMG_WIDTH_REDUCED = int(IMG_WIDTH / 2)
IMG_HEIGHT_REDUCED = int(IMG_HEIGHT / 2)

data_path = os.path.join('..', '..', 'Daten_sorted')


def get_datasets():
    training, validation = keras.utils.image_dataset_from_directory(
        directory=data_path,
        label_mode='categorical',
        batch_size=8,  # Adjust to avoid OOM
        image_size=(IMG_HEIGHT_REDUCED, IMG_WIDTH_REDUCED),
        seed=42,
        subset='both',
        validation_split=0.2,
    )
    # training_normalized = training.map(lambda x, y: (x / 255.0, y))
    # validation_normalized = validation.map(lambda x, y: (x / 255.0, y))
    # return training_normalized, validation_normalized
    return training, validation
