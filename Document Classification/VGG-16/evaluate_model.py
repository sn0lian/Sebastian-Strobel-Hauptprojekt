import os

import keras

import datasets

model = keras.models.load_model(os.path.join('models', 'model_scaled_lr_001.h5'))
_, validation = datasets.get()
print(model.evaluate(validation))
