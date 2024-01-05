# -*- coding: utf-8 -*-
"""extra_model-ec7

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11ngxf5TP8VNZ5cCTsF6Pr7v39aFgpfCl
"""

import numpy as np
import os
import tensorflow as tf

print(tf.__version__)


X = np.linspace(-10.0, 10.0, 400)

B = 490 * X - 900 + np.random.normal(0, 5, len(X))

tf.keras.backend.clear_session()
model_C = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1], name='Single')
])

model_C.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.mean_squared_error)
print(model_C.summary())

test_values_D = np.linspace(-10.0, 10.0, 12).reshape((-1, 1))
predictions_D = model_C.predict(test_values_D).flatten()
print("Predictions:", predictions_D)

export_path_E = './model-ec7/1/'
tf.saved_model.save(model_C, os.path.join('./', export_path_E))

weights_F, biases_F = model_C.layers[0].get_weights()
print(f"W: {weights_F.flatten()[0]}")
print(f" b: {biases_F[0]}")

model_C.fit(X, B, epochs=100)