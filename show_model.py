from tensorflow import keras
import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt 
import os

# sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
# tf.config.list_physical_devices('GPU')

model = keras.models.load_model('Model_RNN/')

tf.keras.utils.plot_model(
    model,
    to_file="model_rnn.png",
    show_shapes=True,
    show_dtype=True,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=False,
    dpi=96,
    layer_range=None,
    show_layer_activations=True,
)