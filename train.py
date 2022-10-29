from tensorflow import keras
import numpy as np 
import tensorflow as tf

# set gpu
# device = tf.config.list_physical_devices('GPU') 
# tf.config.experimental.set_memory_growth(device[0], True) 
# tf.config.experimental.set_virtual_device_configuration(device[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

# load data
dataset =  keras.preprocessing.image_dataset_from_directory('Foto/Akuarium/train', batch_size=6, label_mode="categorical", image_size=(1280, 720))

# for data, labels in dataset:
#     print(data)
#     print(labels);print(labels.shape)
#     break
#    print(data.shape)  # (64, 200, 200, 3)
#    print(data.dtype)  # float32
#    print(labels)  # (64,)
#    print(labels.dtype)  # int32
# dataset.eval()
# print(dataset)

# rescale input data value to normalized values
# from tensorflow.keras.layers import Rescaling

# scaler = Rescaling(scale=1.0/255)
# scaled_data = []
# for data, labels in dataset:
#     scaled_data.append(scaler(data))

# dataset.data = scaled_data

# build models
dense = keras.layers.Dense(units=16)

inputs = keras.Input(shape=(1280, 720, 3))

from tensorflow.keras import layers

# Rescale inputs
from tensorflow.keras.layers import Rescaling
x = Rescaling(scale=1.0/255)(inputs)

# Apply some convolution and pooling layers
x = layers.Conv2D(filters=32, kernel_size=(5, 5), activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(3, 3))(x)
x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(3, 3))(x)
x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(3, 3))(x)
x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)

# Apply global average pooling to get flat feature vectors
x = layers.GlobalAveragePooling2D()(x)

# Add a dense classifier on top
num_classes = 2
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

# Get result from the model
# for data, labels in dataset:
#     processed_data = model(data)
#     print(processed_data)
#     break

# Get model summary
# model.summary()

# Train model
model.compile(optimizer='Adamax', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
#model.fit(dataset, epochs=20)
history = model.fit(dataset, epochs=20)

import matplotlib.pyplot as plt
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Save model
model.save('Model/')
print("HI!")