from tensorflow import keras
import numpy as np 
import tensorflow as tf
import cv2

IMG_SIZE = (1080, 1920)

def load_video(path, max_frames=0, resize=(200,200)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (540,960))
            frame = frame[200:650,50:, [2, 1, 0]]
            frame = cv2.resize(frame, resize)
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)

# build model
###################################################################################################
from keras import layers
inputs = keras.Input(shape=(10,200,200,3))
x = layers.Rescaling(scale=1.0/255.0)(inputs)
x = layers.Conv3D(filters=32, kernel_size=(3,3,3), activation='relu')(x)
x = layers.MaxPooling3D(pool_size=(1,3,3))(x)
x = layers.Conv3D(filters=32, kernel_size=(3,3,3), activation='relu')(x)
x = layers.MaxPooling3D(pool_size=(1,3,3))(x)
x = layers.Conv3D(filters=32, kernel_size=(3,3,3), activation='relu')(x)
x = layers.GlobalAveragePooling3D()(x)

num_classes = 2
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
###################################################################################################

# create dataset
###################################################################################################
import os
directory_lapar = 'Video/1 detik/Lapar/'
directory_kenyang = 'Video/1 detik/Kenyang/'
data = []
label = []

for filename in os.listdir(directory_lapar):
    f = os.path.join(directory_lapar, filename)
    # checking if it is a file
    if os.path.isfile(f):
        data.append(tf.convert_to_tensor(load_video(f, max_frames=10), dtype='float'))
        label.append([1, 0])
        print("test")

print("1 clear")
# data = tf.convert_to_tensor(data, dtype='float')

for filename in os.listdir(directory_kenyang):
    f = os.path.join(directory_kenyang, filename)
    # checking if it is a file
    if os.path.isfile(f):
        data.append(tf.convert_to_tensor(load_video(f, max_frames=10), dtype='float'))
        label.append([0, 1])
        print("test2")

label = tf.convert_to_tensor(label, dtype='float')

dataset = tf.data.Dataset.from_tensor_slices((data, label))
dataset = dataset.batch(1)
dataset = dataset.shuffle(30, reshuffle_each_iteration=True)
# print(dataset)
# i = 0
# for ele in dataset:
#     if i == 5:
#         print(ele[0])
#         print("============")
#         print(ele[1])
#     i = i+1
###################################################################################################

# train model
###################################################################################################
model.summary()
model.fit(dataset, epochs=30)