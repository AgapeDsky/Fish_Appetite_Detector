from tensorflow import keras
import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt 
import os

# sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
tf.config.list_physical_devices('GPU')

# model = keras.models.load_model('Model/')

# dataset =  keras.preprocessing.image_dataset_from_directory('Foto/', batch_size=12, label_mode="categorical", image_size=(270, 480))

# # img = keras.preprocessing.image.load_img('Foto/ikan_lapar/ikan_lapar_168.jpg', target_size=(270, 480))
# # img = keras.preprocessing.image.load_img('Foto/ikan_kenyang/ikan_kenyang_107.jpg', target_size=(270, 480))
# img = keras.preprocessing.image.load_img('20220924_162655_000/20220924_162655_000.jpg', target_size=(270, 480))
# img.show()

# # predict singular
# img_array = keras.preprocessing.image.img_to_array(img)
# img_array = tf.expand_dims(img_array, 0)  # Create batch axis

# predictions = model.predict(img_array)
# print("ikan ", predictions[0][1]*100, " % lapar")


# # evaluate
# # loss = model.evaluate(dataset, batch_size=12)