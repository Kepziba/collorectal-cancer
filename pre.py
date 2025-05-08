import os

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
classifierLoad = tf.keras.models.load_model('model.h5')

import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

base_dir = 'Dataset/'
catgo = os.listdir(base_dir)

img = image.load_img('Dataset/no-esophagus/0a0be03a-3c54-4a44-83dd-73b573bb580b.jpg', target_size=(150, 150))



test_image = np.expand_dims(img, axis=0)
result = classifierLoad.predict(test_image)
ind = np.argmax(result)

print(ind)
print(catgo[ind])