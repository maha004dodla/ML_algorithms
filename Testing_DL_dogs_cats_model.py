import warnings
warnings.filterwarnings("ignore")
from keras.models import load_model
from time import sleep



from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
from keras.models import model_from_json

import tensorflow as tf

json_file = open("model.json",'r')
loaded_model_json = json_file.read()
json_file.close()


loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("E:\ML_programs\saved_models_ML\modelDC.h5")
print("Loaded model from disk")

#loaded_model.save('model_num.hdf5')
#loaded_model=load_model('model_num.hdf5')
#To predict for different data you can use this
import numpy as np
from keras.preprocessing import image
test_image=image.load_img(r"C:\Users\MAHA LAKSHMI\desktop tags\Downloads\dog11.jpeg",target_size=(150,150))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result=loaded_model.predict(test_image)
validation_generator.class_indices
if result[0][0]>=0.5:
    print('dog')
else:
    print('cat')