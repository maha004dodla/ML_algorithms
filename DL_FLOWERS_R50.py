from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop  
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model

daisy_dir = os.path.join(r'F:\flowers\daisy')

# Directory with dandelion pictures 
dandelion_dir = os.path.join(r'F:\flowers\dandelion')

# Directory with rose pictures
rose_dir = os.path.join(r'F:\flowers\rose')

# Directory with sunflower pictures
sunflower_dir = os.path.join(r'F:\flowers\sunflower')

# Directory with tulip pictures
tulip_dir = os.path.join(r'F:\flowers\tulip')

train_daisy_names = os.listdir(daisy_dir)
print(train_daisy_names[:5])

train_rose_names = os.listdir(rose_dir)
print(train_rose_names[:5])

batch_size = 128



# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(r'F:\flowers',  # This is the source directory for training images
        target_size=(200, 200),  # All images will be resized to 200 x 200
        batch_size=batch_size,
        # Specify the classes explicitly
        classes = ['daisy','dandelion','rose','sunflower','tulip'],
        # Since we use categorical_crossentropy loss, we need categorical labels
        class_mode='categorical')
target_size=(200,200)



dnn_model =  tf.keras.models.Sequential()
imported_model= tf.keras.applications.ResNet50(include_top=False,input_shape=(180,180,3),
                                               pooling='avg',classes=5,
                                               weights='imagenet')
for layer in imported_model.layers:
    layer.trainable=False
from tensorflow.python.keras.layers import Dense, Flatten

dnn_model.add(imported_model)
dnn_model.add(Flatten())
dnn_model.add(Dense(512, activation='relu'))
dnn_model.add(Dense(5, activation='softmax'))

# Optimizer and compilation
dnn_model.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.Adam(lr=0.4),metrics=['acc'])#RMSprop(lr=0.001)
# Now you can train your custom model with additional convolutional layers
total_sample=train_generator.n
# Training
num_epochs = 35
dnn_model.fit_generator(
        train_generator, 
        steps_per_epoch=int(total_sample/batch_size),  
        epochs=num_epochs,
        verbose=1)