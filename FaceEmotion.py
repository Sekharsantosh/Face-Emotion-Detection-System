# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 10:19:32 2023

@author: SANTOSH
"""

'''
 What i wrote Here is a Blunder
import os
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop  
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model

angry_dir = os.path.join(r"C:\cars123.csv\AIML classesTHUB\happy\angry")

# Directory with happy pictures 
happy_dir = os.path.join(r"C:\cars123.csv\AIML classesTHUB\happy\happy")

# Directory with neutral pictures
neutral_dir = os.path.join(r"C:\cars123.csv\AIML classesTHUB\happy\neutral")

# Directory with sad pictures
sad_dir = os.path.join(r"C:\cars123.csv\AIML classesTHUB\happy\sad")

# Directory with surprise pictures
surprise_dir = os.path.join(r"C:\cars123.csv\AIML classesTHUB\happy\surprise")

train_angry_names = os.listdir(angry_dir)
print(train_angry_names[:5])

train_happy_names = os.listdir(happy_dir)
print(train_happy_names[:5])

train_neutral_names = os.listdir(neutral_dir)
print(train_neutral_names[:5])

train_sad_names = os.listdir(sad_dir)
print(train_sad_names[:5])

train_surprise_names = os.listdir(surprise_dir)
print(train_surprise_names[:5])

batch_size = 16



# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(r"C:\cars123.csv\AIML classesTHUB\happy",  # This is the source directory for training images
        target_size=(48, 48),  # All images will be resized to 200 x 200
        batch_size=batch_size,
        # Specify the classes explicitly
        classes = ['Angry','Happy','Neutral','Sad','Surprise'],
        # Since we use categorical_crossentropy loss, we need categorical labels
        class_mode='categorical')
target_size=(48,48)

#$input_shape = tuple(list(target_size)+[3])

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 200x 200 with 3 bytes color
    # The first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(48, 48, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    #tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    #tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    #tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    #tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a dense layer
    tf.keras.layers.Flatten(),
    # 128 neuron in the fully-connected layer
    tf.keras.layers.Dense(64, activation='relu'),
    # 5 output neurons for 5 classes with the softmax activation
    tf.keras.layers.Dense(5, activation='softmax')
])
model.summary()


# Optimizer and compilation
model.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.Adam(lr=0.001),metrics=['acc'])#RMSprop(lr=0.001)
# Total sample count
total_sample=train_generator.n
# Training
num_epochs = 5
model.fit_generator(train_generator,steps_per_epoch=int(total_sample/batch_size),
                    epochs=num_epochs,verbose=1)

from keras.models import model_from_json
from keras.models import load_model
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("modelgg.h5")
print("Saved model to disk")'''

import os
#import matplotlib.pyplot as plt


from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop  
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model

#Directory with angry images
Angry_dir= os.path.join(r"C:\cars123.csv\AIML classesTHUB\happy\angry")

# Directory with Happy images
Happy_dir= os.path.join(r"C:\cars123.csv\AIML classesTHUB\happy\happy")

# Directory with Neutral images
Neutral_dir = os.path.join(r"C:\cars123.csv\AIML classesTHUB\happy\neutral")

# Directory with Sad images
Sad_dir = os.path.join(r"C:\cars123.csv\AIML classesTHUB\happy\sad")

# Directory with Surprise images
Surprise_dir = os.path.join(r"C:\cars123.csv\AIML classesTHUB\happy\surprise")

train_Angry_names = os.listdir(Angry_dir)
print(train_Angry_names[:5])

train_Happy_names = os.listdir(Happy_dir)
print(train_Happy_names[:5])
batch_size = 16

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(r"C:\cars123.csv\AIML classesTHUB\happy",  # This is the source directory for training images
        target_size=(48, 48),  # All images will be resized to 48 x 48
        batch_size=batch_size,
        color_mode='grayscale',
        
        
        # Specify the classes explicitly
        classes = ['Angry','Happy','Neutral','Sad','Surprise'],
        # Since we use categorical_crossentropy loss, we need categorical labels
        class_mode='categorical')
target_size=(48,48)

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 48*48 with 3 bytes color

     # The first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(48, 48, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    # Flatten the results to feed into a dense layer
    tf.keras.layers.Flatten(),
    # 64 neuron in the fully-connected layer
    tf.keras.layers.Dense(64, activation='relu'),
    
    
    # 5 output neurons for 5 classes with the softmax activation
    tf.keras.layers.Dense(5,activation='softmax')
])
model.summary()

model.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.Adam(lr=0.001),
              metrics=['acc'])#RMSprop(lr=0.001)
# Total sample count

total_sample=train_generator.n
# Training
num_epochs = 25
model.fit_generator(train_generator,steps_per_epoch=int(total_sample/batch_size),
                    epochs=num_epochs,verbose=1)

# serialize model to JSON
model_json = model.to_json()
with open("modelGG.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model2GG.h5")
print("Saved model to disk")


'''"
import os: Import the Python os module, which provides a way to interact with the operating system.

from tensorflow.keras.preprocessing.image import ImageDataGenerator: Import the ImageDataGenerator class from TensorFlow's Keras module. This class is used for data augmentation and preprocessing of images.

import tensorflow as tf: Import TensorFlow, a popular machine learning framework.

from tensorflow.keras.optimizers import RMSprop: Import the RMSprop optimizer from TensorFlow's Keras module. However, you're not using it in your code.

from tensorflow.keras.models import model_from_json: Import the model_from_json function from TensorFlow's Keras module. This is used for loading a model architecture from a JSON file.

from tensorflow.keras.models import load_model: Import the load_model function from TensorFlow's Keras module. This function is used for loading a saved model from an HDF5 file.

Define directories for different emotion categories, such as "Angry," "Happy," "Neutral," "Sad," and "Surprise." These directories point to the locations of your image data for each emotion.

Use os.listdir() to list the filenames of images in the "Angry" directory and print the first five filenames.

Repeat the same for the "Happy" directory, listing the first five filenames.

batch_size = 16: Define the batch size for training. This determines how many images are processed in each iteration during training.

Create an ImageDataGenerator object named train_datagen and set the rescale factor to 1/255. This object will be used to preprocess and augment the training data.

Create a train_generator object using the flow_from_directory method of train_datagen. This generator reads images from a specified directory and generates batches of data for training. You specify various parameters such as target size, batch size, color mode, and class mode.

Define the architecture of your convolutional neural network (CNN) model using TensorFlow's Keras Sequential API. You create a sequential model and add layers to it. This model consists of convolutional layers, max-pooling layers, a flattening layer, dense (fully connected) layers, and an output layer.

Compile the model using the categorical cross-entropy loss function, the Adam optimizer with a learning rate of 0.001, and accuracy as the evaluation metric.

Calculate the total number of training samples using the train_generator.n attribute.

Define the number of training epochs as num_epochs.

Train the model using the fit_generator method, providing the training generator, the number of steps per epoch, and the number of epochs.

Serialize the model architecture to a JSON file named "modelGG.json."

Save the model weights to an HDF5 file named "model2GG.h5."

Print a message indicating that the model has been saved to disk."''
