import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import cv2
import numpy as np
import matplotlib.image as mpimg

# Load the dataset
data_df = pd.read_csv('Sim_Data/IMG/driving_log.csv', names=['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed'])

# split the data into training and validation
train_samples, validation_samples = train_test_split(data_df, test_size=0.2)

def batch_generator(data, batch_size=32, is_training=True):
    num_samples = len(data)
    while True: 
        data = data.sample(frac=1)  # reshuffle data at each epoch
        for offset in range(0, num_samples, batch_size):
            batch_samples = data[offset:offset+batch_size]

            images = []
            steerings = []
            for _, batch_sample in batch_samples.iterrows():
                

                img_name = batch_sample[0].split('/')[-1]
                img_name = img_name.replace('\\', '/')
                img = mpimg.imread(img_name)  # read the image
                img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)  # convert to YUV color space
                img = cv2.resize(img, (200, 66))  # resize
                img = img / 255.0  # normalize
                images.append(img)
                
                steering = float(batch_sample[3])  # get the steering angle
                steerings.append(steering)

                # Data augmentation: flip the image and invert the steering angle
                if is_training and np.random.rand() < 0.6:
                    img = cv2.flip(img, 1)
                    steering = -steering
                    images.append(img)
                    steerings.append(steering)

            inputs = np.array(images)
            targets = np.array(steerings)
            yield inputs, targets


def build_model():
    model = Sequential()
    
    # Normalization layer
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(66, 200, 3)))
    
    # Convolutional layers
    model.add(Conv2D(16, (5, 5), activation='elu', strides=(1, 1)))
    model.add(Conv2D(32, (3, 3), activation='elu', strides=(1, 1)))
    model.add(Conv2D(64, (3, 3), activation='elu', strides=(1, 1)))
    
    # Max pooling layers
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
    
    model.add(Dropout(0.2))  # Dropout layer to avoid overfitting
    model.add(Flatten())
    
    # Fully connected layers
    model.add(Dense(19456, activation='elu'))
    model.add(Dense(500, activation='elu'))
    model.add(Dense(2))  # output layer: steering angle and throttle
    
    return model

# Use GPU
from keras import backend as K
K.tensorflow_backend._get_available_gpus()


model = build_model()

# compile and train the model
model.compile(loss='mean_squared_error', optimizer=Adam(lr=1.0e-4))
checkpoint = ModelCheckpoint('model-{epoch:03d}.h5', monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
model.fit_generator(batch_generator(train_samples), 
                    steps_per_epoch=100, 
                    epochs=2, 
                    validation_data=batch_generator(validation_samples, is_training=False),
                    validation_steps=len(validation_samples),
                    callbacks=[checkpoint],
                    verbose=1)
