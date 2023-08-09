# Final Project

# Author: Sujai Rajan
# CS-6140 Machine Learning

# Description: This file is used to train the model using the dataset generated from the simulator.


## Importing the functions from functions.py
from functions import *
print('\nInitializing Code...\n')

# Step 1 - Importing the dataset
path = 'Sim_Data/'
data = import_dataset_info(path)
print('Dataset Imported\n')


## Step 2 - Visualizing the dataset 
balance_dataset(data, display=False)
print('Dataset Balanced\n')


## Step 3 - Preprocessing the dataset
image_path, steering_angle = load_dataset(path, data)
print('Dataset Loaded\n')


## Step 4 - Splitting the dataset into training and validation set
test_size_given = 0.2
random_state_given = 7  # User defined parameters
image_path_train, image_path_val, steering_angle_train, steering_angle_val = training_validation_split(image_path, steering_angle, test_size_given, random_state_given)
print('Dataset Splitted\n')


## Step 5 - Augmenting the dataset

# The function is made to augmnet 6 functions in random order of 50% probability
# Function is applied to the training set only by the batch generator function


## Step 6 - Preprocessing the dataset

# The function is made to crop the image, change the color space to YUV, apply Gaussian Blur, resize and normalize the image
# Function is applied to both training and validation set by the batch generator function


## Step 7 - Creating batches of the dataset
batch_size_given = 100
batch_generator_train = batch_generator(image_path_train, steering_angle_train, batch_size_given, True)      # Training set batch generator 
batch_generator_val = batch_generator(image_path_val, steering_angle_val, batch_size_given, False)           # Validation set batch generator with augmentation set to False



## Step 8 - Creating the model
model = create_model()
model.summary()    # Summary of the model
print('Model Created\n')


## Step 9 - Training the model
epochs_given = 10
steps_per_epoch_given = 300
validation_steps_per_epoch_given = 200
history = model.fit_generator(batch_generator_train, steps_per_epoch=steps_per_epoch_given, epochs=epochs_given, validation_data=batch_generator_val, validation_steps=validation_steps_per_epoch_given, verbose=True, shuffle=True)
print('Model Trained\n')


# Step 10 - Saving the model
model.save('autonomous_model.h5')
print('Model Saved\n')


# Step 11 - Plotting the training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()
print('Training and Validation Loss Plotted\n')


# # Step 12 - Plotting the training and validation accuracy
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.legend(['Training', 'Validation'])
# plt.title('Accuracy')
# plt.xlabel('Epoch')
# plt.show()
# print('Training and Validation Accuracy Plotted\n')


# # Step 13 - Plotting the training and validation mean squared error
# plt.plot(history.history['mean_squared_error'])
# plt.plot(history.history['val_mean_squared_error'])
# plt.legend(['Training', 'Validation'])
# plt.title('Mean Squared Error')
# plt.xlabel('Epoch')
# plt.show()
# print('Training and Validation Mean Squared Error Plotted\n')





