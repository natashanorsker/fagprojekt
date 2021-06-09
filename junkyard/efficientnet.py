#%% Model
import tensorflow as tf
from keras import applications
from efficientnet.tfkeras import EfficientNetB0
from keras import callbacks
from keras.models import Sequential
import cv2
from tensorflow.keras.applications import EfficientNetB0
model = EfficientNetB0(include_top=False, weights='imagenet')
# IMG_SIZE is determined by EfficientNet model choice (must be this)
# https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
IMG_SIZE = 224

#%%
from keras.layers import Dense
from keras.optimizers import Adam

efficient_net = EfficientNetB0(
    weights='imagenet',
    input_shape=(32,32,3),
    include_top=False,
    pooling='max'
)

model = Sequential()
model.add(efficient_net)
model.add(Dense(units = 120, activation='relu'))
model.add(Dense(units = 120, activation = 'relu'))
model.add(Dense(units = 1, activation='sigmoid'))
model.summary()

model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    epochs = 50,
    steps_per_epoch = 15,
    validation_data = val_generator,
    validation_steps = 7
)