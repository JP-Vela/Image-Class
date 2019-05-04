from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Dense, Activation, Dropout, Flatten
from keras import optimizers
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

# step 1: load data

img_width = 150
img_height = 150
train_data_dir = 'data/train'
valid_data_dir = 'data/validation'
train_batch_size = 5

datagen = ImageDataGenerator(rescale = 1./255)

train_generator = datagen.flow_from_directory(directory=train_data_dir,
											   target_size=(img_width,img_height),
											   classes=['dogs','cats'],
											   class_mode='binary',
											   batch_size=train_batch_size)

validation_generator = datagen.flow_from_directory(directory=valid_data_dir,
											   target_size=(img_width,img_height),
											   classes=['dogs','cats'],
											   class_mode='binary',
											   batch_size=32)


# step-2 : build model

model =Sequential([
	Conv2D(32,(3,3), input_shape=(img_width, img_height, 3)),
	Conv2D(32,(3,3), input_shape=(img_width, img_height, 3)),
	Conv2D(32,(3,3), input_shape=(img_width, img_height, 3)),
	MaxPooling2D(pool_size=(2,2)),
	Conv2D(42,(3,3), input_shape=(img_width, img_height, 3)),
	Conv2D(42,(3,3), input_shape=(img_width, img_height, 3)),
	Conv2D(42,(3,3), input_shape=(img_width, img_height, 3)),
	MaxPooling2D(pool_size=(2,2)),
	Conv2D(42,(3,3), input_shape=(img_width, img_height, 3)),
	Conv2D(52,(3,3), input_shape=(img_width, img_height, 3)),
	Conv2D(52,(3,3), input_shape=(img_width, img_height, 3)),
	Activation('relu'),
	MaxPooling2D(pool_size=(2,2)),
	Conv2D(54,(3,3), input_shape=(img_width, img_height, 3)),
	MaxPooling2D(pool_size=(2,2)),
	Flatten(),
	Dense(64, activation='relu'),
	Dense(64, activation='relu'),
	Dense(1, activation='softmax')

])

model.compile(optimizers.Adam(lr=0.01),loss='binary_crossentropy',metrics=['accuracy'])

print('model complied!!')

print('starting training....')
training = model.fit_generator(generator=train_generator, steps_per_epoch=2048 // train_batch_size,epochs=40,validation_data=validation_generator,validation_steps=832//32)

print('training finished!!')

print('saving weights to simple_CNN.h5')

model.save_weights('models/simple_CNN.h5')

print('all weights saved successfully !!')
#models.load_weights('models/simple_CNN.h5')
