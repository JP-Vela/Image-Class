from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Dense, Activation, Dropout, Flatten
from keras import optimizers
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import categorical_crossentropy
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
											   classes=['dogs','cat'],
											   class_mode='binary',
											   batch_size=32)


# step-2 : build model

model =Sequential([
	Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(img_width, img_height, 3)),
	Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(img_width, img_height, 3)),
	Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(img_width, img_height, 3)),
	Conv2D(filters=42, kernel_size=3, activation='relu', input_shape=(img_width, img_height, 3)),
	Conv2D(filters=42, kernel_size=3, activation='relu', input_shape=(img_width, img_height, 3)),
	Conv2D(filters=42, kernel_size=3, activation='relu', input_shape=(img_width, img_height, 3)),
	Conv2D(filters=52, kernel_size=3, activation='relu', input_shape=(img_width, img_height, 3)),

	MaxPooling2D(pool_size=2),
	Conv2D(filters=52, kernel_size=3, activation='relu', input_shape=(img_width, img_height, 3)),
	Conv2D(filters=52, kernel_size=3, activation='relu', input_shape=(img_width, img_height, 3)),
	Conv2D(filters=52, kernel_size=3, activation='relu', input_shape=(img_width, img_height, 3)),
		Conv2D(filters=52, kernel_size=3, activation='relu', input_shape=(img_width, img_height, 3)),
		Conv2D(filters=52, kernel_size=3, activation='relu', input_shape=(img_width, img_height, 3)),
		Conv2D(filters=52, kernel_size=3, activation='relu', input_shape=(img_width, img_height, 3)),

	MaxPooling2D(pool_size=2),

		Conv2D(filters=52, kernel_size=3, activation='relu', input_shape=(img_width, img_height, 3)),
		Conv2D(filters=52, kernel_size=3, activation='relu', input_shape=(img_width, img_height, 3)),
		Conv2D(filters=52, kernel_size=3, activation='relu', input_shape=(img_width, img_height, 3)),
		Conv2D(filters=52, kernel_size=3, activation='relu', input_shape=(img_width, img_height, 3)),
		Conv2D(filters=52, kernel_size=3, activation='relu', input_shape=(img_width, img_height, 3)),
		Conv2D(filters=52, kernel_size=3, activation='relu', input_shape=(img_width, img_height, 3)),
		Conv2D(filters=52, kernel_size=3, activation='relu', input_shape=(img_width, img_height, 3)),

		MaxPooling2D(pool_size=2),



	Conv2D(filters=64, kernel_size=3, activation='relu', input_shape=(img_width, img_height, 3)),
	Conv2D(filters=64, kernel_size=3, activation='relu', input_shape=(img_width, img_height, 3)),
	MaxPooling2D(pool_size=2),
	#Dropout(0.2),
	Flatten(),
	Dense(64,activation='relu'),
	Dense(2,activation='softmax')

])

model.compile(optimizers.Adam(lr=0.0001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

print('model complied!!')

#print('starting training....')
training = model.fit_generator(train_generator, steps_per_epoch=11238 // train_batch_size,epochs=40,validation_data=validation_generator,validation_steps=832//32)

print('training finished!!')

print('saving weights to simple_CNN.h5')

model.save_weights('models/simple_CNN.h5')

print('all weights saved successfully !!')
models.load_weights('models/simple_CNN.h5')
