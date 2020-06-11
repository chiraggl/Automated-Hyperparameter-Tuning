from keras.layers import Convolution2D, Activation, MaxPooling2D, AveragePooling2D, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop, SGD, Nadam, Adamax, Adadelta
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist
import os
import math

def model_train(epoch, nCRP, nH, ksize, nfilter, neurons, optimizers, pooling):
	# Load the MNIST dataset
	dataset = mnist.load_data()

	# Store the training and test values
	(X_train, y_train), (X_test, y_test)  = dataset

	# Lets store the number of rows and columns
	img_rows = X_train[0].shape[0]
	img_cols = X_train[1].shape[0]

	# Converting data to 4D as required by Keras. Thus, our original image shape of (60000,28,28) changed to (60000,28,28,1)
	X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
	X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

	# Store the shape of a single image 
	input_shape = (img_rows, img_cols, 1)

	# Change our image type to float32 data type
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')

	# Normalize our data by changing the range from (0 to 255) to (0 to 1)
	X_train = X_train / 255
	X_test = X_test / 255

	# Now we one hot encode outputs
	y_train = to_categorical(y_train)
	y_test = to_categorical(y_test)

	# Create model
	model = Sequential()
	
	CRPLayer = 1
	
	while CRPLayer <= nCRP:
		# Set of CRP (Convolution, RELU, Pooling)
		model.add(Convolution2D(filters=nfilter, kernel_size=ksize, input_shape=input_shape))
		model.add(Activation('relu'))
		model.add(pooling)
		CRPLayer = CRPLayer+1

	model.add(Flatten())
	
	layer = 1
	
	while layer <= nH:
		# Fully connected layers (w/ RELU)
		model.add(Dense(units=neurons))
		model.add(Activation('relu'))
		
		layer = layer+1
	
	
	model.add(Dense(units=10))
	
	# Softmax (for Multi Classification)
	model.add(Activation('softmax'))

	model.compile(optimizer=optimizers, loss='categorical_crossentropy', metrics=['accuracy'])

	# Training the Model
	model.fit(X_train, y_train, epochs=epoch)

	# Evaluate the performance of our trained model
	scores = model.evaluate(X_test, y_test)
	loss = scores[0]
	accuracy = scores[1]
	
	# Get the percentage accuracy
	accuracy = accuracy * 100
	
	# Rounding to nearest whole number
	accuracy = math.floor(accuracy)
	
	model.save('mnist_model.h5')
	
	os.system("mv mnist_model.h5 /ws/")
	
	return accuracy

# Set the value of Hyperparameters
epoch=1
nCRP=1
nH=1
ksize=(3,3)
nfilter=32
neurons=128
optimizers=Adam()
pooling=MaxPooling2D()


model_accuracy = model_train(epoch, nCRP, nH, ksize, nfilter, neurons, optimizers, pooling)


# Print and save the accuracy in a file.
with open("accuracy.txt", "w") as f:
    print(model_accuracy, file=f)
    
os.system("mv accuracy.txt /ws/")
    





