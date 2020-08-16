# There are only two primary differences between our implementation and the
# full GoogLeNet architecture used by Szegedy et al. when training the
# network on the complete ImageNet dataset:

# 1. Instead of using 7×7 filters with a stride of 2×2 in the first CONV
# layer, we use 5×5 filters with a 1 × 1 stride. We use these due to the fact
# that our implementation of GoogLeNet is only able to accept 64 × 64 × 3
# input images while the original implementation was constructed to accept 224
# × 224 × 3 images. If we applied 7 × 7 filters with a 2 × 2 stride, we would
# reduce our input dimensions too quickly.

# 2. Our implementation is slightly shallower with two fewer Inception
# modules – in the original Szegedy et al. paper, two more Inception modules
# were added prior to the average pooling operation. This implementation of
# GoogLeNet will be more than enough for us to perform well on Tiny ImageNet
# and claim a spot on the cs231n Tiny ImageNet leaderboard. For readers who
# are interested in training the full GoogLeNet architecture from scratch on
# the entire ImageNet dataset (thereby replicating the performance of the
# Szegedy et al. experiments), please refer to Chapter 7 in the ImageNet Bundle.


# import the necessary packages
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

class DeeperGoogLeNet:
	@staticmethod

	# The conv_module method accepts a number of parameters, including:
	# • x: The input to the network.
	# • K: The number of filters the convolutional layer will learn.
	# • kX and kY: The filter size for the convolutional layer.
	# • stride: The stride (in pixels) for the convolution. Typically we’ll use a 1 × 1 stride, but we could use a larger stride if we wished to reduce the output volume size.
	# • chanDim: This value controls the dimension (i.e., axis) of the image channel. It is automatically set later in this class based on whether we are using “channels_last” or “channels_first” ordering.
	# • padding: Here we can control the padding of the convolution layer.
	# • reg: The L2 weight decay strength.
	# • name: Since this network is deeper than all others we have worked with in this book, we may wish to name the blocks of layers to help us (1) debug the network and (2) share/explain the network to others.

	def conv_module(x, K, kX, kY, stride, chanDim,
		padding="same", reg=0.0005, name=None):
		# initialize the CONV, BN, and RELU layer names
		(convName, bnName, actName) = (None, None, None)

		# if a layer name was supplied, prepend it
		if name is not None:
			convName = name + "_conv"
			bnName = name + "_bn"
			actName = name + "_act"

		# define a CONV => BN => RELU pattern
		x = Conv2D(K, (kX, kY), strides=stride, padding=padding,
			kernel_regularizer=l2(reg), name=convName)(x)
		x = BatchNormalization(axis=chanDim, name=bnName)(x)
		x = Activation("relu", name=actName)(x)

		# return the block
		return x

	@staticmethod
	def inception_module(x, num1x1, num3x3Reduce, num3x3,
		num5x5Reduce, num5x5, num1x1Proj, chanDim, stage,
		reg=0.0005):
		# define the first branch of the Inception module which
		# consists of 1x1 convolutions: number of filters num1x1
		first = DeeperGoogLeNet.conv_module(x, num1x1, 1, 1,
			(1, 1), chanDim, reg=reg, name=stage + "_first")

		# define the second branch of the Inception module which
		# consists of 1x1 number of filters num3x3Reduce and 3x3 number of
		# filters num3x3 convolutions
		second = DeeperGoogLeNet.conv_module(x, num3x3Reduce, 1, 1,
			(1, 1), chanDim, reg=reg, name=stage + "_second1")
		second = DeeperGoogLeNet.conv_module(second, num3x3, 3, 3,
			(1, 1), chanDim, reg=reg, name=stage + "_second2")

		# define the third branch of the Inception module which
		# are our 1x1 number of filters num5x5Reduce and 5x5 number of
		# filters num5x5 convolutions
		third = DeeperGoogLeNet.conv_module(x, num5x5Reduce, 1, 1,
			(1, 1), chanDim, reg=reg, name=stage + "_third1")
		third = DeeperGoogLeNet.conv_module(third, num5x5, 5, 5,
			(1, 1), chanDim, reg=reg, name=stage + "_third2")

		# define the fourth branch of the Inception module which
		# is the POOL projection
		fourth = MaxPooling2D((3, 3), strides=(1, 1),
			padding="same", name=stage + "_pool")(x)
		fourth = DeeperGoogLeNet.conv_module(fourth, num1x1Proj,
			1, 1, (1, 1), chanDim, reg=reg, name=stage + "_fourth")

		# concatenate across the channel dimension
		x = concatenate([first, second, third, fourth], axis=chanDim,
			name=stage + "_mixed")

		# return the block
		return x

	@staticmethod
	def build(width, height, depth, classes, reg=0.0005):
		# initialize the input shape to be "channels last" and the
		# channels dimension itself
		inputShape = (height, width, depth)
		chanDim = -1

		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		# define the model input, followed by a sequence of CONV =>
		# POOL => (CONV * 2) => POOL layers
		inputs = Input(shape=inputShape)
		x = DeeperGoogLeNet.conv_module(inputs, 64, 5, 5, (1, 1),
			chanDim, reg=reg, name="block1")
		x = MaxPooling2D((3, 3), strides=(2, 2), padding="same",
			name="pool1")(x)
		x = DeeperGoogLeNet.conv_module(x, 64, 1, 1, (1, 1),
			chanDim, reg=reg, name="block2")
		x = DeeperGoogLeNet.conv_module(x, 192, 3, 3, (1, 1),
			chanDim, reg=reg, name="block3")
		x = MaxPooling2D((3, 3), strides=(2, 2), padding="same",
			name="pool2")(x)

		# apply two Inception modules followed by a POOL
		x = DeeperGoogLeNet.inception_module(x, 64, 96, 128, 16,
			32, 32, chanDim, "3a", reg=reg)
		x = DeeperGoogLeNet.inception_module(x, 128, 128, 192, 32,
			96, 64, chanDim, "3b", reg=reg)
		x = MaxPooling2D((3, 3), strides=(2, 2), padding="same",
			name="pool3")(x)

		# apply five Inception modules followed by POOL
		x = DeeperGoogLeNet.inception_module(x, 192, 96, 208, 16,
			48, 64, chanDim, "4a", reg=reg)
		x = DeeperGoogLeNet.inception_module(x, 160, 112, 224, 24,
			64, 64, chanDim, "4b", reg=reg)
		x = DeeperGoogLeNet.inception_module(x, 128, 128, 256, 24,
			64, 64, chanDim, "4c", reg=reg)
		x = DeeperGoogLeNet.inception_module(x, 112, 144, 288, 32,
			64, 64, chanDim, "4d", reg=reg)
		x = DeeperGoogLeNet.inception_module(x, 256, 160, 320, 32,
			128, 128, chanDim, "4e", reg=reg)
		x = MaxPooling2D((3, 3), strides=(2, 2), padding="same",
			name="pool4")(x)

		# apply a POOL layer (average) followed by dropout
		x = AveragePooling2D((4, 4), name="pool5")(x)
		x = Dropout(0.4, name="do")(x)

		# softmax classifier
		x = Flatten(name="flatten")(x)
		x = Dense(classes, kernel_regularizer=l2(reg),
			name="labels")(x)
		x = Activation("softmax", name="softmax")(x)

		# create the model
		model = Model(inputs, x, name="googlenet")

		# return the constructed network architecture
		return model