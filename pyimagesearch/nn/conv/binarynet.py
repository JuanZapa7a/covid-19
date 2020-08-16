# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from larq.layers import QuantConv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import custom_object_scope
from larq.layers import QuantDense
from larq.quantizers import DoReFaQuantizer, SteTern, MagnitudeAwareSign, \
    _clipped_gradient
from tensorflow.keras import backend as K
import tensorflow as tf


@tf.custom_gradient
def sign(x):
    def grad(dy):
        return dy

    return tf.sign(x), grad


@custom_object_scope
def binarize(x):
    return sign(tf.clip_by_value(x, -1, 1))


custom_objects = {"binarize": binarize}


# @tf.custom_gradient
# def identity_sign(x):
#     def grad(dy):
#         return dy
#
#     return tf.sign(tf.clip_by_value(x, -1, 1)), grad
#
#
@custom_object_scope
@tf.custom_gradient
def ste_quad(x, threshold_value: float = 0.0125, clip_value: float = 1.0):
    def grad(dy):
        return _clipped_gradient(x, dy, clip_value)

    threshold = threshold_value
    return (tf.sign(x + threshold) + tf.sign(x - threshold) +
            tf.sign(x +
                    3 * threshold) + tf.sign(x - 3 * threshold)) / 4, grad


@custom_object_scope
def quantize(x):
    return tf.quantization.fake_quant_with_min_max_args(
        inputs = x, min = -3, max = 3, num_bits = 8, narrow_range = True
    )


class BinaryNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # All quantized layers except the first will use the same options
        kwargs = dict(input_quantizer = "ste_sign",
                      kernel_quantizer = "ste_sign",
                      kernel_constraint = "weight_clip",
                      use_bias = False)

        # first layer
        # In the first layer we only quantize the weights and not the input
        model.add(QuantConv2D(128, (3, 3),
                              kernel_quantizer = "ste_sign",
                              kernel_constraint = "weight_clip",
                              use_bias = False,
                              padding = "valid",
                              input_shape = inputShape))
        model.add(BatchNormalization(axis = chanDim))

        # second layer
        model.add(QuantConv2D(128, (3, 3), padding = "valid", **kwargs))
        model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
        model.add(BatchNormalization(axis = chanDim))

        # third layer
        model.add(QuantConv2D(256, (3, 3), padding = "valid", **kwargs))
        model.add(BatchNormalization(axis = chanDim))

        # fourth layer
        model.add(QuantConv2D(256, (3, 3), padding = "valid", **kwargs))
        model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
        model.add(BatchNormalization(axis = chanDim))

        # fifth layer
        model.add(QuantConv2D(512, (3, 3), padding = "valid", **kwargs))
        model.add(BatchNormalization(axis = chanDim))

        # sixth layer
        model.add(QuantConv2D(512, (3, 3), padding = "valid", **kwargs))
        # model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
        model.add(BatchNormalization(axis = chanDim))
        model.add(Flatten())

        # seventh layer
        model.add(QuantDense(1024, **kwargs))
        model.add(BatchNormalization(axis = chanDim))

        # eigth layer
        model.add(QuantDense(1024, **kwargs))
        model.add(BatchNormalization(axis = chanDim))

        # ninth layer
        model.add(QuantDense(10, **kwargs))
        model.add(BatchNormalization(axis = chanDim))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model


class BinaryNet8:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # All quantized layers except the first will use the same options
        kwargs = dict(input_quantizer = "ste_sign",
                      kernel_quantizer = "ste_sign",
                      kernel_constraint = "weight_clip",
                      use_bias = False)

        # first layer
        # In the first layer we only quantize the weights and not the input
        model.add(QuantConv2D(64, (2, 2),
                              kernel_quantizer = "ste_sign",
                              kernel_constraint = "weight_clip",
                              use_bias = False,
                              padding = "valid",
                              input_shape = inputShape))
        model.add(BatchNormalization(axis = chanDim))

        # second layer
        model.add(QuantConv2D(64, (2, 2), padding = "valid", **kwargs))
        model.add(BatchNormalization(axis = chanDim))

        # third layer
        model.add(QuantConv2D(64, (2, 2), padding = "valid", **kwargs))
        model.add(BatchNormalization(axis = chanDim))

        # fourth layer
        model.add(QuantConv2D(64, (2, 2), padding = "valid", **kwargs))
        model.add(BatchNormalization(axis = chanDim))
        model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

        # fifth layer
        model.add(QuantConv2D(128, (2, 2), padding = "valid", **kwargs))
        model.add(BatchNormalization(axis = chanDim))

        # sixth layer
        model.add(QuantConv2D(128, (2, 2), padding = "valid", **kwargs))
        model.add(BatchNormalization(axis = chanDim))
        model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

        # seventh layer
        model.add(QuantConv2D(256, (2, 2), padding = "valid", **kwargs))
        model.add(BatchNormalization(axis = chanDim))

        # eight layer
        model.add(QuantConv2D(256, (2, 2), padding = "valid", **kwargs))
        model.add(BatchNormalization(axis = chanDim))
        model.add(Flatten())

        # ninth layer
        model.add(QuantDense(10, **kwargs))
        model.add(BatchNormalization(axis = chanDim))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model

class TernaryNet8:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # All quantized layers except the first will use the same options
        kwargs = dict(input_quantizer = "ste_tern",
                          kernel_quantizer = "ste_tern",
                          kernel_constraint = "weight_clip",
                          use_bias = False)

        # Zero layer
        # In the first layer we only quantize the weights and not the input
        # k=64 size (2,2) stride = 1

        model.add(QuantConv2D(64, (2, 2),
                              kernel_quantizer = "ste_tern",
                              kernel_constraint = "weight_clip",
                              use_bias = False,
                              padding = "valid",
                              input_shape = inputShape))
        model.add(BatchNormalization(axis = chanDim, momentum = 0.9))

        # first layer
        # k=64 size (2,2) stride = 1
        model.add(QuantConv2D(64, (2, 2), padding = "valid", **kwargs))
        model.add(BatchNormalization(axis = chanDim, momentum = 0.9))

        # second layer
        # k=64 size (2,2) stride = 1
        model.add(QuantConv2D(64, (2, 2), padding = "valid", **kwargs))
        model.add(BatchNormalization(axis = chanDim, momentum = 0.9))

        # third layer
        # k=64 size (2,2) stride = 1
        model.add(QuantConv2D(64, (2, 2), padding = "valid", **kwargs))
        model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
        model.add(BatchNormalization(axis = chanDim, momentum = 0.9))

        # fourth layer
        # k=128 size (2,2) stride = 1
        model.add(QuantConv2D(128, (2, 2), padding = "valid", **kwargs))
        model.add(BatchNormalization(axis = chanDim, momentum = 0.9))

        # fifth layer
        # k=128 size (2,2) stride = 1
        model.add(QuantConv2D(128, (2, 2), padding = "valid", **kwargs))
        model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
        model.add(BatchNormalization(axis = chanDim, momentum = 0.9))

        # # sixth layer
        # # k=256 size (2,2) stride = 1
        model.add(QuantConv2D(256, (2, 2), padding = "valid", **kwargs))
        model.add(BatchNormalization(axis = chanDim, momentum = 0.9))

        # seventh layer
        # k=256 size (2,2) stride = 1
        model.add(QuantConv2D(256, (2, 2), padding = "valid", **kwargs))
        model.add(BatchNormalization(axis = chanDim, momentum = 0.9))
        model.add(Flatten())

        # eighth layer
        # model.add(QuantDense(1024, **kwargs))
        # model.add(BatchNormalization(axis = chanDim, momentum = 0.9))
        # model.add(QuantDense(256, **kwargs))
        # model.add(BatchNormalization(axis = chanDim, momentum = 0.9))
        model.add(QuantDense(10, **kwargs))
        model.add(BatchNormalization(axis = chanDim, momentum = 0.9))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model

class DorefaNet8:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # All quantized layers except the first will use the same options
        kwargs = dict(input_quantizer =  "dorefa_quantizer",
                          kernel_quantizer = "dorefa_quantizer",
                          kernel_constraint = "weight_clip",
                          use_bias = False)

        # Zero layer
        # In the first layer we only quantize the weights and not the input
        # k=64 size (2,2) stride = 1

        model.add(QuantConv2D(128, (2, 2),
                              kernel_quantizer = "dorefa_quantizer",
                              kernel_constraint = "weight_clip",
                              use_bias = False,
                              padding = "valid",
                              input_shape = inputShape))
        model.add(BatchNormalization(axis = chanDim, momentum = 0.9))

        # first layer
        # k=64 size (2,2) stride = 1
        model.add(QuantConv2D(128, (2, 2), padding = "valid", **kwargs))
        model.add(BatchNormalization(axis = chanDim, momentum = 0.9))

        # second layer
        # k=64 size (2,2) stride = 1
        model.add(QuantConv2D(128, (2, 2), padding = "valid", **kwargs))
        model.add(BatchNormalization(axis = chanDim, momentum = 0.9))

        # third layer
        # k=64 size (2,2) stride = 1
        model.add(QuantConv2D(128, (2, 2), padding = "valid", **kwargs))
        model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
        model.add(BatchNormalization(axis = chanDim, momentum = 0.9))

        # fourth layer
        # k=128 size (2,2) stride = 1
        model.add(QuantConv2D(256, (2, 2), padding = "valid", **kwargs))
        model.add(BatchNormalization(axis = chanDim, momentum = 0.9))

        # fifth layer
        # k=128 size (2,2) stride = 1
        model.add(QuantConv2D(256, (2, 2), padding = "valid", **kwargs))
        model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
        model.add(BatchNormalization(axis = chanDim, momentum = 0.9))

        # # sixth layer
        # # k=256 size (2,2) stride = 1
        model.add(QuantConv2D(512, (2, 2), padding = "valid", **kwargs))
        model.add(BatchNormalization(axis = chanDim, momentum = 0.9))

        # seventh layer
        # k=256 size (2,2) stride = 1
        model.add(QuantConv2D(512, (2, 2), padding = "valid", **kwargs))
        model.add(BatchNormalization(axis = chanDim, momentum = 0.9))
        model.add(Flatten())

        # eighth layer
        # model.add(QuantDense(1024, **kwargs))
        # model.add(BatchNormalization(axis = chanDim, momentum = 0.9))
        model.add(QuantDense(256, **kwargs))
        model.add(BatchNormalization(axis = chanDim, momentum = 0.9))
        model.add(QuantDense(10, **kwargs))
        model.add(BatchNormalization(axis = chanDim, momentum = 0.9))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model
