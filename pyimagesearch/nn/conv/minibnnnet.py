# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from larq.layers import QuantConv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from larq.layers import QuantDense
from tensorflow.keras import backend as K


class MiniBNNNet:
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
        kwargs = dict(input_quantizer="ste_sign",
                      kernel_quantizer="ste_sign",
                      kernel_constraint="weight_clip")

        # first quantCONV => ste_sign,weight_clip => POOL layer set => BN
        model.add(QuantConv2D(32, (3, 3),
                              kernel_quantizer="ste_sign",
                              kernel_constraint="weight_clip",
                              use_bias=False,
                              padding="same",
                              input_shape=inputShape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization(axis=chanDim))

        # second quantCONV => ste_sign,weight_clip => POOL layer set => BN
        model.add(QuantConv2D(64, (3, 3),
                              use_bias=False,
                              padding="same",
                              **kwargs))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization(axis=chanDim))

        # third quantCONV => ste_sign,weight_clip => BN => Flattened
        model.add(QuantConv2D(64, (3, 3),
                              use_bias=False,
                              padding="same",
                              **kwargs))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Flatten())

        # fourth quantFC => BN => quantFC => BN = SoftMax
        model.add(QuantDense(64, use_bias=False, **kwargs))
        model.add(BatchNormalization(axis=chanDim))
        model.add(QuantDense(10, use_bias=False, **kwargs))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model
