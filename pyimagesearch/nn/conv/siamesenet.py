# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K


class SiameseNet:
    @staticmethod
    def build(width, height, depth, classes=1, reg=0.0002):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = ( height, width, depth)
        chanDim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # Define the tensors for the two input images
        left_input = Input(inputShape)
        right_input = Input(inputShape)


        # Block #1: first CONV => RELU => POOL layer set
        model.add(Conv2D(64, (10, 10),
                         input_shape = inputShape,
                         #kernel_initializer = initialize_weights,
                         kernel_regularizer = l2(reg)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D())

        # Block #2: second CONV => RELU => POOL layer set
        model.add(Conv2D(128, (7, 7),
                         input_shape = inputShape,
                         #kernel_initializer = initialize_weights,
                         #bias_initializer = initializa_bias,
                         kernel_regularizer = l2(reg)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D())

        # Block #3: CONV => RELU => POOL layer set
        model.add(Conv2D(128, (4, 4),
                         input_shape = inputShape,
                         #kernel_initializer = initialize_weights,
                         #bias_initializer = initializa_bias,
                         kernel_regularizer = l2(reg)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D())

        # Block #4: CONV => RELU => FLATTEN => DENSE =< SIGMOID layer set
        model.add(Conv2D(256, (4, 4),
                         input_shape = inputShape,
                         #kernel_initializer = initialize_weights,
                         #bias_initializer = initializa_bias,
                         kernel_regularizer = l2(reg)))
        model.add(Activation("relu"))
        model.add(Flatten())
        model.add(Dense(4096,
                        #kernel_initializer = initialize_weights,
                        #bias_initializer = initializa_bias,
                        kernel_regularizer = l2(1e-3)))
        model.add(Activation("sigmoid"))

        # Generate the encodings (feature vectors) for the two images
        encoded_l = model(left_input)
        encoded_r = model(right_input)

        # Add a customized layer to compute the absolute difference between the encodings
        L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
        L1_distance = L1_layer([encoded_l, encoded_r])

        # Add a dense layer with a sigmoid unit to generate the similarity score
        prediction = Dense( classes , activation = 'sigmoid'#,
                           #bias_initializer = initialize_bias
                            )(L1_distance)

        # Connect the inputs with the outputs
        model = Model(inputs = [left_input, right_input],
                            outputs = prediction)

        # return the constructed network architecture
        return model
