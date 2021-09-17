from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization


class LeNet5:

    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self):
        lenet_model = Sequential()
        lenet_model.add(Conv2D(filters=6, kernel_size=(3, 3), input_shape=self.input_shape, activation='relu'))
        lenet_model.add(MaxPooling2D())

        lenet_model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
        lenet_model.add(MaxPooling2D())

        lenet_model.add(Flatten())
        lenet_model.add(Dense(units=120, activation='relu'))
        lenet_model.add(Dense(units=84, activation='relu'))
        lenet_model.add(Dense(units=1, activation='sigmoid'))

        return lenet_model

class VGG16:

    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self):
        vgg_model = Sequential()
        vgg_model.add(Conv2D(input_shape=self.input_shape, filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
        vgg_model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
        vgg_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        vgg_model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        vgg_model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        vgg_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

        vgg_model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        vgg_model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        vgg_model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        vgg_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        vgg_model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        vgg_model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        vgg_model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        vgg_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        vgg_model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        vgg_model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        vgg_model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        vgg_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        vgg_model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        vgg_model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        vgg_model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        vgg_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        vgg_model.add(Flatten())
        vgg_model.add(Dense(4096, activation='relu'))
        #vgg_model.add(Dropout(0.5))
        vgg_model.add(Dense(4096, activation='relu'))
        #vgg_model.add(Dropout(0.5))
        vgg_model.add(Dense(1, activation='sigmoid'))

        return vgg_model

class Convnet:

    def __init__(self, input_shape):
        self.input_shape = input_shape


    def build(self):
        convnet = Sequential()
        convnet.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=self.input_shape, activation='relu'))
        convnet.add(MaxPooling2D((2, 2)))
        convnet.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        convnet.add(MaxPooling2D((2, 2)))
        convnet.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

        convnet.add(Flatten())
        convnet.add(Dense(units=64, activation='relu'))
        convnet.add(Dropout(0.5))
        convnet.add(Dense(units=1, activation='sigmoid'))

        return convnet


class Convnet2:

    def __init__(self, input_shape):
        self.input_shape = input_shape


    def build(self):
        convnet2 = Sequential()
        convnet2.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=self.input_shape, activation='relu'))
        convnet2.add(MaxPooling2D((2, 2)))
        convnet2.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        convnet2.add(MaxPooling2D((2, 2)))
        convnet2.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
        convnet2.add(MaxPooling2D((2, 2)))
        convnet2.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
        convnet2.add(MaxPooling2D((2, 2)))

        convnet2.add(Flatten())
        convnet2.add(Dropout(0.5))
        convnet2.add(Dense(units=512, activation='relu'))
        convnet2.add(Dense(units=1, activation='sigmoid'))

        return convnet2

class Convnet3:

    def __init__(self, input_shape):
        self.input_shape = input_shape


    def build(self):
        convnet3 = Sequential()
        convnet3.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=self.input_shape, activation='elu'))
        convnet3.add(MaxPooling2D((2, 2)))
        convnet3.add(Conv2D(filters=64, kernel_size=(3, 3), activation='elu'))
        convnet3.add(MaxPooling2D((2, 2)))
        convnet3.add(Conv2D(filters=128, kernel_size=(3, 3), activation='elu'))
        convnet3.add(MaxPooling2D((2, 2)))
        convnet3.add(Conv2D(filters=128, kernel_size=(3, 3), activation='elu'))
        convnet3.add(BatchNormalization())

        convnet3.add(Flatten())
        convnet3.add(Dropout(0.5))
        convnet3.add(Dense(units=512, activation='elu'))
        convnet3.add(Dense(units=1, activation='sigmoid'))

        return convnet3

class Convnet4:

    def __init__(self, input_shape):
        self.input_shape = input_shape


    def build(self):
        convnet4 = Sequential()
        convnet4.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=self.input_shape, activation='relu'))
        convnet4.add(MaxPooling2D((3, 3)))
        convnet4.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
        convnet4.add(MaxPooling2D((2, 2)))
        convnet4.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        #convnet4.add(BatchNormalization())

        convnet4.add(Flatten())
        convnet4.add(Dropout(0.2))
        convnet4.add(Dense(units=360, activation='relu'))
        convnet4.add(Dense(units=1, activation='sigmoid'))

        return convnet4