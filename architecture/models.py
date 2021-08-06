from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Dropout


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
        vgg_model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        vgg_model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

        vgg_model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        vgg_model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        vgg_model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

        vgg_model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        vgg_model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        vgg_model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        vgg_model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

        vgg_model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        vgg_model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        vgg_model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        vgg_model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

        vgg_model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        vgg_model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        vgg_model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        vgg_model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

        vgg_model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        vgg_model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        vgg_model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        vgg_model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

        vgg_model.add(Flatten())
        vgg_model.add(Dense(4096, activation='relu'))
        #vgg_model.add(Dropout(0.5))
        vgg_model.add(Dense(4096, activation='relu'))
        #vgg_model.add(Dropout(0.5))
        vgg_model.add(Dense(1, activation='sigmoid'))

        return vgg_model