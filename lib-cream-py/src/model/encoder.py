import tensorflow as tf
from keras import Model
from keras.src.layers import Activation, Conv2D
from keras.saving import register_keras_serializable


def reflect_pad(x, pad_size):
    return tf.pad(x, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]], mode="REFLECT")


@register_keras_serializable(package="DeepCreamPy")
class Encoder(Model):
    def __init__(self, name: str, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.name = name

        # Define convolutional layers
        self.conv1 = Conv2D(32, (5, 5), strides=(1, 1), padding='valid')
        self.conv2 = Conv2D(64, (3, 3), strides=(2, 2), padding='valid')
        self.conv3 = Conv2D(64, (3, 3), strides=(1, 1), padding='valid')
        self.conv4 = Conv2D(128, (3, 3), strides=(2, 2), padding='valid')
        self.conv5 = Conv2D(128, (3, 3), strides=(1, 1), padding='valid')
        self.conv6 = Conv2D(256, (3, 3), strides=(2, 2), padding='valid')

        # Dilated Convolutions
        self.dilated1 = Conv2D(256, (3, 3), dilation_rate=2, strides=(1, 1), padding='valid')
        self.dilated2 = Conv2D(256, (3, 3), dilation_rate=4, strides=(1, 1), padding='valid')
        self.dilated3 = Conv2D(256, (3, 3), dilation_rate=8, strides=(1, 1), padding='valid')
        self.dilated4 = Conv2D(256, (3, 3), dilation_rate=16, strides=(1, 1), padding='valid')

    def call(self, inputs, training=False):
        x = reflect_pad(inputs, 2)
        x = self.conv1(x)
        x = Activation('elu')(x)  # 256 256 32

        x = reflect_pad(x, 1)
        x = self.conv2(x)
        x = Activation('elu')(x)  # 128 128 64

        x = reflect_pad(x, 1)
        x = self.conv3(x)
        x = Activation('elu')(x)  # 128 128 64

        x = reflect_pad(x, 1)
        x = self.conv4(x)
        x = Activation('elu')(x)  # 64 64 128

        x = reflect_pad(x, 1)
        x = self.conv5(x)
        x = Activation('elu')(x)  # 64 64 128

        x = reflect_pad(x, 1)
        x = self.conv6(x)
        x = Activation('elu')(x)  # 32 32 128

        # Dilated Convolutions
        x = reflect_pad(x, 2)
        x = self.dilated1(x)
        x = Activation('elu')(x)

        x = reflect_pad(x, 4)
        x = self.dilated2(x)
        x = Activation('elu')(x)

        x = reflect_pad(x, 8)
        x = self.dilated3(x)
        x = Activation('elu')(x)

        x = reflect_pad(x, 16)
        x = self.dilated4(x)
        x = Activation('elu')(x)  # 32 32 128

        return x
