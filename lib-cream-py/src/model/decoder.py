import tensorflow as tf

from keras import Layer
from keras.src.layers import Conv2D
from keras.saving import register_keras_serializable


@register_keras_serializable(package="DeepCreamPy")
class ConvNN(Layer):
    def __init__(self, dims1, dims2, size1, size2, k_size=3, **kwargs):
        super(ConvNN, self).__init__(**kwargs)
        self.dims1 = dims1
        self.dims2 = dims2
        self.size1 = size1
        self.size2 = size2
        self.k_size = k_size

        # Define layers
        self.conv1 = Conv2D(dims1, (k_size, k_size), strides=(1, 1), padding='valid', activation="elu")
        self.conv2 = Conv2D(dims2, (k_size, k_size), strides=(1, 1), padding='valid', activation="elu")

    def get_config(self):
        base_config = super().get_config()
        config = {
            "dims1": self.dims1,
            "dims2": self.dims2,
            "size1": self.size1,
            "size2": self.size2,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        size1 = config.pop("size1")
        size2 = config.pop("size2")
        dims1 = config.pop("dims1")
        dims2 = config.pop("dims2")
        return cls(dims1, dims2, size1, size2, **config)

    def call(self, inputs, training=False):
        x = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        x = self.conv1(x)

        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        x = self.conv2(x)

        # Resize output
        x = tf.image.resize(x, (self.size1, self.size2), method='nearest')
        return x


@register_keras_serializable(package="DeepCreamPy")
class Decoder(Layer):
    def __init__(self, size1, size2, name: str, **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.size1 = size1
        self.size2 = size2
        self.name = name
        self.dl1 = ConvNN(128, 128, int(size1 // 4), int(size2 // 4))
        self.dl2 = ConvNN(64, 64, int(size1 // 2), int(size2 // 2))
        self.dl3 = ConvNN(32, 32, int(size1), int(size2))
        self.dl4 = ConvNN(16, 16, int(size1), int(size2))
        self.final_conv = Conv2D(3, (3, 3), strides=(1, 1), padding='same', activation=None)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "size1": self.size1,
            "size2": self.size2,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        size1 = config.pop("size1")
        size2 = config.pop("size2")
        return cls(size1, size2, **config)

    def call(self, inputs, training=False):
        x = self.dl1(inputs)
        x = self.dl2(x)
        x = self.dl3(x)
        x = self.dl4(x)
        x = self.final_conv(x)
        x = tf.clip_by_value(x, -1.0, 1.0)
        return x
