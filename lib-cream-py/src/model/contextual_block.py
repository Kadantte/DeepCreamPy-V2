import tensorflow as tf
from keras import Layer
from keras.src.layers import Conv2D
from keras.saving import register_keras_serializable


def ceil(a, b):
    return -(-a // b)


def calc_c(h, kn, w, stride):
    # c = 0
    # for _ in range(kn, h - kn, stride):
    #    for q in range(kn, w - kn, stride):
    #        c += 1
    return ceil((h - 2 * kn), stride) * ceil((w - 2 * kn), stride)


def softmax(x):
    # todo: why - 3
    exp_x = tf.exp(x - 3)
    return exp_x / tf.reduce_sum(exp_x, axis=-1, keepdims=True)


@register_keras_serializable(package="DeepCreamPy")
class ContextualBlock(Layer):
    def __init__(self, k_size, lamda, stride=1, name=str, **kwargs):
        super(ContextualBlock, self).__init__(name=name, **kwargs)
        self.conv_layer = None
        self.k_size = k_size
        self.lamda = lamda
        self.stride = stride

    def get_config(self):
        base_config = super().get_config()
        config = {
            "k_size": self.k_size,
            "lamda": self.lamda,
            "stride": self.stride,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        k_size = config.pop("k_size")
        lamda = config.pop("lamda")
        return cls(k_size, lamda, **config)

    def build(self, input_shape):
        bg_in, fg_in, mask = input_shape

        b, h, w, dims = bg_in
        self.conv_layer = Conv2D(dims, (1, 1), padding="valid", name="ML", activation="elu")

    def call(self, inputs):
        bg_in, fg_in, mask = inputs
        b, h, w, dims = bg_in.shape

        # Resize mask
        temp = tf.image.resize(mask, (h, w), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        temp = tf.expand_dims(temp[:, :, :, 0], axis=-1)
        mask_r = tf.tile(temp, [1, 1, 1, dims])

        bg = bg_in * mask_r

        kn = (self.k_size - 1) // 2
        c = calc_c(h, kn, w, self.stride)
        # Extract patches
        patch1 = tf.image.extract_patches(bg,
                                          sizes=[1, self.k_size, self.k_size, 1],
                                          strides=[1, self.stride, self.stride, 1],
                                          rates=[1, 1, 1, 1],
                                          padding='VALID')

        patch1 = tf.reshape(patch1, (b, 1, 1, c, self.k_size * self.k_size * dims))
        patch1 = tf.transpose(patch1, [0, 1, 2, 4, 3])

        patch2 = tf.image.extract_patches(fg_in,
                                          sizes=[1, self.k_size, self.k_size, 1],
                                          strides=[1, 1, 1, 1],
                                          rates=[1, 1, 1, 1],
                                          padding='SAME')

        ACL = []
        for ib in range(b):
            k1 = patch1[ib]
            k1d = tf.reduce_sum(tf.square(k1), axis=2)
            k2 = tf.reshape(k1, (self.k_size, self.k_size, dims, c))

            ww = patch2[ib]
            wwd = tf.reduce_sum(tf.square(ww), axis=2, keepdims=True)
            ft = tf.expand_dims(ww, 0)

            CS = tf.nn.conv2d(ft, k1, strides=[1, 1, 1, 1], padding='SAME')

            tt = k1d + wwd
            DS1 = tf.expand_dims(tt, 0) - 2 * CS
            DS2 = (DS1 - tf.reduce_mean(DS1, axis=-1, keepdims=True)) / tf.math.reduce_std(DS1, axis=-1, keepdims=True)
            DS2 = -1 * tf.nn.tanh(DS2)

            CA = softmax(self.lamda * DS2)

            ACLt = tf.nn.conv2d_transpose(CA, k2, output_shape=[1, h, w, dims], strides=[1, 1, 1, 1], padding='SAME')
            ACLt = ACLt / (self.k_size ** 2)

            if ib == 0:
                ACL = ACLt
            else:
                ACL = tf.concat([ACL, ACLt], axis=0)

        ACL = bg + ACL * (1.0 - mask_r)

        con1 = tf.concat([bg_in, ACL], axis=-1)
        ACL2 = self.conv_layer(con1)
        return ACL2


if __name__ == '__main__':
    import math


    # The original function with the loop
    def calc_c(h, kn, w, stride):
        c = 0
        for _ in range(kn, h - kn, stride):
            for q in range(kn, w - kn, stride):
                c += 1
        return c


    # The formula function
    def calc_c_formula(h, kn, w, stride):
        return math.ceil((h - 2 * kn) / stride) * math.ceil((w - 2 * kn) / stride)


    # Test to compare both functions
    def test_calc_c_equal():
        test_cases = [
            (5, 1, 5, 1),  # Expected output: 9
            (6, 1, 6, 1),  # Expected output: 16

            # Test case with larger stride
            (10, 1, 10, 2),  # Expected output: 9
            (10, 1, 10, 3),  # Expected output: 4

            # Test case with larger kernel size
            (10, 3, 10, 1),  # Expected output: 4
            (10, 3, 10, 2),  # Expected output: 4

            # Test case with large height and width
            (100, 5, 100, 10),  # Expected output: 90
            (100, 5, 100, 20),  # Expected output: 45

            # Test case with small stride and kernel
            (15, 1, 15, 1),  # Expected output: 169
            (15, 1, 15, 3),  # Expected output: 36

            # Test case with kernel and stride being equal
            (20, 2, 20, 2),  # Expected output: 36
            (20, 4, 20, 4),  # Expected output: 16

            # Test case with larger height and width with minimal stride
            (50, 3, 50, 1),  # Expected output: 44 * 44 = 1936
            (50, 4, 50, 1),  # Expected output: 43 * 43 = 1849

            # Test case with minimal kernel and larger stride
            (30, 1, 30, 5),  # Expected output: 6 * 6 = 36
            (30, 2, 30, 6),  # Expected output: 5 * 5 = 25
        ]

        for h, kn, w, stride in test_cases:
            loop_result = calc_c(h, kn, w, stride)
            formula_result = calc_c_formula(h, kn, w, stride)

            assert loop_result == formula_result, f"Failed for {h=}, {kn=}, {w=}, {stride=}: loop result ({loop_result}) != formula result ({formula_result})"


    # Run the test
    test_calc_c_equal()
