import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer


class AveragePoolMask(Layer):
    def __init__(self, **kwargs):
        super(AveragePoolMask, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        if mask is None:
            raise ValueError('Mask cannot be None')
        mask = K.cast(mask, K.floatx())
        mask = K.repeat(mask, x.shape[-1])
        mask = tf.transpose(mask, [0, 2, 1])
        x = x * mask
        epsilon = K.epsilon()
        average = K.sum(x, axis=1) / (K.sum(mask, axis=1) + epsilon)
        return average

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])
