import tensorflow.keras.backend as K
from tensorflow.keras import initializers
from tensorflow.keras.layers import Layer


class Attention(Layer):
    """
    Keras Layer that implements an Attention mechanism for temporal data.
    Supports Masking.
    Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    :param kwargs:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(Attention())
    """
    def __init__(self,
                 step_dim=30,
                 features_dim=0,
                 factor=1,
                 **kwargs):
        self.init = initializers.get('glorot_uniform')

        self.step_dim = step_dim
        self.features_dim = features_dim
        self.factor = factor
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.features_dim = input_shape[-1]

        self.W_omega = self.add_weight(shape=(self.features_dim, self.features_dim * self.factor),
                                       initializer=self.init,
                                       name='{}_W_omega'.format(self.name),
                                       trainable=True)

        self.U_omega = self.add_weight(shape=(self.features_dim * self.factor,),
                                       initializer=self.init,
                                       name='{}_U_omega'.format(self.name),
                                       trainable=True)
        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        reshape = K.reshape(x, (-1, features_dim))
        attn_tanh = K.tanh(K.dot(reshape, self.W_omega))
        attn_hidden = K.dot(attn_tanh, K.reshape(self.U_omega, (-1, 1)))
        exps = K.exp(attn_hidden)

        exps = K.reshape(exps, (-1, step_dim))

        if mask is not None:
            exps *= K.cast(mask, K.floatx())

        attn = exps / K.cast(K.sum(exps, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        attn = K.expand_dims(attn)
        weighted_input = x * attn

        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'step_dim': self.step_dim,
            'features_dim': self.features_dim,
            'factor': self.factor
            }
        )

        return config