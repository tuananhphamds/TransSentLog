import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dropout, Dense, Embedding, Layer, \
                                    SpatialDropout1D
from tensorflow.keras.models import Sequential, load_model
from basemodel import BaseModel
from averagepoolmask import AveragePoolMask
from official.nlp import optimization


class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential(
            [
                Dense(ff_dim, activation='relu'),
                Dense(embed_dim)
            ]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, tf.newaxis, :]
            attn_output = self.att(inputs, inputs, attention_mask=mask)
        else:
            attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class PositionEmbedding(Layer):
    def __init__(self, maxlen, embed_dim):
        super(PositionEmbedding, self).__init__()
        position_embedding_matrix = self.get_position_encoding(maxlen, embed_dim)
        self.position_embedding_layer = Embedding(
            input_dim=maxlen, output_dim=embed_dim,
            weights=[position_embedding_matrix],
            trainable=False
        )
        self.max_len = maxlen

    def get_position_encoding(self, seq_len, d, n=10000):
        P = np.zeros((seq_len, d))
        for k in range(seq_len):
            for i in np.arange(int(d/2)):
                denominator = np.power(n, 2 * i / d)
                P[k, 2*i] = np.sin(k/denominator)
                P[k, 2*i+1] = np.cos(k/denominator)
        return P

    def call(self, x):
        position_indices = tf.range(start=0, limit=self.max_len, delta=1)
        return x + self.position_embedding_layer(position_indices)


class Transformer(Layer):
    def __init__(self, num_blocks, embed_dim, num_heads, ff_dim, maxlen, dropout=0.1, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.num_blocks = num_blocks
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.maxlen = maxlen
        self.dropout = dropout
        self.blocks = [TransformerBlock(embed_dim=self.embed_dim,
                                        num_heads=self.num_heads,
                                        ff_dim=self.ff_dim,
                                        rate=self.dropout) for i in range(self.num_blocks)]
        self.pos_embedding = PositionEmbedding(maxlen=self.maxlen, embed_dim=self.embed_dim)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        inputs = self.pos_embedding(inputs)

        for i in range(len(self.blocks)):
            inputs = self.blocks[i](inputs, mask=mask)

        return inputs

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_blocks': self.num_blocks,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'maxlen': self.maxlen,
            'dropout': self.dropout
            }
        )

        return config

class TransSentLog(BaseModel):
    def __init__(self, embedding_data_dct, cfg):
        super(TransSentLog, self).__init__(embedding_data_dct, cfg)
        self.ff_dim = 2048
        self.num_blocks = cfg['num_blocks']
        self.num_heads = cfg['num_heads']
        self.dropout = cfg['dropout']
        self.batch_size = cfg['batch_size']
        self.epochs = cfg['epochs']
        self.log_length = cfg['log_length']
        self.glove_dim = cfg['dim']
        self.max_num_words = cfg['max_num_words']

    def _create_optimizer(self):
        steps_per_epoch = len(self.x_train) // self.batch_size
        num_train_steps = steps_per_epoch * self.epochs
        num_warmup_steps = int(0.2 * num_train_steps)
        init_lr = 1e-4
        optimizer = optimization.create_optimizer(init_lr=init_lr,
                                                  num_train_steps=num_train_steps,
                                                  num_warmup_steps=num_warmup_steps,
                                                  optimizer_type='adamw')
        return optimizer

    def _build_model(self):
        # build model and compile
        embedding_layer = Embedding(self.max_num_words + 1,
                                    self.glove_dim,
                                    weights=[self.embedding_matrix],
                                    input_length=self.log_length,
                                    trainable=False,
                                    mask_zero=True)
        model = Sequential()
        model.add(embedding_layer)
        model.add(SpatialDropout1D(0.1))
        model.add(Transformer(num_blocks=self.num_blocks,
                              embed_dim=self.glove_dim,
                              num_heads=self.num_heads,
                              ff_dim=self.ff_dim,
                              maxlen=self.log_length,
                              dropout=self.dropout))
        model.add(AveragePoolMask())
        model.add(Dense(2, activation='softmax'))
        self.optimizer = self._create_optimizer()
        model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['acc'])
        print('\n\n\n----------MODEL SUMMARY----------')
        print(model.summary())
        return model

    def _load_model(self, model_file):
        return load_model(model_file, custom_objects={'Transformer': Transformer,
                                                      'AdamWeightDecay': self.optimizer,
                                                      'AveragePoolMask': AveragePoolMask})