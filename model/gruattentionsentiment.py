from tensorflow.keras.layers import Embedding, SpatialDropout1D, Dense, GRU, Dropout, InputSpec
from tensorflow.keras.models import Sequential, load_model
from basemodel import BaseModel
from attentionmul import Attention
from tensorflow.keras.initializers import glorot_uniform

class GRUAttentionSentiment(BaseModel):
    def __init__(self, embedding_data_dct, cfg):
        super(GRUAttentionSentiment, self).__init__(embedding_data_dct, cfg)
        self.dropout = cfg['dropout']
        self.units = cfg['units']
        self.activation = 'tanh'
        self.batch_size = cfg['batch_size']
        self.epochs = cfg['epochs']
        self.log_length = cfg['log_length']
        self.glove_dim = cfg['dim']
        self.max_num_words = cfg['max_num_words']

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
        model.add(SpatialDropout1D(0.4))
        model.add(GRU(self.units,
                      dropout=self.dropout,
                      recurrent_dropout=self.dropout,
                      activation=self.activation,
                      return_sequences=True))
        model.add(Attention(step_dim=self.log_length))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        print(model.summary())
        return model

    def _load_model(self, model_file):
        return load_model(model_file, custom_objects={'Attention': Attention,
                                                      'GlorotUniform': glorot_uniform})



