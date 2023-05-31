from tensorflow.keras.layers import Embedding, SpatialDropout1D, Dense, GRU
from tensorflow.keras.models import Sequential

from basemodel import BaseModel

class PyLogSentiment(BaseModel):
    def __init__(self, embedding_data_dct, cfg):
        super(PyLogSentiment, self).__init__(embedding_data_dct,
                                             cfg)
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
                                    trainable=False)
        model = Sequential()
        model.add(embedding_layer)
        model.add(SpatialDropout1D(self.dropout))
        model.add(GRU(self.units, dropout=self.dropout, recurrent_dropout=self.dropout, activation=self.activation))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

        return model


