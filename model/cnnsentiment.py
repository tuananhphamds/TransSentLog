from tensorflow.keras.layers import Embedding, SpatialDropout1D, Dense, GlobalMaxPool1D, Conv1D
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers

from basemodel import BaseModel

class CNNSentiment(BaseModel):
    def __init__(self, embedding_data_dct, cfg):
        super(CNNSentiment, self).__init__(embedding_data_dct, cfg)
        self.activation = 'relu'
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
        model.add(Conv1D(filters=self.glove_dim,
                         kernel_size=5,
                         padding='same',
                         kernel_regularizer=regularizers.L2(l2=1e-4),
                         activation=self.activation))
        model.add(GlobalMaxPool1D())
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        print(model)
        return model

