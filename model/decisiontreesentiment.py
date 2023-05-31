import os
import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from basemodel import BaseModel

class DecisionTreeSentiment(BaseModel):
    def __init__(self, embedding_data_dct, cfg):
        super(DecisionTreeSentiment, self).__init__(embedding_data_dct, cfg)

    def _build_model(self):
        # build model
        model = DecisionTreeClassifier()
        return model

    def _get_embedded_data(self, data):
        # Transform data
        embed_data = []
        for x in data:
            embedding_x = []
            for idx in x:
                embedding_x.append(self.embedding_matrix[idx])
            embed_data.append(embedding_x)
        embed_data = np.array(embed_data)
        embed_data = embed_data.reshape(len(embed_data), -1)
        return embed_data

    def train(self):
        print('--------------TRAINING DECISION TREE MODEL--------------')
        model = self._build_model()
        embed_x_train = self._get_embedded_data(self.x_train)
        model.fit(embed_x_train, self.y_train)

        # Save model
        with open(os.path.join(self.base_path, 'ml_model.pickle'), 'wb') as f:
            pickle.dump(model, f)

    def test(self, x_test, y_test):
        embed_x_test = self._get_embedded_data(x_test)

        # Load model
        try:
            with open(os.path.join(self.base_path, 'ml_model.pickle'), 'rb') as f:
                model = pickle.load(f)
        except Exception as e:
            raise Exception('Could not load model decision tree model', e)

        y_pred = model.predict(embed_x_test)
        precision, recall, f1 = self._evaluation(y_test, y_pred)
        return precision, recall, f1

    def test_spark(self, x_test, y_test):
        embed_x_test = self._get_embedded_data(x_test)

        # Load model
        try:
            with open(os.path.join(self.base_path, 'ml_model.pickle'), 'rb') as f:
                model = pickle.load(f)
        except Exception as e:
            raise Exception('Could not load model decision tree model', e)

        y_pred = model.predict(embed_x_test)

        with open(os.path.join(self.base_path, 'spark', 'unique_messages.pkl'), 'rb') as f:
            unique_messages = pickle.load(f)

        idx_messages = dict()
        cur_idx = 0
        for msg, info in unique_messages.items():
            idx_messages[cur_idx] = info
            cur_idx += 1

        predicted = []
        actuals = []

        assert len(y_pred) == len(idx_messages)

        for idx, y_p in enumerate(y_pred):
            count = idx_messages[idx]['count']
            actual_label = idx_messages[idx]['label']

            predicted += [int(y_p)] * count
            actuals += [actual_label] * count

        precision, recall, f1 = self._evaluation(actuals, predicted)
        return precision, recall, f1


