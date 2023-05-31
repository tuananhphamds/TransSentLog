import os
import numpy as np
import pickle
import sys
import tensorflow as tf

from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils import shuffle
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback

HOME = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
sys.path.append(HOME + '/lib')

from sampling import Sampling

class ShowLRate(Callback):
    def on_epoch_end(self, epoch, logs=None):
        try:
            lr = self.model.optimizer._decayed_lr(tf.float32).numpy()
        except:
            lr = self.model.optimizer.learning_rate
        print('Learning rate at epoch {}: {}'.format(epoch, lr))

class BaseModel(object):
    def __init__(self, embedding_data_dct, cfg):
        self.x_train = embedding_data_dct['x_train']
        self.y_train = embedding_data_dct['y_train']
        self.x_val = embedding_data_dct['x_val']
        self.y_val = embedding_data_dct['y_val']
        self.word_index = embedding_data_dct['word_index']
        self.embedding_matrix = embedding_data_dct['embedding_matrix']

        self.use_sampler = cfg['use_tomek']
        if self.use_sampler:
            self.sampling = Sampling('tomek-links')
            self.sampler = self.sampling.get_sampler()

        self.base_path = os.path.join(HOME, 'datasets')
        self.model = None

    def _evaluation(self, true_label, predicted_label):
        precision, recall, f1, _ = precision_recall_fscore_support(true_label, predicted_label, average='macro')

        return precision * 100, recall * 100, f1 * 100

    def train(self):
        # ---------------------------BUILD AND COMPILE MODEL-------------------------------
        model = self._build_model()

        # ---------------------------INITIALIZE CALLBACKS----------------------------------
        model_file = os.path.join(self.base_path, 'best-model.hdf5')
        earlystop = EarlyStopping(monitor='loss', patience=10, verbose=1, mode='min')
        checkpoint = ModelCheckpoint(model_file, monitor='loss', verbose=1, save_best_only=True,
                                     save_weights_only=False, mode='min')

        # ---------------------------FIX IMBALANCED DATASET, TRAIN MODEL-------------------
        if self.use_sampler:
            sampled_data_path = os.path.join(self.base_path, 'train_val_resample.pickle')
            if os.path.exists(sampled_data_path):
                with open(sampled_data_path, 'rb') as f:
                    resample_pickle = pickle.load(f)
                    x_train_resample = resample_pickle['x_train_resample']
                    y_train_resample = resample_pickle['y_train_resample']
                    x_val = resample_pickle['x_val']
                    y_val = resample_pickle['y_val']
            else:
                # sample the data
                print('Resampling data ...')
                x_train_resample, y_train_resample = self.sampler.fit_resample(self.x_train, self.y_train)

                x_train_resample = np.asarray(x_train_resample)
                y_train_resample = to_categorical(y_train_resample)

                x_val = np.asarray(self.x_val)
                y_val = to_categorical(self.y_val)

                train_val = {
                    'x_train_resample': x_train_resample,
                    'y_train_resample': y_train_resample,
                    'x_val': x_val,
                    'y_val': y_val
                }

                with open(sampled_data_path, 'wb') as f:
                    pickle.dump(train_val, f, protocol=pickle.HIGHEST_PROTOCOL)

            x_train_resample, y_train_resample = shuffle(x_train_resample, y_train_resample)

            # train model
            model.fit(x_train_resample, y_train_resample, validation_data=(x_val, y_val),
                      batch_size=self.batch_size, epochs=self.epochs,
                      callbacks=[earlystop, checkpoint, ShowLRate()], shuffle=True, verbose=1)
        else:
            x_train = np.asarray(self.x_train)
            y_train = to_categorical(self.y_train)
            x_val = np.asarray(self.x_val)
            y_val = to_categorical(self.y_val)

            print('Shuffling data ...')
            x_train, y_train = shuffle(x_train, y_train)
            model.fit(x_train, y_train, validation_data=(x_val, y_val),
                      batch_size=self.batch_size, epochs=self.epochs,
                      callbacks=[earlystop, checkpoint, ShowLRate()], shuffle=True, verbose=1)

        self.model = model

    def _load_model(self, model_file):
        return load_model(model_file)

    def test(self, x_test, y_test):
        x_test = np.asarray(x_test)

        model_file = os.path.join(self.base_path, 'best-model.hdf5')
        model = self._load_model(model_file)

        y_prob = model.predict(x_test)
        y_pred = y_prob.argmax(axis=-1)

        precision, recall, f1 = self._evaluation(y_test, y_pred)

        return precision, recall, f1

    def test_spark(self, x_test, y_test):
        x_test = np.asarray(x_test)

        model_file = os.path.join(self.base_path, 'best-model.hdf5')
        model = self._load_model(model_file)

        y_prob = model.predict(x_test)
        y_pred = y_prob.argmax(axis=-1)

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

        precision, recall, f1= self._evaluation(actuals, predicted)
        return precision, recall, f1
