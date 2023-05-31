import os
import sys
import pickle
import re
import csv
import numpy as np
import contractions
from optparse import OptionParser
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

HOME = os.path.split(os.path.abspath(__file__))[0]
sys.path.append(HOME + '/model')
sys.path.append(HOME + '/lib')

from official.nlp import optimization
from transsentlog import Transformer
from averagepoolmask import AveragePoolMask
from integratedgradient import get_weights


class Tester:
    def __init__(self, cfg):
        self.log_file = cfg['log_file']
        self.output_file = cfg['output_file']
        self.base_path = os.path.join(HOME, 'datasets')
        self.log_length = cfg['log_length']
        self.cfg = cfg

    def __load_word_index(self):
        path = os.path.join(self.base_path, 'word_index.pickle')
        with open(path, 'rb') as f:
            word_index = pickle.load(f)

        return word_index

    def __get_numeric_padding(self, data_list, word_index):
        # get integer representation of log message based on word index
        numeric_data = []
        for message in data_list:
            numeric_message = []
            for word in message:
                try:
                    numeric_message.append(word_index[word])
                except KeyError:
                    numeric_message.append(0)

            numeric_data.append(numeric_message)

        # padding
        data_pad = pad_sequences(numeric_data, maxlen=self.log_length, padding='post', truncating='post')

        return data_pad

    def __get_content_as_tokens(self, content):
        content = content.strip()
        content_tokens = content.split()
        return content_tokens

    def __only_letter(self, word):
        for c in word:
            if not c.isalpha():
                return False
        return True

    def __preprocess(self, message):
        message = message.lower()
        content_tokens = self.__get_content_as_tokens(message)

        # Expand contractions (doesn't -> do not, I'll -> I will, I'd -> I would)
        contraction_tokens = []
        for word in content_tokens:
            if word != '':
                contraction_tokens += contractions.fix(word).split()

        # Filter words
        filtered_tokens = []
        for token in contraction_tokens:
            # Remove special characters at the end of a word (driver: -> driver, terminated! -> terminated)
            temp = re.sub('[^a-z0-9]+$', '', token)
            # Remove special characters at the beginning of a word (:driver -> driver, ~terminated -> terminated)
            temp = re.sub('^[^a-z0-9]+', '', temp)
            # Remove info and warn log levels
            if temp in ['info', 'warn']:
                continue
            # Remove month of the year
            temp = re.sub(
                'jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|jun(e)?|jul(y)?|aug(ust)?|sep(tember)?|oct(ober)?|nov(ember)?|dec(ember)?',
                '', temp)
            # Remove day of the week
            temp = re.sub('(mon|tues|wed(nes)?|thur(s)?|fri|sat(ur)?|sun)(day)?', '', temp)
            if self.__only_letter(temp) and temp != '' and len(temp) > 1:
                filtered_tokens.append(temp)

        return filtered_tokens

    def __preprocess_log_file(self):
        processed_messages = []
        with open(self.log_file, 'r') as f:
            for message in f:
                processed_messages.append(self.__preprocess(message))
        return processed_messages

    def __create_optimizer(self):
        steps_per_epoch = 123000 // 1024
        num_train_steps = steps_per_epoch * 20
        num_warmup_steps = int(0.2 * num_train_steps)
        init_lr = 1e-4
        optimizer = optimization.create_optimizer(init_lr=init_lr,
                                                  num_train_steps=num_train_steps,
                                                  num_warmup_steps=num_warmup_steps,
                                                  optimizer_type='adamw')
        return optimizer

    def __save_to_csv(self, y_pred):
        # output csv file
        if self.output_file is None:
            self.output_file = os.path.join(self.log_file + '.anomaly-results.csv')

        if os.path.exists(self.output_file):
            os.remove(self.output_file)
        f_csv = open(self.output_file, 'wt', newline='')
        writer = csv.writer(f_csv)
        writer.writerow(['Prediction', 'Log message'])

        # save anomaly detection results to csv
        with open(self.log_file, 'r') as f:
            for line_index, line in enumerate(f):
                if line not in ['\n', '\r\n']:
                    writer.writerow([y_pred[line_index], line.rstrip()])

        print('Write detection results to:', self.output_file)
        f_csv.close()

    def __save_weights(self, idxs, weights, x_pad):
        # LOAD INDEX_WORD dictionary
        with open(os.path.join(HOME, 'datasets/index_word.pickle'), 'rb') as f:
            index_word = pickle.load(f)
            index_word[0] = '<UNK>'

        # INITIALIZE OUTPUT FILE NAME
        if self.output_file is None:
            self.output_file = os.path.join(self.log_file + '.intepretation_weights.csv')
        else:
            self.output_file = 'datasets/intepretation_weights.csv'

        # LOAD ALL LOGS
        with open(self.log_file, 'r') as f:
            logs = f.readlines()

        # SAVE WEIGHTS TO CSV
        if os.path.exists(self.output_file):
            os.remove(self.output_file)
        f_csv = open(self.output_file, 'wt', newline='')
        writer = csv.writer(f_csv)
        writer.writerow(['Index', 'Log message', 'Processed log message', 'Normalized weights'])

        for lineidx, idx in enumerate(idxs):
            original_log = logs[idx]
            processed_log = []
            for word_idx in x_pad[idx]:
                processed_log.append(index_word[word_idx])
            processed_log = ' '.join(processed_log)
            log_weights = weights[lineidx]
            writer.writerow([idx+1, original_log, processed_log, log_weights])

        print('Write intepretation weights to {}'.format(self.output_file))
        f_csv.close()


    def run(self):
        # -----------------PREPROCESS-----------------
        data_list = self.__preprocess_log_file()
        word_index = self.__load_word_index()
        x_pad = self.__get_numeric_padding(data_list, word_index)

        # -----------------ANOMALY DETECTION-----------------
        model_file = os.path.join(self.base_path, 'best-model.hdf5')
        model = load_model(model_file, custom_objects={
            'Transformer': Transformer,
            'AveragePoolMask': AveragePoolMask,
            'AdamWeightDecay': self.__create_optimizer()
        })

        x_test = np.asarray(x_pad)
        y_prob = model.predict(x_test)
        y_pred = y_prob.argmax(axis=-1)

        self.__save_to_csv(y_pred)

        # -------------------INTEGRATED GRADIENTS---------------------------
        if self.cfg['use_ig']:
            idxs, scaled_weights = get_weights(model, x_test, y_pred, self.cfg['ig_steps'])
            self.__save_weights(idxs, scaled_weights, x_pad)


def check_config(options):
    cfg = dict()

    model_file = options.model_file
    model_path = os.path.join(HOME, model_file)
    if not os.path.exists(model_path):
        raise FileNotFoundError('Model file {} does not exist'.format(model_path))
    cfg['model_file'] = model_file

    log_file = options.log_file
    log_path = os.path.join(HOME, log_file)
    if not os.path.exists(log_path):
        raise FileNotFoundError('Log file {} does not exist'.format(log_path))
    cfg['log_file'] = log_file

    cfg['log_length'] = options.log_length
    cfg['output_file'] = options.output_file
    cfg['use_ig'] = options.use_ig
    cfg['ig_steps'] = options.ig_steps
    return cfg


if __name__ == '__main__':
    parser = OptionParser(usage='Classify log events and retrieve contribution weights')
    parser.add_option('-f', '--model_file', action='store', dest='model_file', default='datasets/best-model.hdf5')
    parser.add_option('-l', '--log_length', action='store', type='int', dest='log_length', default=10)
    parser.add_option('-i', '--log_file', action='store', dest='log_file', default='')
    parser.add_option('-o', '--output_file', action='store', dest='output_file', default='datasets/output.csv')
    parser.add_option('-e', '--use_ig', action='store_true', dest='use_ig', default=False)
    parser.add_option('-n', '--ig_steps', action='store', type='int', dest='ig_steps', default=50)

    (options, args) = parser.parse_args()
    cfg = check_config(options)
    tester = Tester(cfg)
    tester.run()