import os
import sys
import csv
import pickle

HOME = os.path.split(os.path.abspath(__file__))[0]
sys.path.append(HOME + '/lib')
sys.path.append(HOME + '/model')

from optparse import OptionParser
from embedding import WordEmbedding
from svmsentiment import SVMSentiment
from decisiontreesentiment import DecisionTreeSentiment
from cnnsentiment import CNNSentiment
from pylogsentiment import PyLogSentiment
from gruattentionsentiment import GRUAttentionSentiment
from transsentlog import TransSentLog
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Trainer:
    def __init__(self, datasets, cfg):
        self.cfg = cfg
        self.datasets = datasets
        self.dataset_path = os.path.join(HOME, 'datasets')
        self.result_path = os.path.join(HOME, 'results')

    def __get_embedding(self):
        word_embedding = WordEmbedding(self.datasets, self.cfg)
        embedding_data_dct = word_embedding.get_data_and_embedding()

        return embedding_data_dct

    def __read_test_set(self, dataset):
        test_path = os.path.join(self.dataset_path, dataset, 'test.pickle')
        with open(test_path, 'rb') as f:
            data = pickle.load(f)

        return data['x_test'], data['y_test']

    def __save_test_file_for_untrained_datasets(self, dataset, word_index):
        with open(os.path.join(self.dataset_path, '{}/log.all.pickle'.format(dataset)), 'rb') as f:
            ground_truths = pickle.load(f)

        data_list = []
        y_test = []
        for line_id, info in ground_truths.items():
            y_test.append(info['label'])
            data_list.append(info['message'])

        # convert to integer and padding
        x_pad = self.__get_numeric_embedding(data_list, word_index)

        data = {
            'x_test': x_pad,
            'y_test': y_test
        }
        with open(os.path.join(self.dataset_path, '{}/test.pickle'.format(dataset)), 'wb') as f:
            pickle.dump(data, f)

    def __get_numeric_embedding(self, data_list, word_index):
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
        data_pad = pad_sequences(numeric_data, maxlen=self.cfg['log_length'], padding='post', truncating='post')

        return data_pad

    def get_evaluation_file_name(self, evaluation_file):
        dir_path = evaluation_file.replace('.evaluation.csv', '')
        if not os.path.exists(os.path.join(dir_path)):
            os.makedirs(dir_path)

        count = len(os.listdir(dir_path))
        new_file_name = os.path.join(dir_path, 'evaluation_{}.csv'.format(count+1))
        return new_file_name

    def run(self):
        # ----------------------------------PREPROCESS DATA------------------------------------
        embedding_data_dct = self.__get_embedding()

        # ----------------------------------INITIALIZE MODEL-----------------------------------
        model_name = self.cfg['model']

        if model_name == 'svm':
            model = SVMSentiment(embedding_data_dct, self.cfg)
        elif model_name == 'decisiontree':
            model = DecisionTreeSentiment(embedding_data_dct, self.cfg)
        elif model_name == 'cnn':
            model = CNNSentiment(embedding_data_dct, self.cfg)
        elif model_name == 'pylogsentiment':
            model = PyLogSentiment(embedding_data_dct, self.cfg)
        elif model_name == 'gruattention':
            model = GRUAttentionSentiment(embedding_data_dct, self.cfg)
        elif model_name == 'transsentlog':
            model = TransSentLog(embedding_data_dct, self.cfg)
        else:
            raise Exception('Model {} is not supported'.format(model_name))

        # ----------------------------------TRAIN MODEL----------------------------------------
        model.train()

        # ----------------------------------EVALUATE MODEL-------------------------------------
        evaluation_file = os.path.join(self.result_path, model_name + '.evaluation.csv')
        evaluation_file = self.get_evaluation_file_name(evaluation_file)
        f = open(evaluation_file, 'wt', newline='')
        writer = csv.writer(f)
        writer.writerow(['Dataset', 'Model', 'Precision', 'Recall', 'F1'])

        untrained_datasets = ['spark', 'windows', 'honeynet-challenge5']
        self.datasets += untrained_datasets

        for dataset in self.datasets:
            if dataset in untrained_datasets:
                self.__save_test_file_for_untrained_datasets(dataset, embedding_data_dct['word_index'])

            x_test, y_test = self.__read_test_set(dataset)
            if dataset == 'spark':
                precision, recall, f1 = model.test_spark(x_test, y_test)
            else:
                precision, recall, f1 = model.test(x_test, y_test)
            writer.writerow([dataset, model_name, precision, recall, f1])

        f.close()

if __name__ == '__main__':
    parser = OptionParser(usage='Training a model')
    parser.add_option('-m', '--model',
                      action='store',
                      dest='model',
                      help='A method will be used for performance evaluation',
                      default='transsentlog')
    parser.add_option('-e', '--epochs', action='store', type='int', dest='epochs', default=20)
    parser.add_option('-t', '--tomek', action='store_false', dest='use_tomek', default=True)
    parser.add_option('-l', '--length', action='store', type='int', dest='log_length', default=10)
    parser.add_option('-d', '--dim', action='store', type='int', dest='dim', default=50)
    parser.add_option('-u', '--units', action='store', type='int', dest='units', default=256)
    parser.add_option('-r', '--dropout', action='store', type='float', dest='dropout', default=0.1)
    parser.add_option('-b', '--blocks', action='store', type='int', dest='num_blocks', default=2)
    parser.add_option('-a', '--heads', action='store', type='int', dest='num_heads', default=2)
    parser.add_option('-f', '--use_file_cfg', action='store_true', dest='use_file_cfg', default=False)
    parser.add_option('-s', '--batch_size', action='store', type='int', dest='batch_size', default=1024)
    parser.add_option('-x', '--train_rate', action='store', type='float', dest='train_rate', default=0.6)
    parser.add_option('-y', '--val_rate', action='store', type='float', dest='val_rate', default=0.2)
    parser.add_option('-z', '--max_num_words', action='store', type='int', dest='max_num_words', default=400000)

    (options, args) = parser.parse_args()

    cfg = dict()
    cfg['model'] = options.model
    cfg['epochs'] = options.epochs
    cfg['use_tomek'] = options.use_tomek
    cfg['log_length'] = options.log_length
    cfg['dim'] = options.dim
    cfg['units'] = options.units
    cfg['dropout'] = options.dropout
    cfg['num_blocks'] = options.num_blocks
    cfg['num_heads'] = options.num_heads
    cfg['batch_size'] = options.batch_size
    cfg['train_rate'] = options.train_rate
    cfg['val_rate'] = options.val_rate
    cfg['max_num_words'] = options.max_num_words

    datasets = ['casper-rw', 'dfrws-2009-jhuisi', 'dfrws-2009-nssal', 'honeynet-challenge7', 'zookeeper', 'hadoop']
    trainer = Trainer(datasets, cfg)
    trainer.run()