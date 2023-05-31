import os
import pickle
import re
import sys
import contractions
from time import time
from nltk import corpus
from nerlogparser.nerlogparser import Nerlogparser

from grammar import LogGrammar

class GroundTruth(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.parser = Nerlogparser()
        self.stopwords = corpus.stopwords.words('english')

    @staticmethod
    def __loading(number):
        if number % 1000 == 0:
            s = str(number) + ' ...'
            print(s, end='')
            print('\r', end='')

    @staticmethod
    def __read_wordlist(log_type):
        # read word list of particular log type
        current_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'wordlist'))

        # open word list files in the specified directory
        wordlist_path = os.path.join(current_path, log_type + '.txt')
        with open(wordlist_path, 'r') as f:
            wordlist_temp = f.readlines()

        # get word list
        wordlist = []
        for wl in wordlist_temp:
            wordlist.append(wl.strip())

        return wordlist

    def __get_preprocessed_logs(self, log_file):
        # parse log files
        parsed_logs = self.parser.parse_logs(log_file)

        return parsed_logs

    def only_letter(self, word):
        for c in word:
            if not c.isalpha():
                return False
        return True

    def get_content_as_tokens(self, content):
        content = content.strip()
        content_tokens = content.split()
        return content_tokens

    def __preprocess(self, message):
        message = message.lower()
        content_tokens = self.get_content_as_tokens(message)

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
            if self.only_letter(temp) and temp != '' and len(temp) > 1:
                filtered_tokens.append(temp)

        return filtered_tokens

    @staticmethod
    def __set_anomaly_label(wordlist, parsed_logs):
        anomaly_label = {}

        # check sentiment for each log line
        for line_id, parsed in parsed_logs.items():
            log_lower = parsed['message'].lower().strip()

            # 0 = negative
            # 1 = positive
            label = 1
            for word in wordlist:
                # negative sentiment
                if word in log_lower:
                    label = 0
                    anomaly_label[line_id] = label
                    break

            # positive sentiment
            if label == 1:
                anomaly_label[line_id] = label

        return anomaly_label

    def __save_groundtruth(self, groundtruth):
        current_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets', self.dataset))
        groundtruth_file = os.path.join(current_path, 'log.all.pickle')
        with open(groundtruth_file, 'wb') as handle:
            pickle.dump(groundtruth, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __preprocess_logs(self, file_path):
        processed_logs = []
        with open(file_path, 'r') as f:
            for line in f:
                processed_line = self.__preprocess(line)
                processed_logs.append(processed_line)
        return processed_logs

    def get_ground_truth(self):
        # get log file
        current_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets',
                                                    self.dataset, 'logs'))
        log_files = os.listdir(current_path)

        groundtruth = {}
        groundtruth_id = 0
        for log_file in log_files:
            print('\nProcessing', log_file, '...')
            # set path
            file_path = os.path.join(current_path, log_file)

            parsed_logs = self.__get_preprocessed_logs(file_path)

            # get log type
            log_type = log_file.split('.')[0].lower()

            # set label for each line in a log file
            wordlist = self.__read_wordlist(log_type)
            print('\nProcessing', log_file, '...')

            # get label
            anomaly_label = self.__set_anomaly_label(wordlist, parsed_logs)
            processed_logs = self.__preprocess_logs(file_path)
            assert len(processed_logs) == len(parsed_logs)
            for line_id, label in anomaly_label.items():
                preprocessed_message = self.__preprocess(parsed_logs[line_id]['message'])

                groundtruth[groundtruth_id] = {
                    'message': processed_logs[line_id],
                    'label': label
                }
                groundtruth_id += 1


        # save ground truth
        self.__save_groundtruth(groundtruth)

    def __save_groundtruth_temp(self, groundtruth, groundtruth_file):
        current_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets', self.dataset))
        groundtruth_file = os.path.join(current_path, '{}.pickle'.format(groundtruth_file))
        with open(groundtruth_file, 'wb') as handle:
            pickle.dump(groundtruth, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_ground_truth_warn(self):
        # get log file
        current_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets',
                                                    self.dataset, 'logs'))
        log_files = os.listdir(current_path)

        groundtruth = {}
        groundtruth_id = 0
        for log_file in log_files:
            # set path
            file_path = os.path.join(current_path, log_file)

            # log parsing
            parsed_logs = self.__get_preprocessed_logs(file_path)
            processed_logs = self.__preprocess_logs(file_path)

            assert len(parsed_logs) == len(processed_logs)
            for line_id, entities in parsed_logs.items():
                label = 0
                flag = 0
                if 'status' in entities.keys() or 'service' in entities.keys():
                    try:
                        if 'WARN' in entities['status']:
                            label = 0
                            flag = 1

                    except KeyError:
                        if 'WARN' in entities['service']:
                            label = 0
                            flag = 1

                if flag == 0:
                    label = 1

                preprocessed_message = self.__preprocess(entities['message'])
                groundtruth[groundtruth_id] = {
                    'message': processed_logs[line_id],
                    'label': label
                }
                groundtruth_id += 1
                self.__loading(groundtruth_id)

        # save ground truth
        self.__save_groundtruth(groundtruth)

    def get_ground_truth_negative_log_levels(self):
        # get log file
        current_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets',
                                                    self.dataset, 'logs'))
        log_files = os.listdir(current_path)

        wordlist = self.__read_wordlist('hadoop')

        groundtruth = {}
        groundtruth_id = 0
        for log_file in log_files:
            # set path
            file_path = os.path.join(current_path, log_file)

            with open(file_path, 'r') as f:
                for line in f:
                    label = 1
                    if re.search('WARN|ERROR|FATAL', line.strip()):
                        label = 0

                    lower_line = line.lower().strip()
                    for word in wordlist:
                        if word in lower_line:
                            label = 0
                            break

                    processed_line = self.__preprocess(line)


                    groundtruth[groundtruth_id] = {
                        'message': processed_line,
                        'label': label
                    }
                    groundtruth_id += 1
                    self.__loading(groundtruth_id)

        # save ground truth
        self.__save_groundtruth(groundtruth)

    def get_ground_truth_dash(self, log_type):
        # get log file
        current_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',
                                                    'datasets', self.dataset, 'logs'))
        log_files = os.listdir(current_path)

        groundtruth = {}
        groundtruth_id = 0
        grammar_parser = LogGrammar(log_type)

        for log_file in log_files:
            # set path
            file_path = os.path.join(current_path, log_file)

            with open(file_path, 'r') as f:
                for line in f:
                    # parse log entry
                    if log_type == 'blue-gene':
                        parsed = grammar_parser.parse_bluegenelog(line)

                        # check 'sock' entity
                    label = 0
                    if 'sock' in parsed.keys():
                        if parsed['sock'] != '-':
                            label = 0
                        elif parsed['sock'] == '-':
                            label = 1

                    # check 'message' entity
                    if 'message' in parsed.keys():
                        preprocessed_message = self.__preprocess(parsed['message'])
                    else:
                        preprocessed_message = 'null'

                    processed_line = self.__preprocess(line)
                    groundtruth[groundtruth_id] = {
                        'message': processed_line,
                        'label': label
                    }
                    groundtruth_id += 1
                    self.__loading(groundtruth_id)

        # save ground truth
        self.__save_groundtruth(groundtruth)

if __name__ == '__main__':
    dataset_list = ['casper-rw', 'dfrws-2009-jhuisi', 'dfrws-2009-nssal', 'honeynet-challenge7',
                    'blue-gene', 'zookeeper', 'hadoop', 'honeynet-challenge5']
    if len(sys.argv) < 2:
        print('Please input dataset name.')
        print('python groundtruth.py dataset_name')
        print('Supported datasets:', dataset_list)
        sys.exit(1)

    else:
        start = time()
        dataset_name = sys.argv[1]
        gt = GroundTruth(dataset_name)

        if dataset_name in ['blue-gene']:
            gt.get_ground_truth_dash(dataset_name)

        elif dataset_name in ['zookeeper', 'hadoop']:
            gt.get_ground_truth_negative_log_levels()

        elif dataset_name in ['casper-rw', 'dfrws-2009-jhuisi', 'dfrws-2009-nssal', 'honeynet-challenge7',
                              'honeynet-challenge5', 'windows', 'spark']:
            gt.get_ground_truth()

        # print runtime
        duration = time() - start
        minute, second = divmod(duration, 60)
        hour, minute = divmod(minute, 60)
        print("Runtime: %d:%02d:%02d" % (hour, minute, second))
