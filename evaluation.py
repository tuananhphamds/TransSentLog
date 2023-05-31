from optparse import OptionParser
from multiprocessing import Process
import json
from train import Trainer


def run_evaluation(datasets, cfg):
    print('----------Running evaluation with config: ')
    for key, value in cfg.items():
        print('{}: {}'.format(key, value))
    trainer = Trainer(datasets, cfg)
    trainer.run()


def check_config(cfg):
    if 'train_rate' not in cfg:
        cfg['train_rate'] = 0.6
    if 'val_rate' not in cfg:
        cfg['val_rate'] = 0.2
    if 'max_num_words' not in cfg:
        cfg['max_num_words'] = 400000
    if 'use_tomek' not in cfg:
        cfg['use_tomek'] = True
    if 'ntrials' not in cfg:
        cfg['n_trials'] = 20
    if 'epochs' not in cfg:
        cfg['epochs'] = 20
    if 'dim' not in cfg:
        cfg['dim'] = 50
    return cfg


if __name__ == "__main__":
    parser = OptionParser(usage='Conduct experiments with n trials')
    parser.add_option('-m', '--model',
                      action='store',
                      dest='model',
                      help='A method will be used for performance evaluation',
                      default='transsentlog')
    parser.add_option('-n', '--ntrials', action='store', type='int', dest='ntrials', default=20)
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

    datasets = ['casper-rw', 'dfrws-2009-jhuisi', 'dfrws-2009-nssal', 'honeynet-challenge7', 'zookeeper', 'hadoop']
    if options.use_file_cfg:
        try:
            with open('config.json', 'r') as f:
                cfg = json.load(f)
        except Exception as e:
            raise Exception('Cannot load the model config file', e)

        for idx, m_cfg in cfg.items():
            print('----------Model id: {}'.format(idx))
            m_cfg = check_config(m_cfg)
            for n in range(m_cfg['ntrials']):
                print('Trial {}:'.format(n+1))
                p_run = Process(target=run_evaluation, args=(datasets, m_cfg))
                p_run.start()
                p_run.join()
    else:
        cfg = dict()
        cfg['model'] = options.model
        cfg['ntrials'] = options.ntrials
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
        p_run = Process(target=run_evaluation, args=(datasets, cfg))
        p_run.start()
        p_run.join()