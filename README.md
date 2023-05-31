# TransSentLog
TransSentLog implementation
## TransSentLog: Interpretable Anomaly Detection Using Transformer and Sentiment Analysis on Individual Log Event

TransSentLog is a supervised-learning method which combines Transfomer and sentiment analysis to detect anomalies on individual log events. When a system is functioning normally, event logs generally convey positive sentiment. However, if unexpected behaviors like errors or failures occur, negative sentiment can be detected. 

## How to use the repository
### Clone the repository
```bash
git clone https://github.com/tuananhphamds/TransSentLog.git
cd TransSentLog
```

### Datasets
Link download: https://bit.ly/42eaYoK

DATA.zip file contains:
- All datasets and their groundtruths
- Trainedmodel file (best-model.hdf5)
- log test file (log_test.txt)

Unzip the file and copy `datasets` and `glove` to `TransSentLog` folder

### Install experiment environment (GRU is needed)
1. Install Anaconda version: 4.11.0
2. Create an environment with Python 3.8, cudatoolkit and cudnn
```bash
conda create -n transsentlog python=3.8 cudatoolkit cudnn
pip install -r requirements.txt
```

### How to evaluation, train, and test model
There are three files: train.py, test.py, and evaluation.py (run `n` trials)

To train TransSentLog with default hyperparameters, run:
```bash
python train.py --model transsentlog 
```

or with modified hyperparameters:
```bash
python train.py --model transsentlog --length 10 --tomek True --dim 50 --epochs 20 --dropout 0.1 --batch_size 1024 
```

To test TransSentLog with a given log file, run:
```bash
python test.py --model_file datasets/best-model.hdf5 --log_file datasets/log_test.txt --output_file datasets/output.csv --use_ig True --ig_steps 50
```
The prediction results will be saved in `datasets/output.csv` and prediction intepretation is saved in `datasets/intepretation_weights.csv`

To run the evaluation with n_trials and model config (config.json), run:
```bash
python evaluation --use_file_cfg True
```

### Evaluation results from the paper
Details of experiment results are located in `EVALUATION_RESULTS` folder
- Table4
- Table5
- Table6
- Figure4
- Figure5
- Figure7
