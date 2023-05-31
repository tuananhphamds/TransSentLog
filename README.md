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
Link download: 
Unzip the file and copy `datasets` and `glove` to `TransSentLog` folder

### Install experiment environment (GRU is needed)
1. Install Anaconda version: 4.11.0
2. Create an environment with Python 3.8, 

### Evaluation results from the paper
Details of experiment results are located in `EVALUATION_RESULTS` folder
- Table4
- Table5
- Table6
- Figure4
- Figure5
- Figure7
