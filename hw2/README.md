# hw2
Semantic Relations Classification

## Preprocess
- first download ELMo pretrained weights and options from this website https://allennlp.org/elmo
- clone allennlp (for ELMo)
```
git clone https://github.com/allenai/allennlp.git
```
- follow the instruction in the elmo_preprocess.ipynb

- create required directories
```
mkdir models   # to store model
mkdir results  # to store result
```

## Training Model

### LSTM classifer
```
# choose standard lstm cell
python lstm_elmo.py --process_name [your process name] --cell lstm

# choose bidirectional lstm cell
python lstm_elmo.py --process_name [your process name] --cell bi-lstm

# choose multi-layer lstm cell
python lstm_elmo.py --process_name [your process name] --cell multi-lstm
```

### CNN classifier

code adapted from https://github.com/Shawn1993/cnn-text-classification-pytorch
```
python cnn_clf.py --process_name [your process name]
```

## Evaluate
```
perl semeval2010_task8_scorer-v1.2.pl results/proposed_answer_[your process name].txt answer_key.txt > evaluate_result.txt
```
