# OpenNMT-py: Open-Source Neural Machine Translation

This repo implements custom LSTM and SCRN models for machine translation. Instructions and results are below

## Requirements

Install `OpenNMT-py` from `pip`:
```bash
pip install OpenNMT-py
```

or from the sources:
```bash
git clone https://github.com/OpenNMT/OpenNMT-py.git
cd OpenNMT-py
python setup.py install
```

Note!: If you have MemoryError in the install try to use `pip` with `--no-cache-dir`.

*(Optional)* some advanced features (e.g. working audio, image or pretrained models) requires extra packages, you can install it with:
```bash
pip install -r requirements.opt.txt
```

Note:

- some features require Python 3.5 and after (eg: Distributed multigpu, entmax)
- we currently only support PyTorch 1.4

### Step 0: Download the data

Information about the data can be found in the below links
```
mkdir -p data/multi30k
wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz &&  tar -xf training.tar.gz -C data/multi30k && rm training.tar.gz
wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz && tar -xf validation.tar.gz -C data/multi30k && rm validation.tar.gz
wget http://www.quest.dcs.shef.ac.uk/wmt17_files_mmt/mmt_task1_test2016.tar.gz && tar -xf mmt_task1_test2016.tar.gz -C data/multi30k && rm mmt_task1_test2016.tar.gz
```

### Step 1: Preprocess the data

```
for l in en de; do for f in data/multi30k/*.$l; do if [[ "$f" != *"test"* ]]; then sed -i "$ d" $f; fi;  done; done
for l in en de; do for f in data/multi30k/*.$l; do perl tools/tokenizer.perl -a -no-escape -l $l -q  < $f > $f.atok; done; done
onmt_preprocess -train_src data/multi30k/train.en.atok -train_tgt data/multi30k/train.de.atok -valid_src data/multi30k/val.en.atok -valid_tgt data/multi30k/val.de.atok -save_data data/multi30k.atok.low -lower
```

### Step 2: Train the model

To train baseline Pytorch LSTM model, run:
```
python train.py -data data/multi30k.atok.low -save_model multi30k_model_baseline -gpu_ranks 0 -dropout 0 -enc_layers 1 -dec_layers 1
```

To train custom implementaion of LSTM, run:
```
python train.py -data data/multi30k.atok.low -save_model multi30k_model_customLSTM -gpu_ranks 0 --rnn_type custom_LSTM -dropout 0 -enc_layers 1 -dec_layers 1
```

To train SCRN model, run:
```
python train.py -data data/multi30k.atok.low -save_model multi30k_model_SCRN -gpu_ranks 0 --rnn_type SCRN -dropout 0 -enc_layers 1 -dec_layers 1
```


### Step 3: Translate the sentences

Change -model parameter to YOUR_MODEL and -output to YOUR_OUTPUT
```
onmt_translate -gpu 0 -model multi30k_model_*_e13.pt -src data/multi30k/test2016.en.atok -tgt data/multi30k/test2016.de.atok -replace_unk -verbose -output multi30k.test.pred.atok
```

### Step 4: Evaluate
Change to YOUR_OUTPUT

```
perl tools/multi-bleu.perl data/multi30k/test2016.de.atok < multi30k.test.pred.atok
```

### Model Parameters

LSTM: \
Hidden size = 500 \
Embedding Size = 500 \
Number of layers = 1 \
Dropout = 0 

SCRN: \
Hidden size = 460 \
Context size = 40 \
Alpha = 0.95 \
Embedding Size = 500 \
Number of layers = 1 \
Dropout = 0 


### Results

Model | Pytorch LSTM | Custom LSTM | SCRN
--- | --- | --- | --- 
Bleu | 22.88 | 21.98 | 12

#
