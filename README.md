# Implementation of Ranknet to LambdaRank in Tensorflow2.0

**This repository has OpenAi GPT-2 pre-training and sequence generation implementation in tensorflow 2.0, **


**Requirements**

*  python >= 3.6
*  setuptools==41.0.1
*  ftfy==5.6
*  tqdm==4.32.1
*  Click==7.0
*  tensorflow-gpu==2.0.0
*  numpy==1.16.4

**Setup**

```
$ git clone https://github.com/akanyaani/ranknet-tensorflow2.0
$ cd ranknet-tensorflow2.0
$ pip install -r requirements.txt
```

Pre-Training model on sample data available in repository
```
$ python pre_process.py --help

Options:
  --data-dir TEXT           training data path  [default: /data/rank_data]
  --per-file-limit INTEGER  no of example per tfrecords  [default: 50000]
  --help                    Show this message and exit.
  
```

Preprocessing and and creating the TF Records of MSLR Data

```
>> python pre_process.py --data-dir data/path
```



```
$ python train_model.py --help

Options:
  --data-path TEXT       out directory  [default: ./data/tf_records]
  --out-dir TEXT         tf records path  [default:
                         /media/akanyaani/Disk2/ranknet]
  --algo TEXT            LTR algo name  [default: ranknet]
  --ranknet-type TEXT    Ranknet type (default or factor)  [default: default]
  --optimizer TEXT       optimizer type  [default: adam]
  --window-size INTEGER  optimizer type  [default: 512]
  --batch-size INTEGER   optimizer type  [default: 128]
  --lr FLOAT             learning rate  [default: 0.0001]
  --graph-mode BOOLEAN   graph execution  [default: True]
  --help                 Show this message and exit.

  
  
$ python train_model.py --data-path /data/path \
                        --out-dir /model/data/path \
                        --algo lambdarank \
                        --batch-size=32 \
                        --batch-size 128 --lr 1e-4 \
                        --graph-mode True
```

Distributed training on multiple gpu.
```
>> python train_gpt2.py --num-layers=8 --embedding-size=768 --batch-size=32 --distributed=Ture
```

Start TensorBoard through the command line.
```
$ tensorboard --logdir /log
```

After pretraining your model, you can generate sequences by giving some context to model.
Open this notebook and load the pretrained model and pass context to model it will return the generated sequence.

```
$ sequence_generator.ipynb
```

**References:**

* ["Microsoft Learning to Rank Datasets"](https://www.microsoft.com/en-us/research/project/mslr/)
* ["From RankNet to LambdaRank"](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf)
* ["Tensorflow Transformers"](https://www.tensorflow.org/beta/tutorials/text/transformer)


**Contribution**

* Your issues and PRs are always welcome.

**Author**

* Abhay Kumar
* Author Email : akanyaani@gmail.com
* Follow me on [Twitter](https://twitter.com/akanyaani)

**License**

* [MIT](https://github.com/akanyaani/gpt-2-tensorflow2.0/blob/master/LICENSE)
