# Implementation of Ranknet to LambdaRank in TensorFlow2.0

**This repository has From RankNet to LambdaRank implementation in tensorflow 2.0, **


**Requirements**

* tqdm==4.32.1
* numpy==1.16.4
* Click==7.0
* tensorflow_gpu==2.1.0

**Setup**

```
$ git clone https://github.com/akanyaani/ranknet-tensorflow2.0
$ cd ranknet-tensorflow2.0
$ pip install -r requirements.txt
```
Download data from here https://www.microsoft.com/en-us/research/project/mslr/ and pass any of the fold to pre_process.

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

Training learning to rank model.

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
                        --window-size=512 \
                        --batch-size 128 --lr 1e-4 \
                        --graph-mode True
```

Start TensorBoard through the command line.
```
$ tensorboard --logdir /model/data/path
```

**References:**

* ["Microsoft Learning to Rank Datasets"](https://www.microsoft.com/en-us/research/project/mslr/)
* ["From RankNet to LambdaRank"](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf)
* ["Tensorflow-LTR"](https://github.com/ChenglongChen/tensorflow-LTR)



**Contribution**

* Your issues and PRs are always welcome.

**Author**

* Abhay Kumar
* Author Email : akanyaani@gmail.com
* Follow me on [Twitter](https://twitter.com/akanyaani)

**License**

* [MIT](https://github.com/akanyaani/gpt-2-tensorflow2.0/blob/master/LICENSE)
