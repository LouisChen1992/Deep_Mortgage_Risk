# Deep_Mortgage_Risk

This repository contains implementations of a five-layer neural network for predicting mortgage risk. Please read the paper [PDF](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2799443) for details. 

### Requirements

  * Python v3.5
  * TensorFlow v1.2+

### Train

```
$ python3 run.py --mode=train --logdir=../output
```

### Test

```
$ python3 run.py --mode=test --logdir=../output
```
