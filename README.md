# Deep_Mortgage_Risk

This repository contains implementations of a five-layer neural network for predicting mortgage risk. Please read the paper [PDF](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2799443) for details. 

### Requirements

  * Python v3.5
  * TensorFlow v1.2+

### Train, Validation & Test

```
$ python3 run.py --mode=train --logdir=output --num_epochs=10
$ python3 run.py --mode=test --logdir=output
```
The table below reports test loss for the best model (on validation set):

| Epoch | Train Loss | Validation Loss | Test Loss |
|:-----:|:----------:|:---------------:|:---------:|
| 9     | 0.1642     | 0.1930          | 0.1666    |

### Sensitivity Analysis

```
$ python3 run.py --mode=sens_anlys --logdir=output
```
