# Deep_Mortgage_Risk

This repository contains implementations of a five-layer neural network for predicting mortgage risk. Please read the paper [PDF](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2799443) for details. 

### Requirements
  * Python v3.5
  * TensorFlow v1.2+

### Train, Validation & Test
```
$ python3 run.py --mode=train --logdir=model --num_epochs=10
$ python3 run.py --mode=test --logdir=model
```
The table below reports test loss for the best model (on validation set):

| Epoch | Train Loss | Validation Loss | Test Loss |
|:-----:|:----------:|:---------------:|:---------:|
| 9     | 0.1642     | 0.1930          | 0.1666    |

### Sensitivity Analysis
```
$ python3 run.py --mode=sens_anlys --logdir=model
$ python3 run.py --mode=sens_anlys_pair --logdir=model --sample_size=1
```
The table below reports covariate ranking by average absolute gradient for transition current -> paid off: 

| Feature                                                        | Ave. Absolute Gradient |
|:--------------------------------------------------------------:|:----------------------:|
| current outstanding balance                                    | 0.1878                 |
| original loan amount                                           | 0.0856                 |
| original interest rate                                         | 0.0503                 |
| current interest rate as well - national mortgage rate as well | 0.0478                 |
| initial interest rate - national mortgate rate                 | 0.0463                 |
| housing price increase/decrease since origination              | 0.0386                 |
| number of occurrences of 3                                     | 0.0384                 |
| scheduled principle and interest due                           | 0.0364                 |
| number of occurrences of 6                                     | 0.0362                 |
| zillow housing prices (macro_data[21] != 0)                    | 0.0346                 |
| total days delinquent >= 160                                   | 0.0322                 |
| time since origination                                         | 0.0306                 |
| ARM first rate reset period                                    | 0.0295                 |
| fico score                                                     | 0.0293                 |
| lagged prepayment rate                                         | 0.0292                 |
| number of occurrences of 9                                     | 0.0237                 |
| current interest rate - original interest rate                 | 0.0228                 |
| total days delinquent >= 130 & < 160                           | 0.0225                 |
| state unemployment rate                                        | 0.0214                 |
| total days delinquent                                          | 0.0195                 |
| ARM contract details (X_static[37])                            | 0.0191                 |
| lagged default rate                                            | 0.0190                 |
| total number of prime mortgages currently alive                | 0.0190                 |
| lien type == 5                                                 | 0.0185                 |
| channel type == 8                                              | 0.0181                 |
| current delinquency status == 6                                | 0.0180                 |
| current delinquency status == 3                                | 0.0177                 |
| current delinquency status == 9                                | 0.0177                 |
| total days delinquent > 100 & < 130                            | 0.0172                 |
| ARM contract details (X_static[33])                            | 0.0157                 |
| ...                                                            | ...                    |

### Analysis
  * 1d Nonlinear Relationship
```
$ python3 run_anlys.py --logdir=model --task=1d_nonlinear --plot_out=plot
```

  * 2d Nonlinear Relationship
```
$ python3 run_anlys.py --logdir=model --task=2d_nonlinear --plot_out=plot
```
