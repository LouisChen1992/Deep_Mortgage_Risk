# Deep_Mortgage_Risk

This repository contains implementations of a five-layer neural network for predicting mortgage risk. Please read the paper [PDF](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2799443) for details. 

### Requirements
  * Python v3.5
  * TensorFlow v1.2+
  * Vtk v5.0+ (required for Mayavi)
  * Mayavi v4.5.0
  
For MacOSX, first install VTK with Homebrew, then install Mayavi with pip. 
```
$ /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install
$ brew install vtk
$ mkdir -p /Users/luyangchen/Library/Python/2.7/lib/python/site-packages
$ echo 'import site; site.addsitedir("/usr/local/lib/python2.7/site-packages")' >> /Users/luyangchen/Library/Python/2.7/lib/python/site-packages/homebrew.pth
$ brew install wxpython
$ sudo pip install mayavi
```
For Linux, first install VTK, then install Mayavi. 
```
$ sudo apt-get install python-vtk
$ sudo pip install mayavi
```
For more 3D visualization, please refer to [LINK](http://www.sethanil.com/python-for-reseach/5). 

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
$ python3 run.py --mode=sens_anlys_trio --logdir=model --sample_size=1
```

  * The first table below reports covariate (float) ranking by average absolute gradient for transition current -> paid off. 
  * The second table below reports covariate-pair (float) ranking by average absolute mixed gradient (estimated by finite difference) for transition current -> paid off. 
  * The third table below reports covariate-trio (float) ranking by average absolute mixed gradient (estimated by finite difference) for transition current -> paid off. 

| Feature                                                               | Ave. Absolute Gradient |
|:---------------------------------------------------------------------:|:----------------------:|
| Current Outstanding Balance                                           | 0.1878                 |
| Original Loan Balance                                                 | 0.0856                 |
| Original Interest Rate                                                | 0.0503                 |
| Current Interest Rate - National Mortgage Rate                        | 0.0478                 |
| Original Interest Rate - National Mortgate Rate                       | 0.0463                 |
| Zillow Zip Code Housing Price Change Since Origination                | 0.0386                 |
| Number of Times 30 Days Delinquent in Last 12 Months                  | 0.0384                 |
| Scheduled Interest and Principle Due                                  | 0.0364                 |
| Number of Times 60 Days Delinquent in Last 12 Months                  | 0.0362                 |
| Zillow Housing Prices (macro_data[21] != 0)                           | 0.0346                 |
| Time Since Origination                                                | 0.0306                 |
| ARM First Rate Reset Period                                           | 0.0295                 |
| FICO Score                                                            | 0.0293                 |
| Lagged Prime Prepayment Rate in Same Zip Code                         | 0.0292                 |
| Number of Times 90+ Days Delinquent in Last 12 Months                 | 0.0237                 |
| Current Interest Rate - Original Interest Rate                        | 0.0228                 |
| State Unemployment Rate                                               | 0.0214                 |
| Number of Days Delinquent                                             | 0.0195                 |
| ARM Contract Details (X_static[37])                                   | 0.0191                 |
| Lagged Prime Default Rate in Same Zip Code                            | 0.0190                 |
| Total Number of Prime Mortgages in Same Zip Code                      | 0.0190                 |
| ARM Contract Details (X_static[33])                                   | 0.0157                 |
| Number of Times Current in Last 12 Months                             | 0.0145                 |
| ARM Contract Details (X_static[32])                                   | 0.0133                 |
| Original Appraised Value                                              | 0.0132                 |
| Original Interest Rate - National Mortgage Rate at Origination        | 0.0129                 |
| LTV Ratio                                                             | 0.0116                 |
| Lagged Default Rate for Subprime Mortgages in Same Zip Code (>= 1000) | 0.0115                 |
| Zillow Housing Prices (macro_data[20] != 0)                           | 0.0110                 |
| ARM Contract Details (X_static[34])                                   | 0.0107                 |
| ...                                                                   | ...                    |
  
| Feature Pair                                                            | Ave. Absolute Mixed Gradient |
|:-----------------------------------------------------------------------:|:----------------------------:|
| Original Interest Rate, State Unemployment Rate                         | 0.00133                      |
| Original Interest Rate, Number of Times Current in Last 12 Months       | 0.00121                      |
| Original Interest Rate, Original Term of the Loan                       | 0.00100                      |
| FICO Score, Original Interest Rate                                      | 0.00087                      |
| Number of Times Current in Last 12 Months, Original Term of the Loan    | 0.00086                      |
| State Unemployment Rate, Original Term of the Loan                      | 0.00069                      |
| Number of Times Current in Last 12 Months, State Unemployment Rate      | 0.00067                      |
| FICO Score, Original Term of the Loan                                   | 0.00065                      |
| FICO Score, State Unemployment Rate                                     | 0.00060                      |
| Original Interest Rate, Original Interest Rate - National Mortgate Rate | 0.00059                      |
| FICO Score, Number of Times Current in Last 12 Months                   | 0.00059                      |
| Original Interest Rate, Original Loan Balance                           | 0.00055                      |
| Original Interest Rate, Margin for ARM Mortgages (Error)                | 0.00049                      |
| Original Loan Balance, Original Term of the Loan                        | 0.00048                      |
| Original Interest Rate, Current Outstanding Balance                     | 0.00043                      |
| Original Interest Rate, Scheduled Principle Due (Error)                 | 0.00042                      |
| LTV Ratio, Original Term of the Loan                                    | 0.00041                      |
| State Unemployment Rate, Scheduled Principle Due (Error)                | 0.00040                      |
| Original Term of the Loan, Current Outstanding Balance                  | 0.00040                      |
| LTV Ratio, Original Interest Rate                                       | 0.00040                      |
| State Unemployment Rate, Margin for ARM Mortgages (Error)               | 0.00038                      |
| Original Term of the Loan, Scheduled Principle Due (Error)              | 0.00037                      |
| FICO Score, Original Loan Balance                                       | 0.00037                      |
| Original Loan Balance, State Unemployment Rate                          | 0.00037                      |
| Original Term of the Loan, Num_IO_mon (error)                           | 0.00035                      |
| Original Interest Rate, Num_IO_mon (error)                              | 0.00035                      |
| Original Term of the Loan, Margin for ARM Mortgages (Error)             | 0.00035                      |
| Original Loan Balance, Current Outstanding Balance                      | 0.00033                      |
| Original Interest Rate, Lagged Prime Prepayment Rate in Same Zip Code   | 0.00032                      |
| Original Loan Balance, Number of Times Current in Last 12 Months        | 0.00031                      |
| ...                                                                     | ...                          |

| Feature Trio                                                                                                            | Ave. Absolute Mixed Gradient |
|:-----------------------------------------------------------------------------------------------------------------------:|:------------------------------:|
| Original Interest Rate, FICO Score, State Unemployment Rate                                                             | 0.000752                     |
| Original Interest Rate, Original Interest Rate - National Mortgate Rate, State Unemployment Rate                        | 0.000531                     |
| Original Loan Balance, Original Interest Rate, State Unemployment Rate                                                  | 0.000478                     |
| Original Loan Balance, Original Interest Rate, FICO Score                                                               | 0.000453                     |
| Current Outstanding Balance, Original Interest Rate, State Unemployment Rate                                            | 0.000372                     |
| Original Loan Balance, FICO Score, State Unemployment Rate                                                              | 0.000371                     |
| Current Outstanding Balance, Original Loan Balance, Original Interest Rate                                              | 0.000368                     |
| Original Interest Rate, Original Interest Rate - National Mortgate Rate, FICO Score                                     | 0.000363                     |
| Current Outstanding Balance, Original Interest Rate, FICO Score                                                         | 0.000344                     |
| Current Outstanding Balance, Original Loan Balance, FICO Score                                                          | 0.000311                     |
| Original Interest Rate, Lagged Prime Prepayment Rate in Same Zip Code, State Unemployment Rate                          | 0.000307                     |
| Current Outstanding Balance, Original Loan Balance, State Unemployment Rate                                             | 0.000296                     |
| Current Outstanding Balance, FICO Score, State Unemployment Rate                                                        | 0.000288                     |
| Original Interest Rate, Current Interest Rate - National Mortgage Rate, State Unemployment Rate                         | 0.000278                     |
| Original Interest Rate - National Mortgate Rate, FICO Score, State Unemployment Rate                                    | 0.000270                     |
| Original Interest Rate, FICO Score, Lagged Prime Prepayment Rate in Same Zip Code                                       | 0.000229                     |
| Original Loan Balance, Original Interest Rate, Original Interest Rate - National Mortgate Rate                          | 0.000218                     |
| Original Interest Rate, Current Interest Rate - National Mortgage Rate, FICO Score                                      | 0.000215                     |
| Original Interest Rate, Current Interest Rate - National Mortgage Rate, Original Interest Rate - National Mortgate Rate | 0.000212                     |
| Original Interest Rate, Time Since Origination, State Unemployment Rate                                                 | 0.000211                     |
| Original Interest Rate, Time Since Origination, FICO Score                                                              | 0.000184                     |
| FICO Score, Lagged Prime Prepayment Rate in Same Zip Code, State Unemployment Rate                                      | 0.000178                     |
| Current Interest Rate - National Mortgage Rate, FICO Score, State Unemployment Rate                                     | 0.000173                     |
| Original Loan Balance, Original Interest Rate, Lagged Prime Prepayment Rate in Same Zip Code                            | 0.000172                     |
| Original Interest Rate, Zillow Zip Code Housing Price Change Since Origination, State Unemployment Rate                 | 0.000168                     |
| Original Loan Balance, Original Interest Rate - National Mortgate Rate, State Unemployment Rate                         | 0.000167                     |
| Current Outstanding Balance, Original Interest Rate, Original Interest Rate - National Mortgate Rate                    | 0.000166                     |
| Original Interest Rate, Original Interest Rate - National Mortgate Rate, Lagged Prime Prepayment Rate in Same Zip Code  | 0.000163                     |
| Original Interest Rate, Zillow Zip Code Housing Price Change Since Origination, FICO Score                              | 0.000162                     |
| Original Loan Balance, Original Interest Rate - National Mortgate Rate, FICO Score                                      | 0.000160                     |
| ...                                                                                                                     | ...                          |

### Analysis
```
$ python3 run_anlys.py --logdir=model --task=1d_nonlinear --plot_out=plot # 1d Nonlinear 3D Plot
$ python3 run_anlys.py --logdir=model --task=2d_nonlinear --plot_out=plot # 2d Nonlinear 3D Plot
$ python3 run_anlys.py --logdir=model --task=2d_contour --plot_out=plot # 2d Nonlinear Contour Plot
```
