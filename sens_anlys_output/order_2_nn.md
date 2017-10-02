- The first table below reports covariate-pair (float) ranking by average absolute mixed gradient (estimated by finite difference) for transition current -> paid off. 
- The second table below reports covariate-pair (float) ranking by average absolute mixed gradient (estimated by finite difference) for transition current -> 30 days delinquent.

------
### current -> paid off

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
| Original Term of the Loan, Num_IO_mon (Error)                           | 0.00035                      |
| Original Interest Rate, Num_IO_mon (Error)                              | 0.00035                      |
| Original Term of the Loan, Margin for ARM Mortgages (Error)             | 0.00035                      |
| Original Loan Balance, Current Outstanding Balance                      | 0.00033                      |
| Original Interest Rate, Lagged Prime Prepayment Rate in Same Zip Code   | 0.00032                      |
| Original Loan Balance, Number of Times Current in Last 12 Months        | 0.00031                      |
| ...                                                                     | ...                          |

------
### current -> 30 days delinquent

| Feature Pair                                                                      | Ave. Absolute Mixed Gradient |
|:---------------------------------------------------------------------------------:|:----------------------------:|
| FICO Score, Original Term of the Loan                                             | 0.00168                      |
| Original Interest Rate, Original Term of the Loan                                 | 0.00128                      |
| Number of Times Current in Last 12 Months, Original Term of the Loan              | 0.00127                      |
| State Unemployment Rate, Original Term of the Loan                                | 0.00091                      |
| Original Term of the Loan, Scheduled Principle Due (Error)                        | 0.00066                      |
| Original Term of the Loan, Num_IO_mon (Error)                                     | 0.00066                      |
| Original Term of the Loan, Margin for ARM Mortgages (Error)                       | 0.00060                      |
| FICO Score, Original Interest Rate                                                | 0.00058                      |
| LTV Ratio, Original Term of the Loan                                              | 0.00048                      |
| Original Term of the Loan, Current Outstanding Balance                            | 0.00040                      |
| Original Interest Rate, State Unemployment Rate                                   | 0.00037                      |
| Original Loan Balance, Original Term of the Loan                                  | 0.00035                      |
| FICO Score, LTV Ratio                                                             | 0.00033                      |
| FICO Score, Number of Times Current in Last 12 Months                             | 0.00030                      |
| FICO Score, State Unemployment Rate                                               | 0.00028                      |
| Original Interest Rate, Number of Times Current in Last 12 Months                 | 0.00026                      |
| Original Interest Rate - National Mortgate Rate, Original Term of the Loan        | 0.00025                      |
| LTV Ratio, Original Interest Rate                                                 | 0.00025                      |
| Original Interest Rate, Original Interest Rate - National Mortgate Rate           | 0.00024                      |
| FICO Score, Current Outstanding Balance                                           | 0.00024                      |
| FICO Score, Original Loan Balance                                                 | 0.00021                      |
| Original Term of the Loan, Zillow Zip Code Housing Price Change Since Origination | 0.00019                      |
| Original Interest Rate, Current Outstanding Balance                               | 0.00016                      |
| LTV Ratio, Number of Times Current in Last 12 Months                              | 0.00016                      |
| Number of Times Current in Last 12 Months, State Unemployment Rate                | 0.00015                      |
| LTV Ratio, State Unemployment Rate                                                | 0.00015                      |
| Original Interest Rate, Original Loan Balance                                     | 0.00015                      |
| Original Interest Rate, Margin for ARM Mortgages (Error)                          | 0.00014                      |
| FICO Score, Original Interest Rate - National Mortgate Rate                       | 0.00014                      |
| FICO Score, Margin for ARM Mortgages (Error)                                      | 0.00014                      |
| ...                                                                               | ...                          |
