- The first table below reports covariate-trio (float) ranking by average absolute mixed gradient (estimated by finite difference) for transition current -> paid off. 
- The second table below reports covariate-trio (float) ranking by average absolute mixed gradient (estimated by finite difference) for transition current -> 30 days delinquent. 

------
### current -> paid off

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

------
### current -> 30 days delinquent

| Feature Trio                                                                                                            | Ave. Absolute Mixed Gradient |
|:-----------------------------------------------------------------------------------------------------------------------:|:------------------------------:|
| FICO Score, Original Interest Rate, Number of Times Current in Last 12 Months                                           | 0.000210                     |
| FICO Score, Original Interest Rate, Original Interest Rate - National Mortgate Rate                                     | 0.000182                     |
| FICO Score, Current Outstanding Balance, Original Interest Rate                                                         | 0.000160                     |
| FICO Score, Original Loan Balance, Original Interest Rate                                                               | 0.000157                     |
| FICO Score, Current Outstanding Balance, Original Loan Balance                                                          | 0.000130                     |
| Current Outstanding Balance, Original Loan Balance, Original Interest Rate                                              | 0.000127                     |
| Original Interest Rate, Original Interest Rate - National Mortgate Rate, Number of Times Current in Last 12 Months      | 0.000120                     |
| Current Outstanding Balance, Original Interest Rate, Number of Times Current in Last 12 Months                          | 0.000100                     |
| FICO Score, Current Outstanding Balance, Number of Times Current in Last 12 Months                                      | 0.000095                     |
| Original Loan Balance, Original Interest Rate, Number of Times Current in Last 12 Months                                | 0.000094                     |
| FICO Score, Original Loan Balance, Number of Times Current in Last 12 Months                                            | 0.000087                     |
| Current Outstanding Balance, Original Loan Balance, Number of Times Current in Last 12 Months                           | 0.000078                     |
| FICO Score, Original Interest Rate - National Mortgate Rate, Number of Times Current in Last 12 Months                  | 0.000076                     |
| FICO Score, Original Interest Rate, Zillow Zip Code Housing Price Change Since Origination                              | 0.000075                     |
| Current Outstanding Balance, Original Interest Rate, Original Interest Rate - National Mortgate Rate                    | 0.000074                     |
| Original Interest Rate, Original Interest Rate - National Mortgate Rate, Current Interest Rate - National Mortgage Rate | 0.000074                     |
| Original Loan Balance, Original Interest Rate, Original Interest Rate - National Mortgate Rate                          | 0.000073                     |
| FICO Score, Original Interest Rate, Lagged Prime Prepayment Rate in Same Zip Code                                       | 0.000071                     |
| FICO Score, Original Interest Rate, Current Interest Rate - National Mortgage Rate                                      | 0.000070                     |
| FICO Score, Original Interest Rate, Time Since Origination                                                              | 0.000068                     |
| Number of Times 30 Days Delinquent in Last 12 Months, FICO Score, Original Interest Rate                                | 0.000066                     |
| FICO Score, Original Loan Balance, Original Interest Rate - National Mortgate Rate                                      | 0.000057                     |
| FICO Score, Current Outstanding Balance, Original Interest Rate - National Mortgate Rate                                | 0.000056                     |
| Current Outstanding Balance, Original Interest Rate, Zillow Zip Code Housing Price Change Since Origination             | 0.000055                     |
| Current Outstanding Balance, Original Loan Balance, Original Interest Rate - National Mortgate Rate                     | 0.000053                     |
| Original Interest Rate, Number of Times Current in Last 12 Months, Lagged Prime Prepayment Rate in Same Zip Code        | 0.000053                     |
| Original Interest Rate, Number of Times Current in Last 12 Months, Current Interest Rate - National Mortgage Rate       | 0.000052                     |
| Current Outstanding Balance, Original Loan Balance, Zillow Zip Code Housing Price Change Since Origination              | 0.000052                     |
| Original Interest Rate, Original Interest Rate - National Mortgate Rate, Lagged Prime Prepayment Rate in Same Zip Code  | 0.000050                     |
| Original Interest Rate, Number of Times Current in Last 12 Months, Time Since Origination                               | 0.000050                     |
| ...                                                                                                                     | ...                          |
