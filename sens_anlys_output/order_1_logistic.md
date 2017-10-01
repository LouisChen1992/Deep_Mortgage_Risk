- The first table below reports covariate (float) ranking by average absolute gradient for transition current -> paid off.
- The second table below reports covariate (float) ranking by average absolute gradient for transition current -> 30 days delinquent.

------
### current -> paid off

| Feature                                                                       | Ave. Absolute Gradient |
|:-----------------------------------------------------------------------------:|:----------------------:|
| Original Interest Rate                                                        | 0.1596                 |
| Lagged Prime Default Rate in Same Zip Code                                    | 0.1505                 |
| Original Interest Rate - National Mortgate Rate                               | 0.1203                 |
| Number of Days Delinquent                                                     | 0.0773                 |
| Lagged Prime Prepayment Rate in Same Zip Code                                 | 0.0697                 |
| ARM Contract Details (X_static[35])                                           | 0.0695                 |
| Current Interest Rate - National Mortgage Rate                                | 0.0674                 |
| ARM Contract Details (X_static[33])                                           | 0.0552                 |
| Current Interest Rate - Original Interest Rate                                | 0.0533                 |
| ARM Contract Details (X_static[34])                                           | 0.0491                 |
| Zillow Housing Prices (macro_data[21] != 0)                                   | 0.0441                 |
| Scheduled Interest and Principle Due (Error)                                  | 0.0421                 |
| Original Loan Balance                                                         | 0.0388                 |
| Current Outstanding Balance                                                   | 0.0356                 |
| Time Since Origination                                                        | 0.0341                 |
| Zillow Housing Prices (macro_data[20] != 0)                                   | 0.0317                 |
| ARM First Rate Reset Period                                                   | 0.0298                 |
| Zillow Zip Code Housing Price Change Since Origination                        | 0.0266                 |
| Number of Times 90+ Days Delinquent in Last 12 Months                         | 0.0238                 |
| Number of Times Current in Last 12 Months                                     | 0.0227                 |
| Lagged Default Rate for Subprime Mortgages in Same Zip Code (>= 500 & < 1000) | 0.0214                 |
| ARM Contract Details (X_static[37])                                           | 0.0203                 |
| Num_IO_mon (error)                                                            | 0.0201                 |
| Margin for ARM Mortgages (Error)                                              | 0.0198                 |
| FICO Score                                                                    | 0.0165                 |
| Total Number of Prime Mortgages in Same Zip Code                              | 0.0163                 |
| Original Term of the Loan (Error)                                             | 0.0159                 |
| Lagged Default Rate for Subprime Mortgages in Same Zip Code (>= 250 & < 500)  | 0.0158                 |
| Lagged Default Rate for Subprime Mortgages in Same Zip Code (>= 1000)         | 0.0150                 |
| ARM Contract Details (X_static[32])                                           | 0.0141                 |
| ...                                                                           | ...                    |

------
### current -> 30 days delinquent

| Feature                                                                       | Ave. Absolute Gradient |
|:-----------------------------------------------------------------------------:|:----------------------:|
| FICO Score                                                                    | 0.0753                 |
| Zillow Zip Code Housing Price Change Since Origination                        | 0.0482                 |
| Number of Times 30 Days Delinquent in Last 12 Months                          | 0.0424                 |
| Current Outstanding Balance                                                   | 0.0296                 |
| Original Loan Balance                                                         | 0.0287                 |
| Original Interest Rate                                                        | 0.0195                 |
| Lagged Default Rate for Subprime Mortgages in Same Zip Code (>= 1000)         | 0.0168                 |
| Number of Times 60 Days Delinquent in Last 12 Months                          | 0.0168                 |
| ARM Contract Details (X_static[33])                                           | 0.0157                 |
| Number of Times Foreclosed in Last 12 Months                                  | 0.0150                 |
| Number of Times 90+ Days Delinquent in Last 12 Months                         | 0.0145                 |
| Zillow Housing Prices (macro_data[21] != 0)                                   | 0.0145                 |
| Lagged Default Rate for Subprime Mortgages in Same Zip Code (>= 500 & < 1000) | 0.0136                 |
| ARM Rate Reset Frequency                                                      | 0.0131                 |
| Lagged Prime Prepayment Rate in Same Zip Code                                 | 0.0112                 |
| ARM Contract Details (X_static[32])                                           | 0.0110                 |
| Lagged Default Rate for Subprime Mortgages in Same Zip Code (>= 250 & < 500)  | 0.0109                 |
| Original Interest Rate - National Mortgate Rate                               | 0.0105                 |
| Original Term of the Loan (Error)                                             | 0.0095                 |
| LTV Ratio                                                                     | 0.0095                 |
| ARM Contract Details (X_static[35])                                           | 0.0092                 |
| Number of Days Delinquent                                                     | 0.0092                 |
| Lagged Default Rate for Subprime Mortgages in Same Zip Code (>= 100 & < 250)  | 0.0087                 |
| Original Interest Rate - National Mortgage Rate at Origination                | 0.0076                 |
| Total Number of Prime Mortgages in Same Zip Code                              | 0.0075                 |
| Lagged Default Rate for Subprime Mortgages in Same Zip Code (< 100)           | 0.0065                 |
| Original Term of the Loan                                                     | 0.0060                 |
| ARM Contract Details (X_static[34])                                           | 0.0055                 |
| State Unemployment Rate                                                       | 0.0042                 |
| Zillow Housing Prices (macro_data[20] != 0)                                   | 0.0037                 |
| ...                                                                           | ...                    |
