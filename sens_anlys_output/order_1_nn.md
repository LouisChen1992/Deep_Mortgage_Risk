- The first table below reports covariate (float) ranking by average absolute gradient for transition current -> paid off.
- The second table below reports covariate (float) ranking by average absolute gradient for transition current -> 30 days delinquent.

------
### current -> paid off

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

------
### current -> 30 days delinquent
