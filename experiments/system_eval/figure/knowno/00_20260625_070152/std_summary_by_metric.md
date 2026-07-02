# Standard Deviation Summary

Source: `experiments/system_eval/figure/knowno/00_20260625_070152/figure_tables.csv`

Blank cells mean the standard deviation is undefined or unavailable in the source CSV, typically because `n <= 1` or `n = 0`.

## Tomato

| Metric ID | Metric Label | GPT-3.5 95% | GPT-4o 95% | Ours |
|---|---|---:|---:|---:|
| success_rate | Success Rate | 0.409360 | 0.502418 | 0.362813 |
| average_step | Average Step | 13.609696 | 3.875278 | 6.853233 |
| average_step_success_only | Average Step | 10.735285 | 1.019787 | 5.295295 |
| average_step_failure_only | Average Step | 15.961113 | 3.606965 | 11.844966 |
| average_question | Average Query Number | 13.609696 | 1.751507 | 15.137871 |
| average_question_success_only | Average Query Number | 10.735285 | 1.465302 | 14.389563 |
| average_question_failure_only | Average Query Number | 15.961113 | 1.465686 | 16.873725 |
| query_probability_per_step | Query Probability per Step |  |  |  |
| elapsed_time | Elapsed Time | 27.364348 | 7.543338 | 293.339283 |
| prediction_set_size_when_asked | Prediction Set Size When Asked | 0.125094 | 0.356334 |  |
| token_overall | Token Usage | 56490.973256 | 11562.435125 |  |

## Waste Sorting

| Metric ID | Metric Label | GPT-3.5 95% | GPT-4o 95% | Ours |
|---|---|---:|---:|---:|
| success_rate | Success Rate | 0.337998 | 0.500908 | 0.279582 |
| average_step | Average Step | 7.237187 | 2.347984 | 2.852117 |
| average_step_success_only | Average Step | 1.211524 | 0.494908 | 2.403922 |
| average_step_failure_only | Average Step | 15.761036 | 2.287542 | 2.124784 |
| average_question | Average Query Number | 7.237187 | 2.558330 | 3.359944 |
| average_question_success_only | Average Query Number | 1.211524 | 1.495509 | 3.347738 |
| average_question_failure_only | Average Query Number | 15.761036 | 1.572253 | 3.101233 |
| query_probability_per_step | Query Probability per Step |  |  |  |
| elapsed_time | Elapsed Time | 14.216794 | 5.666583 | 1.371192 |
| prediction_set_size_when_asked | Prediction Set Size When Asked | 0.212809 | 0.299972 |  |
| token_overall | Token Usage | 18007.965574 | 3059.019870 |  |
