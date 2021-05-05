# DSC381
# DSC-381 Simulation Work

This contains compiled simulation functions in python for statistical analysis related to DSC381.

## Prerequisites:

- Python 3
- The following packages:
    - numpy
    - pandas
    - matplotlib
    - scipy
    - statsmodels

## Available functions
Currently, there is one python script (statistics_functions_py) capable of doing the following tests:

- Hypothesis test for a single mean

```
    hyptest_singlemean(data, null_hyp = 0, n = 10000, alt = 'two-sided')
        
        Returns simulation p-value (float) for hypothesis test of a single mean.

        Parameter(s):
        data (array) is the dataset for analysis
        null_hyp (float) is the null hypothesis
        n (int) is the number of bootstrap simulations
        alt (string) defines alt hypothesis ['two-sided, 'less', 'greater']
```


- Hypothesis test for a single in proportion

```
    hyptest_singleprop(p_sample, p_null = 0.5, size = 30, n = 10000, alt = 'two-sided', disp = 'count')

        Returns simulation p-value (float) for hypothesis test of a single proportion.

        Parameter(s):
        p_sample (float) is the sample proportion
        p_null (float) is the null hypothesis proportion
        size (int) is the sample size
        n (int) is the number of bootstrap simulations
        alt (string) defines alt hypothesis ['two-sided, 'less', 'greater']
        disp (string) choose plot display ['count', 'prop']
```

- Hypothesis test for difference in two means

```
    hyptest_diffmeans(data_1, data_2, n = 10000, alt = 'two-sided')
        
        Returns simulation p-value (float) for hypothesis test of a difference in two means
        Null hypothesis is defined as the two means being equal

        Parameter(s):
        data_1 (array) is the dataset for first mean
        data_2 (array) is the dataset for second mean
        n (int) is the number of bootstrap simulations
        alt (string) defines alt hypothesis ['two-sided, 'less', 'greater']
```

- Hypothesis test for difference in two proportions

```
    hyptest_diffprops(p_1, size_1, p_2, size_2, n = 10000, alt = 'two-sided')
        
        Returns simulation p-value (float) for hypothesis test of a difference in two proportions
        Null hypothesis is defined as the two proportions being equal

        Parameter(s):
        p_1 (float) is proportion for dataset 1
        size_1 (int) is the size of dataset 1
        p_2 (float) is the proportion for dataset 2
        size_2 (int) is the size of dataset 2
        n (int) is the number of bootstrap simulations
        alt (string) defines alt hypothesis ['two-sided, 'less', 'greater']
```
