# DSC381
# DSC-381 Simulation Work

This contains compiled simulation functions in python for statistical analysis related to DSC381.

# Prerequisites:

- Python 3
- The following packages:
    - numpy
    - pandas
    - matplotlib
    - scipy
    - statsmodels



# Functionality

Currently, there is one python script (statistics_functions_py) capable of doing the following tests:

## Hypothesis Testing  

  

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

- Hypothesis test for a slope

```
    hyptest_slope(features, targets, n=10000, alt='two-sided')
        
        Returns simulation p-value (float) for hypothesis test of a slope (correlation) of two datasets
        Null hypothesis is that there is no correlation between the two datasets

        Parameter(s):
        features (array) is a single dimension array of features
        targets (array) is a single dimension array of targets
        n (int) is the number of bootstrap simulations
        alt (string) defines alt hypothesis ['two-sided, 'less', 'greater']
```

## Confidence Intervals

- Confidence Interval for a statistic

```
    ci_statistic(data, n=10000, l_tail=5, r_tail=95, stat='mean')

        Returns returns confidence interval (tuple) for a statistic (left tail, right tail, length)

        Parameter(s):
        data (array) is the dataset for analysis
        n (int) is the number of bootstrap simulations
        l_tail (float) is the cut off percentile for the left tail [0,100]
        r_tail (float) is the cut off percentile for the right tail [0,100]
        stat (string) is the statistic being analyzed ["mean","median","stdev"]
```

- Confidence Interval for a proportion

```
    ci_prop(p, size, n=10000, l_tail=5, r_tail=95)

        Returns returns confidence interval (tuple) a proportion (left tail, right tail, length)

        Parameter(s):
        p (float) is proportion for the dataset
        size (int) is the size of the dataset
        n (int) is the number of bootstrap simulations
        l_tail (float) is the cut off percentile for the left tail [0,100]
        r_tail (float) is the cut off percentile for the right tail [0,100]
```


        
- Confidence Interval for difference of two proportions

```
    ci_diffprops(p_1, size_1, p_2, size_2, n=10000, l_tail=5, r_tail=95)

        Returns returns confidence interval (tuple) for difference in proportions (left tail, right tail, length)

        Parameter(s):
        p_1 (float) is proportion for dataset 1
        size_1 (int) is the size of dataset 1
        p_2 (float) is the proportion for dataset 2
        size_2 (int) is the size of dataset 2
        n (int) is the number of bootstrap simulations
        l_tail (float) is the cut off percentile for the left tail [0,100]
        r_tail (float) is the cut off percentile for the right tail [0,100]
```


- Confidence Interval for difference of two proportions

```
    ci_diffstatistics(data_1, data_2, n=10000, l_tail=5, r_tail=95, stat='mean')

        Returns the confidence interval for the difference in two statistics

        Parameter(s):
        data_1 (array) is the dataset for first mean
        data_2 (array) is the dataset for second mean
        n (int) is the number of bootstrap simulations
        l_tail (float) is the cut off percentile for the left tail [0,100]
        r_tail (float) is the cut off percentile for the right tail [0,100]
        stat (string) is the statistic being analyzed ["mean","median","stdev"]
```


- Confidence Interval for a slope

```
    ci_slope(features, targets, n=10000, l_tail=5, r_tail=95)
        """
        Returns simulation p-value (float) for hypothesis test of a slope (correlation) of two datasets
        Null hypothesis is that there is no correlation between the two datasets

        Parameter(s):
        features (array) is a single dimension array of features
        targets (array) is a single dimension array of targets
        n (int) is the number of bootstrap simulations
        alt (string) defines alt hypothesis ['two-sided, 'less', 'greater']
```