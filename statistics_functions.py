# import relevant packages

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest
from sklearn import linear_model

if __name__ == "__main__":

    print("This is a set of statistics and probability functions for DSC 381")

else:
    '''
    0.0 Section is a set of helper functions for bootstrap simulations
    '''


    # probability value tests
    def simulation_pval(b_array, null_hyp, sample, alt):
        """
        Takes in a evaluated bootstrap array and calculates against the sample value

        Parameter(s):
        b_array (array) an array of simulated bootstrap statistics
        null_hyp (float) is the null hypothesis statistic
        sample (float) is the sample statistic
        alt (string) is the type of test ['less', 'greater', 'two-sided']
        """

        if alt == 'less':
            p_val = np.mean(b_array <= sample)
        elif alt == 'greater':
            p_val = np.mean(b_array >= sample)
        else:
            if null_hyp > sample:
                p_val = 2 * np.mean(b_array <= sample)
            else:
                p_val = 2 * np.mean(b_array >= sample)
        return p_val


    def ztest_numpy(z_stat, alt):
        """
        takes a z_statistic and returns a z_val

        Parameter(s):
        z_stat (float) is the calculated z statistic
        alt (string) is the type of test ['less', 'greater', 'two-sided']
        """
        if alt == 'less':
            z_val = stats.norm.cdf(z_stat)
        elif alt == 'greater':
            z_val = stats.norm.sf(z_stat)
        else:
            if z_stat <= 0:
                z_val = 2 * stats.norm.cdf(z_stat)
            else:
                z_val = 2 * stats.norm.sf(z_stat)

        print(f'z-stat, z-val using numpy           :{z_stat:.4f}, {z_val:.4f}')

        return z_val


    def ttest_numpy(t_stat, dof, alt):
        """
        takes a t_statistic and returns a z_val

        Parameter(s):
        t_stat (float) is the calculated z statistic
        dof (int) is the degrees of freedom
        alt (string) is the type of test ['less', 'greater', 'two-sided']
        """
        if alt == 'less':
            t_val = stats.t.cdf(t_stat, dof)
        elif alt == 'greater':
            t_val = stats.t.sf(t_stat, dof)
        else:
            if t_stat <= 0:
                t_val = 2 * stats.t.cdf(t_stat, dof)
            else:
                t_val = 2 * stats.t.sf(t_stat, dof)

        print(f't-stat, t-val using numpy:          :{t_stat:.4f}, {t_val:.4f}')

        return t_val


    def sm_alt_convert(alt):
        """
        Converts scipy alt terminology to statsmodels.api terminology

        Parameter(s):
        alt (string) ['less', 'greater', 'two-sided'] converts to ['smaller', 'larger', 'two-sided']
        """
        if alt == 'less':
            sm_alt = 'smaller'
        elif alt == 'greater':
            sm_alt = 'larger'
        else:
            sm_alt = 'two-sided'
        return sm_alt


    def hyptest_histplot(b_array, sim_hyp, test_hyp, size, alt, x_label, hist_title):
        """
        Generates a histogram plot for hypothesis test

        Parameter(s):
        b_array (array) bootstrap simulation results
        sim_hyp (float) the assumed statistic for simulation (center of simulation)
        test_hyp (float) the tested statistic for simulation
        alt (string) the type of test ['less', 'greater', 'two-sided']
        x_label (string) x label title
        hist_title (string) histogram title

        """

        plt.figure(figsize=(8, 4))
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel("bootstrap samples", fontsize=12)
        plt.title(hist_title)

        plt.hist(b_array, bins=size)
        plt.axvline(x=test_hyp, c='r')

        diff = sim_hyp - test_hyp
        bar_1 = sim_hyp + diff
        bar_2 = sim_hyp - diff

        if alt == 'two-sided':
            plt.axvline(x=bar_1, c='r')
            plt.axvline(x=bar_2, c='r')

        plt.show()


    '''
    0.1 Section is a set of helper functions for bootstrap simulations for confidence intervals
    '''


    def simulation_ci(b_array, l_tail, r_tail):
        """
        Takes the confidence interval of a bootstrap simulation

        Parameter(s):
        b_array (array) is a bootstrap simulation of a statistic
        l_tail (float) is the cut off percentile for the left tail [0,100]
        r_tail (float) is the cut off for the right tail [0,100]
        """
        left = np.percentile(b_array, l_tail)
        right = np.percentile(b_array, r_tail)
        length = right - left
        return left, right, length


    def ci_histplot(b_array, ci, size, x_label, hist_title):

        """
        Generates a histogram plot for hypothesis test

        Parameter(s):
        b_array (array) bootstrap simulation results
        sim_hyp (float) the assumed statistic for simulation (center of simulation)
        test_hyp (float) the tested statistic for simulation
        alt (string) the type of test ['less', 'greater', 'two-sided']
        x_label (string) x label title
        hist_title (string) histogram title

        """
        plt.figure(figsize=(8, 4))
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel("bootstrap samples", fontsize=12)
        plt.title(hist_title, fontsize=12)

        plt.hist(b_array, bins=size)
        plt.axvline(x=ci[0], c='r')
        plt.axvline(x=ci[1], c='r')
        plt.show()


    def ci_ttest(l_tail, r_tail, dof, sample, se):
        """
        Returns confidence interval using t distribution
        
        Parameter(s):
        l_tail (float) is the cut off percentile for the left tail [0,100]
        r_tail (float) is the cut off for the right tail [0,100]
        dof (int) degrees of freedom
        sample (float) is the statistic
        se (float) is the standard error
        
        """
        r_tstat = stats.t.ppf(r_tail / 100, dof)
        l_tstat = stats.t.ppf(l_tail / 100, dof)

        right_tval = sample + se * r_tstat
        left_tval = sample + se * l_tstat
        length_tval = right_tval - left_tval

        return left_tval, right_tval, length_tval, l_tstat, r_tstat


    def ci_ztest(l_tail, r_tail, sample, se):
        """
        Returns confidence interval using z distribution
        
        Parameter(s):
        l_tail (float) is the cut off percentile for the left tail [0,100]
        r_tail (float) is the cut off for the right tail [0,100]
        sample (float) is the sample statistic or proportion
        se (float) is the standard error
        
        """
        r_zstat = stats.norm.ppf(r_tail / 100)
        l_zstat = stats.norm.ppf(l_tail / 100)

        right_zval = sample + se * r_zstat
        left_zval = sample + se * l_zstat
        length_zval = right_zval - left_zval

        return left_zval, right_zval, length_zval, l_zstat, r_zstat


    '''
    1.0 Hypothesis testing functions
    '''


    # Randomization Hypothesis Testing Functions

    def hyptest_singlemean(data, null_hyp=0, n=10000, alt='two-sided'):
        """
        Returns simulation p-value (float) for hypothesis test of a single mean.

        Parameter(s):
        data (array) is the dataset for analysis
        null_hyp (float) is the null hypothesis
        n (int) is the number of bootstrap simulations
        alt (string) defines alt hypothesis ['two-sided, 'less', 'greater']
        """
        start = time.time()
        size = len(data)

        # Randomization method for null case
        mu_sample = np.mean(data)
        null_shift = null_hyp - mu_sample
        data_shift = data + null_shift

        # Bootstrap sample
        rng = np.random.default_rng()
        b_array = np.empty(0)

        for _ in range(n):
            b_data = rng.choice(data_shift, size, replace=True)
            b_mean = np.mean(b_data)
            b_array = np.append(b_array, b_mean)

        # Tested condition for p_value
        p_val = simulation_pval(b_array, null_hyp, mu_sample, alt)
        end = time.time()

        # Verify with theoretical distribution

        print(f'\nUsing Theoretical Distn:')

        if size < 30:
            print('Warning! Sample size is less than 30 (size: {size}\n')

        dof = size - 1
        s = data.var(ddof=1)
        se = np.sqrt(s / size)

        t_stat = (mu_sample - null_hyp) / se

        # calculate the t_stat
        t_val = ttest_numpy(t_stat, dof, alt)

        # validate with scipy
        check_t_stat, check_t_val = stats.ttest_1samp(data, null_hyp, alternative=alt)

        print(f't-stat, t-val from scipy stats      :{check_t_stat:.4f}, {check_t_val:.4f}')
        print(f'dof: {dof}\n')

        # print results and plot histogram
        print(f'Using Simulation:')
        print(f'time: {end - start:.3f}s')
        print(f'mu of sample: {mu_sample:.3f}')
        print(f'\nCalculated p_value: {p_val}\n')

        # plot histogram
        x_label = "mu"
        hist_title = "Hypothesis test of a single mean"
        hyptest_histplot(b_array, null_hyp, mu_sample, size, alt, x_label, hist_title)

        return p_val


    def hyptest_singleprop(p_sample, p_null=0.5, size=30, n=10000, alt='two-sided', disp='count'):
        """
        Returns simulation p-value (float) for hypothesis test of a single proportion.

        Parameter(s):
        p_sample (float) is the sample proportion
        p_null (float) is the null hypothesis proportion
        size (int) is the sample size
        n (int) is the number of bootstrap simulations
        alt (string) defines alt hypothesis ['two-sided, 'less', 'greater']
        disp (string) choose plot display ['count', 'prop']
        """
        start = time.time()

        count_sample = p_sample * size
        null_hyp = p_null * size

        p_diff = np.abs(p_sample - p_null)
        count_diff = np.abs(count_sample - null_hyp)

        # Bootstrap sample - rounding causes issues so processing with counts first
        rng = np.random.default_rng()
        b_array = np.empty(0)

        for _ in range(n):
            b_data = rng.choice([0, 1], size, replace=True, p=[1 - p_null, p_null])
            b_mean = np.sum(b_data)
            b_array = np.append(b_array, b_mean)

        # Tested condition for p_value
        p_val = simulation_pval(b_array, null_hyp, count_sample, alt)

        end = time.time()

        print(f'\nUsing Theoretical Distn:')

        # Verify with theoretical distribution
        se = np.sqrt(p_null * (1 - p_null) / size)
        z_stat = (p_sample - p_null) / se

        # normal distribution check
        z_val = ztest_numpy(z_stat, alt)

        # validate with statsmodels
        sm_alt = sm_alt_convert(alt)

        check_z_stat, check_z_val = proportions_ztest(count_sample, size, value=p_null, alternative=sm_alt,
                                                      prop_var=p_null)
        print(f'z-stat, z-val from statsmodels      :{check_z_stat:.4f}, {check_z_val:.4f}')

        if min(p_null * size, (1 - p_null) * size) < 10:
            print('\nWarning! np >= 10 or (1-p)*n >= 10 requirement not met')
            print(f'p_null*size: {p_null * size:.1f} (1-p_null)*size): {(1 - p_null) * size:.1f}\n')

        # print results
        print(f'\nUsing Simulation:')
        print(f'time: {end - start:.3f}s')
        print(f'count of sample set: {count_sample:.1f}')
        print(f'size of sample set: {size}')
        print(f'p of sample set: {p_sample:.3f}')
        print(f'\nCalculated p_value: {p_val}\n')

        if disp == 'prop':
            b_array /= size
            null_hyp /= size
            count_sample /= size

        # plot histogram
        x_label = disp
        hist_title = "Hypothesis test of a single proportion"

        binwidth = (np.max(b_array) - np.min(b_array))/100
        bins=np.arange(min(b_array), max(b_array) + binwidth, binwidth)

        hyptest_histplot(b_array, null_hyp, count_sample, bins, alt, x_label, hist_title)

        return p_val


    def hyptest_diffmeans(data_1, data_2, n=10000, alt='two-sided'):
        """
        Returns simulation p-value (float) for hypothesis test of a difference in two means
        Null hypothesis is defined as the two means being equal

        Parameter(s):
        data_1 (array) is the dataset for first mean
        data_2 (array) is the dataset for second mean
        n (int) is the number of bootstrap simulations
        alt (string) defines alt hypothesis ['two-sided, 'less', 'greater']
        """
        start = time.time()

        # values for group 1
        size_1 = data_1.shape[0]
        mu_1 = np.mean(data_1)

        # values for group 2
        size_2 = data_2.shape[0]
        mu_2 = np.mean(data_2)

        mu_diff = mu_1 - mu_2

        null_hyp = 0

        data = np.concatenate([data_1, data_2])
        size = data.shape[0]

        # Bootstrap samples per reallocation method
        rng = np.random.default_rng()
        b_array = np.empty(0)

        for _ in range(n):
            b_data = rng.choice(data, size, replace=False)
            b_1 = b_data[0:size_1]
            b_2 = b_data[size_1:size]

            b_diff = np.mean(b_1) - np.mean(b_2)
            b_array = np.append(b_array, b_diff)

        # Tested condition for p_value
        p_val = simulation_pval(b_array, 0, mu_diff, alt)

        end = time.time()

        # calculate with theoretical distribution
        print(f'Using Theoretical Distn:')

        dof = min(size_1, size_2) - 1

        s_1 = np.sqrt(data_1.var(ddof=1))
        s_2 = np.sqrt(data_2.var(ddof=1))

        se = np.sqrt(s_1 ** 2 / size_1 + s_2 ** 2 / size_2)
        t_stat = ((mu_1 - mu_2) - 0) / se

        # calculate the t_stat
        t_val = ttest_numpy(t_stat, dof, alt)

        # validate with scipy stats
        check_t_stat, check_t_val = stats.ttest_ind(data_1, data_2, equal_var=True, alternative=alt)

        print(f't-stat, t-val from stat module      :{check_t_stat:.4f}, {check_t_val:.4f}')
        print(f'dof: {dof}\n')

        if size_1 < 30:
            print(f'Warning! Sample one (size: {size_1}) is less than 30 \n')

        if size_2 < 30:
            print(f'Warning! Sample two (size: {size_2}) is less than 30 \n')

        print(f'Using Simulation:')

        # print results
        print(f'time: {end - start:.3f}s')
        print(f'sample mean diff: {mu_diff:.3f}')
        print(f'size of sample set (1 and 2): {size_1} and {size_2}')
        print(f'\nCalculated p_value: {p_val}\n')

        x_label = 'mu'
        hist_title = 'Difference in two means'
        hyptest_histplot(b_array, null_hyp, mu_diff, size, alt, x_label, hist_title)

        return p_val


    def hyptest_diffprops(p_1, size_1, p_2, size_2, n=10000, alt='two-sided'):
        """
        Returns simulation p-value (float) for hypothesis test of a difference in two proportions
        Null hypothesis is defined as the two proportions being equal

        Parameter(s):
        p_1 (float) is proportion for dataset 1
        size_1 (int) is the size of dataset 1
        p_2 (float) is the proportion for dataset 2
        size_2 (int) is the size of dataset 2
        n (int) is the number of bootstrap simulations
        alt (string) defines alt hypothesis ['two-sided, 'less', 'greater']
        """

        start = time.time()

        # values for group 1
        count_1 = np.round(p_1 * size_1)

        # values for group 2
        count_2 = np.round(p_2 * size_2)

        # calculate differences
        count_diff = count_1 - count_2
        p_diff = p_1 - p_2

        # define null hypothesis of p_1 = p_2
        size = size_1 + size_2
        count_total = count_1 + count_2
        p_null = count_total / size
        p_null_diff = 0
        null_diff = p_null * size_1 - p_null * size_2

        # Bootstrap samples per reallocation method
        rng = np.random.default_rng()
        b_array = np.empty(0)
        b_p_array = np.empty(0)
        null_hyp_array = np.concatenate([np.zeros(size - int(count_total)), np.ones(int(count_total))])

        for _ in range(n):
            b_data = rng.choice(null_hyp_array, size, replace=False)
            b_1 = b_data[0:size_1]
            b_2 = b_data[size_1:size]

            b_diff = sum(b_1) - sum(b_2)
            b_array = np.append(b_array, b_diff)

            b_p_diff = sum(b_1) / size_1 - sum(b_2) / size_2
            b_p_array = np.append(b_p_array, b_p_diff)

        # Tested condition for p_value
        p_val = simulation_pval(b_array, null_diff, count_diff, alt)

        # convert graph from count to proportions
        b_array = b_p_array
        end = time.time()

        # calculate with theoretical distribution
        print(f'Using Theoretical Distn:')

        dof = min(size_1, size_2) - 1
        se = np.sqrt(p_null * (1 - p_null) / size_1 + p_null * (1 - p_null) / size_2)
        z_stat = ((p_1 - p_2) - 0) / se

        # normal distribution check
        z_val = ztest_numpy(z_stat, alt)

        # validate with statsmodels
        sm_alt = sm_alt_convert(alt)

        # validate with statsmodels
        check_z_stat, check_z_val = proportions_ztest([count_1, count_2], [size_1, size_2], value=0, alternative=sm_alt)
        print(f'z-stat, z-val from statsmodels      :{check_z_stat:.4f}, {check_z_val:.4f}')
        print(f'dof: {dof}\n')

        if min(p_1 * size_1, (1 - p_1) * size_1) < 10:
            print('Warning! group 1 np >= 10 or (1-p)*n >= 10 requirement not met')
            print(f'p_1*size_1: {p_1 * size_1} (1-p_1)*size_1: {(1 - p_1) * size_1}\n')

        if min(p_2 * size_2, (1 - p_2) * size_2) < 10:
            print('Warning! group 2 np >= 10 or (1-p)*n >= 10 requirement not met')
            print(f'p_2*size_2: {p_2 * size_2} (1-p_2)*size_2: {(1 - p_2) * size_2}\n')

        print(f'Using Simulation:')

        print(f'time: {end - start:.3f}s')

        print(f'dataset 1 count: {count_1}')
        print(f'dataset 2 count: {count_2}')
        print(f'sample count diff: {count_diff}')
        print(f'sample p diff: {p_diff:.3f}')
        print(f'\nCalculated p_value: {p_val}\n')

        x_label = 'proportion difference'
        hist_title = 'Hypothesis test of two proportions'

        hyptest_histplot(b_array, p_null_diff, p_diff, size, alt, x_label, hist_title)

        return p_val


    def hyptest_slope(features, targets, n=10000, alt='two-sided'):
        """
        Returns simulation p-value (float) for hypothesis test of a slope (correlation) of two datasets
        Null hypothesis is that there is no correlation between the two datasets

        Parameter(s):
        features (array) is a single dimension array of features
        targets (array) is a single dimension array of targets
        n (int) is the number of bootstrap simulations
        alt (string) defines alt hypothesis ['two-sided, 'less', 'greater']
        """

        start = time.time()

        # perform regression to find slope of the sample

        regr = linear_model.LinearRegression().fit(features, targets)
        sample_slope = float(regr.coef_)

        # Bootstrap the data by randomly pairing target with features 
        rng = np.random.default_rng()
        size = features.shape[0]
        b_array = np.empty(0)
        null_hyp = 0

        for _ in range(n):
            b_features = rng.choice(features, size, replace=False)
            b_targets = rng.choice(targets, size, replace=False)

            regr = linear_model.LinearRegression().fit(b_features, b_targets)
            b_slope = regr.coef_
            b_array = np.append(b_array, b_slope)

        # Tested condition for p_value

        p_val = simulation_pval(b_array, null_hyp, sample_slope, alt)

        end = time.time()

        # Verify with theoretical distribution and check w/ statsmodels OLS

        lm_features = np.column_stack((features, np.ones(size)))
        lm = sm.OLS(targets, lm_features).fit()
        se = lm.bse[0]

        check_t_stat = float(lm.tvalues[0])
        check_t_val = float(lm.pvalues[0])

        print(f'\nUsing Theoretical Distn:')
        dof = size - 2
        t_stat = float(sample_slope / se)

        # calculate the t_stat
        t_val = ttest_numpy(t_stat, dof, alt)

        print(f't-stat, t-val statsmodel (2-sided)  :{check_t_stat:.4f}, {check_t_val:.4f}')
        print(f'dof: {dof}\n')

        # print results and plot histogram
        print(f'Using Simulation:')
        print(f'time: {end - start:.3f}s')
        print(f'slope of sample: {sample_slope:.3f}')
        print(f'\nCalculated p_value: {p_val}\n')

        # plot histogram
        x_label = "slope"
        hist_title = "Hypothesis test of a slope"
        hyptest_histplot(b_array, null_hyp, sample_slope, size, alt, x_label, hist_title)

        return p_val


    '''
    2.0 Confidence interval testing functions
    '''


    def ci_statistic(data, n=10000, l_tail=5, r_tail=95, stat='mean'):
        """
        Returns returns confidence interval (tuple) for a statistic (left tail, right tail, length)

        Parameter(s):
        data (array) is the dataset for analysis
        n (int) is the number of bootstrap simulations
        l_tail (float) is the cut off percentile for the left tail [0,100]
        r_tail (float) is the cut off percentile for the right tail [0,100]
        stat (string) is the statistic being analyzed ["mean","median","stdev"]
        """
        start = time.time()
        size = len(data)

        if stat == 'mean':
            sample_stat = np.mean(data)
        elif stat == 'median':
            sample_stat = np.median(data)
        elif stat == 'stdev':
            sample_stat = np.std(data)
        else:
            raise Exception("invalid input for statistic type")

            # Bootstrap sample
        rng = np.random.default_rng()
        b_array = np.empty(0)

        for _ in range(n):
            b_data = rng.choice(data, size, replace=True)

            if stat == 'mean':
                b_stat = np.mean(b_data)
            elif stat == 'median':
                b_stat = np.median(b_data)
            elif stat == 'stdev':
                b_stat = np.std(b_data)
            else:
                raise Exception("invalid input for statistic type")

            b_array = np.append(b_array, b_stat)

        # Confidence Interval via simulation
        ci = simulation_ci(b_array, l_tail, r_tail)
        end = time.time()

        # Verify with theoretical distribution
        if stat == 'mean':
            print(f'\nUsing Theoretical Distn:')

            if size < 30:
                print('Warning! Sample size is less than 30 \n')

            dof = size - 1
            s = data.var(ddof=1)
            se = np.sqrt(s / size)
            left_tval, right_tval, length_tval, l_tstat, r_tstat = ci_ttest(l_tail, r_tail, dof, sample_stat, se)

            alpha = (r_tail - l_tail) / 100

            left_check, right_check = stats.t.interval(alpha, df=dof, loc=sample_stat, scale=stats.sem(data))
            length_check = right_check - left_check

            print(f't-stat: {r_tstat:.4f} and {l_tstat:.3f}')
            print(f'confidence interval w/ formula  : {left_tval:.3f} to {right_tval:.3f}, length {length_tval:.3f}')
            print(
                f'confidence interval from scipy  : {left_check:.3f} to {right_check:.3f}, length {length_check:.3f}\n')

        else:
            print(f'Theoretical confidence interval not available for {stat}.\n')

        # Print simulation results
        print(f'Using Simulation:')
        print(f'time: {end - start:.3f}s')
        print(f'sample statistic: {sample_stat:.3f}')
        print(f'size of sample set: {size}')
        print(f'Confidence Interval ({l_tail} to {r_tail}): ({ci[0]:.3f} to {ci[1]:.3f}), length {ci[2]:.3f}')

        # Plot histogram
        x_label = stat
        hist_title = f'Confidence Interval of {stat}'
        ci_histplot(b_array, ci, size, x_label, hist_title)

        return ci


    def ci_prop(p, size, n=10000, l_tail=5, r_tail=95):
        """
        Returns returns confidence interval (tuple) a proportion (left tail, right tail, length)

        Parameter(s):
        p (float) is proportion for the dataset
        size (int) is the size of the dataset
        n (int) is the number of bootstrap simulations
        l_tail (float) is the cut off percentile for the left tail [0,100]
        r_tail (float) is the cut off percentile for the right tail [0,100]
        """
        start = time.time()

        # Convert to count for group
        count = np.round(p * size)

        # Bootstrap samples with replacement
        rng = np.random.default_rng()
        b_array = np.empty(0)
        b_p_array = np.empty(0)

        for _ in range(n):
            b_data = rng.choice([0, 1], size, replace=True, p=[1 - p, p])
            b_count = sum(b_data)
            b_array = np.append(b_array, b_count)

            b_p = b_count / size
            b_p_array = np.append(b_p_array, b_p)

        # Confidence Interval via simulation
        b_array = b_p_array

        ci = simulation_ci(b_array, l_tail, r_tail)
        end = time.time()

        # Calculate with theoretical distribution

        print(f'Using Theoretical Distn:')

        se = np.sqrt(p * (1 - p) / size)

        left_zval, right_zval, length_zval, l_zstat, r_zstat = ci_ztest(l_tail, r_tail, p, se)

        print(f'z-stat: left {l_zstat:.3f} and right {r_zstat:.4f}')
        print(f'confidence interval w/ formula  : {left_zval:.3f} to {right_zval:.3f}, length {length_zval:.3f}\n')

        if min(p * size, (1 - p) * size) < 10:
            print('\nWarning! np >= 10 or (1-p)*n >= 10 requirement not met')
            print(f'p*size: {p * size:.1f} (1-p)*size): {(1 - p) * size:.1f}\n')

        # Print simulation results
        print(f'Using Simulation:')
        print(f'Time: {end - start:.3f}s')
        print(f'For proportion {p}, size {size} with count {count}')
        print(f'Confidence Interval ({l_tail} to {r_tail}): ({ci[0]:.3f} to {ci[1]:.3f}), length {ci[2]:.3f}\n')

        # Plot histogram
        x_label = 'proportion'
        hist_title = f'Confidence Interval of a single proportion'
        ci_histplot(b_array, ci, size, x_label, hist_title)
        return ci


    def ci_diffprops(p_1, size_1, p_2, size_2, n=10000, l_tail=5, r_tail=95):
        """
        Returns returns confidence interval (tuple) for difference in proportions (left tail, right tail, length)

        Parameter(s):
        p_1 (float) is proportion for dataset 1
        size_1 (int) is the size of dataset 1
        p_2 (float) is the proportion for dataset 2
        size_2 (int) is the size of dataset 2
        n (int) is the number of bootstrap simulations
        l_tail (float) is the cut off percentile for the left tail [0,100]
        r_tail (float) is the cut off percentile for the right tail [0,100]
        """
        start = time.time()

        # values for group 1
        count_1 = np.round(p_1 * size_1)

        # values for group 2
        count_2 = np.round(p_2 * size_2)

        # calculate sample differences
        count_diff = count_1 - count_2
        p_diff = p_1 - p_2

        # calculate total size and proportion
        size = size_1 + size_2
        count_total = count_1 + count_2
        p_total = count_total / size

        # Bootstrap samples with replacement
        rng = np.random.default_rng()
        b_array = np.empty(0)
        b_p_array = np.empty(0)

        for _ in range(n):
            b_1 = rng.choice([0, 1], size_1, replace=True, p=[1 - p_1, p_1])
            b_2 = rng.choice([0, 1], size_2, replace=True, p=[1 - p_2, p_2])

            # b_1 = rng.choice(data_1, size_1, replace = True)
            # b_2 = rng.choice(data_2, size_2, replace = True)

            b_diff = sum(b_1) - sum(b_2)
            b_array = np.append(b_array, b_diff)

            b_p_diff = sum(b_1) / size_1 - sum(b_2) / size_2
            b_p_array = np.append(b_p_array, b_p_diff)

        # Confidence Interval via simulation
        b_array = b_p_array

        ci = simulation_ci(b_array, l_tail, r_tail)
        end = time.time()

        # Calculate with theoretical distribution

        print(f'Using Theoretical Distn:')

        se = np.sqrt(p_1 * (1 - p_1) / size_1 + p_2 * (1 - p_2) / size_2)

        left_zval, right_zval, length_zval, l_zstat, r_zstat = ci_ztest(l_tail, r_tail, p_diff, se)

        print(f'z-stat: left {l_zstat:.3f} and right {r_zstat:.4f}')
        print(f'confidence interval w/ formula  : {left_zval:.3f} to {right_zval:.3f}, length {length_zval:.3f}\n')

        if min(p_1 * size_1, (1 - p_1) * size_1) < 10:
            print('Warning!')
            print('Group 1 [np >= 10] or [(1-p)*n >= 10] requirement not met')
            print(f'p_1*size_1      :{p_1 * size_1}')
            print(f'(1-p_1)*size_1  :{(1 - p_1) * size_1}\n')

        if min(p_2 * size_2, (1 - p_2) * size_2) < 10:
            print('Warning!')
            print('Group 2 [np >= 10] or [(1-p)*n >= 10] requirement not met')
            print(f'p_2*size_2      :{p_2 * size_2}')
            print(f'(1-p_2)*size_2  :{(1 - p_2) * size_2}\n')

        # Print simulation results
        print(f'Using Simulation:')
        print(f'time: {end - start:.3f}s')
        print(f'proportions: {p_1:.3f} and {p_2:.3f}')
        print(f'size of sample sets: {size_1} and {size_2}')
        print(f'proportion difference: {p_diff:.3f}')
        print(f'\nConfidence Interval ({l_tail} to {r_tail}): ({ci[0]:.3f} to {ci[1]:.3f}), length {ci[2]:.3f}\n')

        # Plot histogram
        x_label = 'difference in two proportions'
        hist_title = f'Confidence Interval of difference in two proportions'
        ci_histplot(b_array, ci, size, x_label, hist_title)
        return ci


    def ci_diffstatistics(data_1, data_2, n=10000, l_tail=5, r_tail=95, stat='mean'):
        """
        Returns the confidence interval for the difference in two statistics

        Parameter(s):
        data_1 (array) is the dataset for first mean
        data_2 (array) is the dataset for second mean
        n (int) is the number of bootstrap simulations
        l_tail (float) is the cut off percentile for the left tail [0,100]
        r_tail (float) is the cut off percentile for the right tail [0,100]
        stat (string) is the statistic being analyzed ["mean","median","stdev"]
        """
        start = time.time()

        # determine dataset sizes
        size_1 = data_1.shape[0]
        size_2 = data_2.shape[0]
        size = size_1 + size_2

        # calculate statistics
        if stat == 'mean':
            data_1_stat = np.mean(data_1)
            data_2_stat = np.mean(data_2)
        elif stat == 'median':
            data_1_stat = np.median(data_1)
            data_2_stat = np.median(data_2)
        elif stat == 'stdev':
            data_1_stat = np.std(data_1)
            data_2_stat = np.std(data_2)
        else:
            raise Exception("invalid input for statistic type")

        stat_diff = data_1_stat - data_2_stat

        # Bootstrap samples per reallocation method
        rng = np.random.default_rng()
        b_array = np.empty(0)

        for _ in range(n):
            b_1 = rng.choice(data_1, size_1, replace=True)
            b_2 = rng.choice(data_2, size_2, replace=True)

            if stat == 'mean':
                b_1_stat = np.mean(b_1)
                b_2_stat = np.mean(b_2)
            elif stat == 'median':
                b_1_stat = np.median(b_1)
                b_2_stat = np.median(b_2)
            elif stat == 'stdev':
                b_1_stat = np.std(b_1)
                b_2_stat = np.std(b_2)
            else:
                raise Exception("invalid input for statistic type")

            b_diff = b_1_stat - b_2_stat
            b_array = np.append(b_array, b_diff)

        # Confidence Interval via simulation
        ci = simulation_ci(b_array, l_tail, r_tail)
        end = time.time()

        # calculate with theoretical distribution

        if stat == 'mean':

            print(f'Using Theoretical Distn:')

            if size_1 < 30:
                print(f'Warning! Sample one (size: {size_1}) is less than 30 \n')

            if size_2 < 30:
                print(f'Warning! Sample two (size: {size_2}) is less than 30 \n')

            dof = min(size_1, size_2) - 1
            s_1 = np.sqrt(data_1.var(ddof=1))
            s_2 = np.sqrt(data_2.var(ddof=1))
            se = np.sqrt(s_1 ** 2 / size_1 + s_2 ** 2 / size_2)

            left_tval, right_tval, length_tval, l_tstat, r_tstat = ci_ttest(l_tail, r_tail, dof, stat_diff, se)

            alpha = (r_tail - l_tail) / 100

            left_check, right_check = stats.t.interval(alpha, df=dof, loc=stat_diff, scale=stats.sem(data_1 - data_2))
            length_check = right_check - left_check

            print(f't-stat: {l_tstat:.3f} and {r_tstat:.3f}')
            print(f'confidence interval w/ formula  : {left_tval:.3f} to {right_tval:.3f}, length {length_tval:.3f}')
            print(
                f'confidence interval from scipy  : {left_check:.3f} to {right_check:.3f}, length {length_check:.3f}\n')

        # Print simulation results
        print(f'Using Simulation:')
        print(f'time: {end - start:.3f}s')
        print(f'sample statistic: {stat_diff:.3f}')
        print(f'size of sample set: {size}')
        print(f'Confidence Interval ({l_tail} to {r_tail}): ({ci[0]:.3f} to {ci[1]:.3f}), length {ci[2]:.3f}')

        # Plot histogram
        x_label = stat
        hist_title = f'Confidence Interval of {stat}'
        ci_histplot(b_array, ci, size, x_label, hist_title)

        return ci


    def ci_slope(features, targets, n=10000, l_tail=5, r_tail=95):
        """
        Returns simulation p-value (float) for hypothesis test of a slope (correlation) of two datasets
        Null hypothesis is that there is no correlation between the two datasets

        Parameter(s):
        features (array) is a single dimension array of features
        targets (array) is a single dimension array of targets
        n (int) is the number of bootstrap simulations
        alt (string) defines alt hypothesis ['two-sided, 'less', 'greater']
        """

        start = time.time()

        # perform regression to find slope of the sample

        regr = linear_model.LinearRegression().fit(features.reshape(-1, 1), targets.reshape(-1, 1))
        sample_slope = float(regr.coef_)

        # Bootstrap the slope
        rng = np.random.default_rng()
        size = features.shape[0]
        b_array = np.empty(0)
        data = np.column_stack((features, targets))

        for _ in range(n):
            b_data = rng.choice(data, size, replace=True)
            regr = linear_model.LinearRegression().fit(b_data[:, 0].reshape(-1, 1), b_data[:, 1].reshape(-1, 1))
            b_slope = regr.coef_
            b_array = np.append(b_array, b_slope)

        # Calculate confidence interval

        ci = simulation_ci(b_array, l_tail, r_tail)
        end = time.time()

        # Verify with theoretical distribution and check w/ statsmodels OLS

        lm_features = np.column_stack((features, np.ones(size)))
        lm = sm.OLS(targets, lm_features).fit()
        se = lm.bse[0]

        print(f'\nUsing Theoretical Distn:')
        dof = size - 2
        t_stat = float(sample_slope / se)

        left_tval, right_tval, length_tval, l_tstat, r_tstat = ci_ttest(l_tail, r_tail, dof, sample_slope, se)

        print(f't-stat: {r_tstat:.4f} and {l_tstat:.3f}')
        print(f'confidence interval w/ formula  : {left_tval:.3f} to {right_tval:.3f}, length {length_tval:.3f}\n')

        # Print simulation results
        print(f'Using Simulation:')
        print(f'time: {end - start:.3f}s')
        print(f'slope of sample: {sample_slope:.3f}')
        print(f'size of sample set: {size}')
        print(f'Confidence Interval ({l_tail} to {r_tail}): ({ci[0]:.3f} to {ci[1]:.3f}), length {ci[2]:.3f}')

        # plot histogram
        x_label = "slope"
        hist_title = "Confidence interval of a slope"
        ci_histplot(b_array, ci, size, x_label, hist_title)

        return ci



    '''
    3.0 Sample size estimations
    '''

    def samplesize_stat(ci, stdev, margin):
        """
        Returns the minimum required sample size (int) given a confidence interval, standard deviation, and sample size for a statistic

        Parameter(s):
        ci_perc (float) is the desired confidence interval as a percent (0,100)
        stdev (float) is the estimated population standard deviation
        margin (float) is the desired margin of error of the statistic
        
        """

        z = stats.norm.ppf((1-ci)/2)
        n_tot = np.ceil((z*stdev/margin)**2)

        return z, n_tot


    def marginsize_stat(ci, stdev, size):
        """
        Returns the margin size (float) given a confidence interval, standard deviation, and sample size for a statistic

        Parameter(s):
        ci_perc (float) is the desired confidence interval as a percent (0,100)
        stdev (float) is the estimated population standard deviation
        margin (float) is the desired margin of error of the statistic
        
        """

        z = stats.norm.ppf((1-ci)/2)
        margin = np.sqrt((z*stdev)**2/size)

        return z, margin



    def samplesize_prop(ci, p_est = 0.5, margin = 0.05):
        """
        Returns the minimum required sample size (int) given a confidence interval, est. proportion, and margin of error for a proportion

        Parameter(s):
        ci_perc (float) is the desired confidence interval as a percent (0,100)
        stdev (float) is the estimated population standard deviation
        margin (float) is the desired margin of error of the statistic
        
        """
        z = stats.norm.ppf((1-ci)/2)
        n_total = np.ceil((z/margin)**2*(p_est*(1-p_est)))

        return z, n_total


    def marginsize_prop(ci, p_est = 0.5, size = 100):
        """
        Returns the margin (float) given a confidence interval, est. proportion, and sample size for a proportion

        Parameter(s):
        ci_perc (float) is the desired confidence interval as a percent (0,100)
        stdev (float) is the estimated population standard deviation
        margin (float) is the desired margin of error of the statistic
        
        """
        z = stats.norm.ppf((1-ci)/2)
        margin = np.sqrt(z**2/size*(p_est*(1-p_est)))

        return z, margin
