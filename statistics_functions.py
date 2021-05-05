# import relevant packages
 
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest


if __name__ == "__main__":

    print("This is a set of statistics and probability functions for DSC 381")


else:
    
    '''
    0.0 Section is a set of helper functions for bootstrap simulations
    '''

    # probability value tests
    def simulation_pval(b_array, null_hyp, sample, alt):
        '''
        Takes in a evaluated bootstrap array and calculates against the sample value

        Parameter(s):
        b_array (array) an array of simulated bootstrap statistics
        null_hyp (float) is the null hypothesis statistic
        sample (float) is the sample statistic
        alt (string) is the type of test ['less', 'greater', 'two-sided']
        '''

        if alt == 'less':
            p_val = np.mean(b_array <= sample)
        elif alt == 'greater':
            p_val = np.mean(b_array >= sample)
        else:
            if null_hyp > sample:
                p_val = 2*np.mean(b_array <= sample)
            else:
                p_val = 2*np.mean(b_array >= sample)
        return p_val

    def ztest_numpy(z_stat, alt):
        '''
        takes a z_statistic and returns a z_val

        Parameter(s):
        z_stat (float) is the calculated z statistic
        alt (string) is the type of test ['less', 'greater', 'two-sided']
        '''
        if alt == 'less':
            z_val = stats.norm.cdf(z_stat)
        elif alt == 'greater':
            z_val = stats.norm.sf(z_stat)
        else:
            if z_stat<=0:
                z_val = 2*stats.norm.cdf(z_stat)
            else:
                z_val = 2*stats.norm.sf(z_stat)

        print(f'z-stat, z-val using numpy           :{z_stat:.4f}, {z_val:.4f}')

        return z_val

    def ttest_numpy(t_stat, dof, alt):
        '''
        takes a t_statistic and returns a z_val

        Parameter(s):
        t_stat (float) is the calculated z statistic
        dof (int) is the degrees of freedom 
        alt (string) is the type of test ['less', 'greater', 'two-sided']
        '''
        if alt == 'less':
            t_val = stats.t.cdf(t_stat, dof)
        elif alt == 'greater':
            t_val = stats.t.sf(t_stat, dof)
        else:
            if t_stat<=0:
                t_val = 2*stats.t.cdf(t_stat, dof)
            else:
                t_val = 2*stats.t.sf(t_stat, dof)

        print(f't-stat, t-val using numpy:          :{t_stat:.4f}, {t_val:.4f}')

        return t_val

    def sm_alt_convert(alt):
        '''
        Converts scipy alt terminology to statsmodels.api terminology

        Parameter(s):
        alt (string) ['less', 'greater', 'two-sided'] converts to ['smaller', 'larger', 'two-sided']
        '''
        if alt == 'less':
            sm_alt = 'smaller'
        elif alt == 'greater':
            sm_alt = 'larger'
        else:
            sm_alt = 'two-sided'
        return sm_alt

    def hyptest_histplot(b_array, sim_hyp, test_hyp, size, alt, x_label, hist_title):
        '''
        Generates a histogram plot for hypothesis test

        Parameter(s):
        b_array (array) bootstrap simulation results
        sim_hyp (float) the assumed statistic for simulation (center of simulation)
        test_hyp (float) the tested statistic for simulation
        alt (string) the type of test ['less', 'greater', 'two-sided']
        x_label (string) x label title
        hist_title (string) histogram title

        '''

        plt.figure(figsize=(8,4))
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel("bootstrap samples", fontsize=12)
        plt.title(hist_title)

        plt.hist(b_array, bins = size);
        plt.axvline(x = test_hyp, c = 'r')

        diff = sim_hyp - test_hyp
        bar_1 = sim_hyp + diff
        bar_2 = sim_hyp - diff

        if alt == 'two-sided':
            plt.axvline(x = bar_1, c = 'r')
            plt.axvline(x = bar_2, c = 'r')

        plt.show();

    
    
    
    
    '''
    1.0 Hypothesis testing functions
    '''

    # Randomization Hypothesis Testing Functions

    def hyptest_singlemean(data, null_hyp = 0, n = 10000, alt = 'two-sided'):
        '''
        Returns simulation p-value (float) for hypothesis test of a single mean.

        Parameter(s):
        data (array) is the dataset for analysis
        null_hyp (float) is the null hypothesis
        n (int) is the number of bootstrap simulations
        alt (string) defines alt hypothesis ['two-sided, 'less', 'greater']
        '''
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
            b_data = rng.choice(data_shift, size, replace = True)
            b_mean = np.mean(b_data)
            b_array = np.append(b_array, b_mean)

        # Tested condition for p_value
        p_val = simulation_pval(b_array, null_hyp, mu_sample, alt)
        end = time.time()

        # Verify with theoretical distribution

        print(f'\nUsing Theoretical Distn:')

        if size < 30:
            print('Warning! Sample size is less than 30 \n')
        
        dof = size-1
        s = data.var(ddof = 1)
        se = np.sqrt(s/size)

        t_stat = (mu_sample - null_hyp)/se

        # calculate the t_stat
        t_val = ttest_numpy(t_stat, dof, alt)

        # validate with scipy
        check_t_stat, check_t_val = stats.ttest_1samp(data, null_hyp, alternative = alt)

        print(f't-stat, t-val from scipy stats      :{check_t_stat:.4f}, {check_t_val:.4f}')
        print(f'dof: {dof}\n')

        # print results and plot histogram
        print(f'Using Simulation:')
        print(f'time: {end-start:.3f}s')
        print(f'mu of sample: {mu_sample:.3f}')
        print(f'\nCalculated p_value: {p_val}')

        # plot histogram
        x_label = "mu"
        hist_title = "Hypothesis test of a single mean"
        hyptest_histplot(b_array, null_hyp, mu_sample, size, alt, x_label, hist_title)

        return p_val


    def hyptest_singleprop(p_sample, p_null = 0.5, size = 30, n = 10000, alt = 'two-sided', disp = 'count'):
        '''
        Returns simulation p-value (float) for hypothesis test of a single proportion.

        Parameter(s):
        p_sample (float) is the sample proportion
        p_null (float) is the null hypothesis proportion
        size (int) is the sample size
        n (int) is the number of bootstrap simulations
        alt (string) defines alt hypothesis ['two-sided, 'less', 'greater']
        disp (string) choose plot display ['count', 'prop']
        '''
        start = time.time()
        
        count_sample = p_sample*size
        null_hyp = p_null*size

        p_diff = np.abs(p_sample - p_null)
        count_diff = np.abs(count_sample - null_hyp)

        # Bootstrap sample - rounding causes issues so processing with counts first
        rng = np.random.default_rng()
        b_array = np.empty(0)

        for _ in range(n):
            b_data = rng.choice([0,1], size, replace = True, p = [1-p_null, p_null])
            b_mean = np.sum(b_data)
            b_array = np.append(b_array, b_mean)

        # Tested condition for p_value
        p_val = simulation_pval(b_array, null_hyp, count_sample, alt)

        end = time.time()

        print(f'\nUsing Theoretical Distn:')

        # Verify with theoretical distribution
        se = np.sqrt(p_null*(1-p_null)/size)
        z_stat = (p_sample - p_null)/se

        #normal distribution check
        z_val = ztest_numpy(z_stat, alt)

        # validate with statsmodels
        sm_alt = sm_alt_convert(alt)

        check_z_stat, check_z_val = proportions_ztest(count_sample, size, value = p_null, alternative = sm_alt, prop_var = p_null)
        print(f'z-stat, z-val from statsmodels      :{check_z_stat:.4f}, {check_z_val:.4f}')

        if min(p_null*size, (1-p_null)*size) < 10:
            print('\nWarning! np >= 10 or (1-p)*n >= 10 requirement not met')
            print(f'p_null*size: {p_null*size:.1f} (1-p_null)*size): {(1-p_null)*size:.1f}\n')

        # print results
        print(f'\nUsing Simulation:')
        print(f'time: {end-start:.3f}s')
        print(f'count of sample set: {count_sample:.1f}')
        print(f'size of sample set: {size}')
        print(f'\nCalculated p_value: {p_val}')

        if disp == 'prop':
            b_array /= size
            null_hyp /= size
            count_sample /= size

        # plot histogram
        x_label = disp
        hist_title = "Hypothesis test of a single proportion"
        hyptest_histplot(b_array, null_hyp, count_sample, 2*size, alt, x_label, hist_title)

        return p_val

    

    def hyptest_diffmeans(data_1, data_2, n = 10000, alt = 'two-sided'):
        '''
        Returns simulation p-value (float) for hypothesis test of a difference in two means
        Null hypothesis is defined as the two means being equal

        Parameter(s):
        data_1 (array) is the dataset for first mean
        data_2 (array) is the dataset for second mean
        n (int) is the number of bootstrap simulations
        alt (string) defines alt hypothesis ['two-sided, 'less', 'greater']
        '''
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
            b_data = rng.choice(data, size, replace = False)
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

        s_1 = np.sqrt(data_1.var(ddof = 1))
        s_2 = np.sqrt(data_2.var(ddof = 1))

        se = np.sqrt(s_1**2/size_1 + s_2**2/size_2)
        t_stat = ((mu_1 - mu_2) - 0)/se

        # calculate the t_stat
        t_val = ttest_numpy(t_stat, dof, alt)

        # validate with scipy stats
        check_t_stat, check_t_val = stats.ttest_ind(data_1, data_2, equal_var = True, alternative = alt)
        
        print(f't-stat, t-val from stat module      :{check_t_stat:.4f}, {check_t_val:.4f}')
        print(f'dof: {dof}\n')

        if size_1 < 30:
            print(f'Warning! Sample one (size: {size_1}) is less than 30 \n')

        if size_2 < 30:
            print(f'Warning! Sample two (size: {size_2}) is less than 30 \n')

        print(f'Using Simulation:')

        # print results
        print(f'time: {end-start:.3f}s')
        print(f'sample mean diff: {mu_diff:.3f}')
        print(f'size of sample set (1 and 2): {size_1} and {size_2}')
        print(f'Calculated p_value: {p_val}')

        x_label = 'mu'
        hist_title = 'Difference in two means'
        hyptest_histplot(b_array, null_hyp, mu_diff, size, alt, x_label, hist_title)

        return p_val


    def hyptest_diffprops(p_1, size_1, p_2, size_2, n = 10000, alt = 'two-sided'):
        '''
        Returns simulation p-value (float) for hypothesis test of a difference in two proportions
        Null hypothesis is defined as the two proportions being equal

        Parameter(s):
        p_1 (float) is proportion for dataset 1
        size_1 (int) is the size of dataset 1
        p_2 (float) is the proportion for dataset 2
        size_2 (int) is the size of dataset 2
        n (int) is the number of bootstrap simulations
        alt (string) defines alt hypothesis ['two-sided, 'less', 'greater']
        '''

        start = time.time()

        # values for group 1
        count_1 = np.round(p_1*size_1)

        # values for group 2
        count_2 = np.round(p_2*size_2)

        # calculate differences
        count_diff = count_1 - count_2
        p_diff = p_1 - p_2

        # define null hypothesis of p_1 = p_2
        size = size_1 + size_2
        count_total = count_1 + count_2
        p_null = count_total/size
        p_null_diff = 0
        null_diff = p_null*size_1 - p_null*size_2

        # Bootstrap samples per reallocation method
        rng = np.random.default_rng()
        b_array = np.empty(0)
        b_p_array = np.empty(0)
        null_hyp_array = np.concatenate([np.zeros(size - int(count_total)), np.ones(int(count_total))])

        for _ in range(n):
            b_data = rng.choice(null_hyp_array, size, replace = False)
            b_1 = b_data[0:size_1]
            b_2 = b_data[size_1:size]

            b_diff = sum(b_1) - sum(b_2)
            b_array = np.append(b_array, b_diff)
            
            b_p_diff = sum(b_1)/size_1 - sum(b_2)/size_2
            b_p_array = np.append(b_p_array, b_p_diff)

        # Tested condition for p_value
        p_val = simulation_pval(b_array, null_diff, count_diff, alt)

        # convert graph from count to proportions
        b_array = b_p_array
        end = time.time()

        # calculate with theoretical distribution
        print(f'Using Theoretical Distn:')

        dof = min(size_1, size_2) - 1
        se = np.sqrt(p_null*(1-p_null)/size_1 + p_null*(1-p_null)/size_2)
        z_stat = ((p_1 - p_2) - 0)/se

        #normal distribution check
        z_val = ztest_numpy(z_stat, alt)

        # validate with statsmodels
        sm_alt = sm_alt_convert(alt)
        
        # validate with statsmodels
        check_z_stat, check_z_val = proportions_ztest([count_1, count_2], [size_1, size_2], value = 0, alternative = sm_alt)
        print(f'z-stat, z-val from statsmodels      :{check_z_stat:.4f}, {check_z_val:.4f}')
        print(f'dof: {dof}\n')

        if min(p_1*size_1, (1-p_1)*size_1) < 10:
            print('Warning! group 1 np >= 10 or (1-p)*n >= 10 requirement not met')
            print(f'p_1*size_1: {p_1*size_1} (1-p_1)*size_1: {(1-p_1)*size_1}\n')

        if min(p_2*size_2, (1-p_2)*size_2) < 10:
            print('Warning! group 2 np >= 10 or (1-p)*n >= 10 requirement not met')
            print(f'p_2*size_2: {p_2*size_2} (1-p_2)*size_2: {(1-p_2)*size_2}\n')

        print(f'Using Simulation:')

        print(f'time: {end-start:.3f}s')

        print(f'dataset 1 count: {count_1}')
        print(f'dataset 2 count: {count_2}')
        print(f'sample count diff: {count_diff}')
        print(f'sample p diff: {p_diff:.3f}')
        print(f'Calculated p_value: {p_val}')

        x_label = 'proportion difference'
        hist_title = 'Hypothesis test of two proportions'

        hyptest_histplot(b_array, p_null_diff, p_diff, size, alt, x_label, hist_title)

        return p_val


    def hyptest_slope():
        return null

    # Bootstrap confidence intervals

    def ci_statistic():
        return null

    def ci_prop():
        return null

    def ci_diffstatistics():
        return null

    def ci_slope():
        return null


    # Chi-squared Goodness of fit



    # Chi-squared test for association




    # ANOVA difference in means




    # ANOVA for regression