"""
pol_metrics.py

Scripts to measure 4 polarization metrics (as defined by DiMaggio) of a
distribution of preferences.
"""

import csv
import os.path
import numpy as np
import pandas as pd

from scipy.stats import moment, kurtosis, uniform, laplace
from scipy.misc import comb, factorialk
from sklearn import mixture
from bimixture_mh import *


"""
weighted_mean_byclust
    Computes mean of inferred distribution by the ratio of the population of
    precinct i to the total population.
"""
def weighted_mean_byclust(clust_df):
    return sum(clust_df['prop_votes'] * clust_df['loc'])

"""
weighted_var_byclust
    Computes variance of inferred distribution by ratio of the population
    of precinct i to the total population.
"""
def weighted_var_byclust(clust_df, mean):
    tmp_series = (clust_df['loc'] ** 2) + (clust_df['scale'] ** 2)
    return sum(clust_df['prop_votes'] * tmp_series) - (mean ** 2)


"""
mixture_central_moment
    Computes the central moment of the mixture distribution
"""
def mixture_central_moment(clust_df, moment, mean, prec_dist):
    mixture_moment = 0.0

    for asst, row in clust_df.iterrows():

        for k in range(1, moment + 1):
            tmp = comb(moment, k) * ((row['loc'] - mean) ** (moment - k)) * \
                    row['prop_votes']

            if prec_dist == 'norm':
                tmp = tmp * normal_central_moment(row['scale'], k)
            elif prec_dist == 'uniform':
                tmp = tmp * uniform_central_moment(row['loc'], row['scale'], k)
            elif prec_dist == 'laplace':
                tmp = tmp * laplace_central_moment(row['loc'], row['scale'], k)
            else:
                print '[error] prec_dist moment not found'
                assert(False)

            mixture_moment += tmp

    return mixture_moment


"""
normal_central_moment
    Computes the central moments of a normal distribution
"""
def normal_central_moment(std, moment):
    if moment % 2 == 1:
        return 0.0
    else:
        return (std ** moment) * factorialk(moment - 1, 2)


"""
uniform_central_moment
"""
def uniform_central_moment(loc, scale, moment):
    if moment < 1 or moment > 4:
        print 'Unable to compute moment of uniform distribution ):'
        assert(False)
    moments_arr = uniform.stats(loc=loc, scale=scale, moments='mvsk')
    return moments_arr[moment - 1]


"""
laplace_central_moment
"""
def laplace_central_moment(loc, scale, moment):
    if moment < 1 or moment > 4:
        print 'Unable to compute moment of laplace distribution ):'
        assert(False)
    moments_arr = laplace.stats(loc=loc, scale=scale, moments='mvsk')
    return moments_arr[moment - 1]



"""
dim_kurtosis
    Measures bimodality principle of DiMaggio
"""
def dim_kurtosis(clust_df, mean, variance, prec_dist):
    return (mixture_central_moment(clust_df, 4, mean, prec_dist) / \
            (variance ** 2)) - 3.0


"""
difference_of_means
    Fits a set of data points to a 2 mixture model and then returns the
    difference of the two mean values.

    NOTE: We divide the true difference by the standard deviation.
"""
def difference_of_means(data, std_dev):
    fitted_param = run(data)

    min_mean = min(fitted_param['mean0'], fitted_param['mean1'])
    max_mean = max(fitted_param['mean0'], fitted_param['mean1'])

    return (max_mean - min_mean)


"""
get_state_data
    Gets the relevant voter and candidate data for the given state and election
    information.
"""
def get_state_data(state, year, prec_dist, num_clust, path_to_results):
    ## for voters
    voter_dir = '%s/results/%s/%s_clusters/mh_%s/%s/charts' % (path_to_results,
            state, num_clust, year, prec_dist)

    prec_df = pd.read_csv('%s/prec_asst_cd.csv' % voter_dir).groupby('asst')
    clust_df = pd.DataFrame({
        'loc': prec_df['loc'].first(),
        'scale': prec_df['scale'].first(),
        'tot_votes': prec_df['tot_votes'].aggregate(np.sum)
    })

    param_df = pd.read_csv('%s/inferred_param_df.csv' % voter_dir)

    cand_file = '%s/data/%s_data/%s_%s/%s_cand_%s.csv' % (path_to_results,
        state, state, year, state, year)
    cand_df = None
    if os.path.isfile(cand_file):
        cand_df = pd.read_csv(cand_file)

    return clust_df, param_df, cand_df


def pol_metrics_one_state(state, year, prec_dist, num_clust, path_to_results,
    outdir):
    clust_df, param_df, cand_df = get_state_data(state, year,
            prec_dist, num_clust, path_to_results)
    return pol_metrics_main(clust_df, param_df, cand_df, prec_dist, state,
        num_clust, year, outdir)


def pol_metrics_combined(state_arr, year, prec_dist, num_clust,
        path_to_results, outdir):
    clust_df = pd.DataFrame()
    param_df = pd.DataFrame()
    cand_df = pd.DataFrame()

    for state in state_arr:
        tmp_clust_df, tmp_param_df, tmp_cand_df = get_state_data(state, year,
                prec_dist, num_clust, path_to_results)

        clust_df = pd.concat([clust_df, tmp_clust_df])
        param_df = pd.concat([param_df, tmp_param_df], ignore_index=True)
        cand_df = pd.concat([cand_df, tmp_cand_df])

    # need to balance theta
    param_df['theta'] = param_df['theta'] / len(state_arr)
    return pol_metrics_main(clust_df, param_df, cand_df, prec_dist, 'ne',
        num_clust, year, outdir)


def pol_metrics_main(clust_df, param_df, cand_df, prec_dist, state, num_clust,
    year, outdir):

    outfile = open("{}/pol_metrics.csv".format(outdir), 'w+')
    pol_csv = csv.writer(outfile, delimiter=',')
    pol_csv.writerow(["metric_name", "value"])

    clust_df['prop_votes'] = clust_df['tot_votes'] / \
            float(sum(clust_df['tot_votes']))

    voter_mean = weighted_mean_byclust(clust_df)
    pol_csv.writerow(["voter_mean", voter_mean])
    voter_var = weighted_var_byclust(clust_df, voter_mean)
    pol_csv.writerow(["voter_variance", voter_var])
    voter_std = np.sqrt(voter_var)
    pol_csv.writerow(["voter_std", voter_std])
    pol_csv.writerow(["voter_kurtosis", dim_kurtosis(clust_df, voter_mean,
      voter_var, prec_dist)])

    # sample from the aggregate voter distribution
    voter_sample = []
    for i in xrange(10000):
        clust = np.random.choice(a=param_df.index, p=param_df['theta'])
        rand_val = -1
        if prec_dist == 'norm':
            rand_val = np.random.normal(param_df.loc[clust, 'loc'], param_df.loc[clust, 'scale'])
        elif prec_dist == 'uniform':
            rand_val = np.random.uniform(param_df.loc[clust, 'loc'], param_df.loc[clust, 'scale'])
        elif prec_dist == 'laplace':
            rand_val = np.random.laplace(param_df.loc[clust, 'loc'], param_df.loc[clust, 'scale'])
        else:
            print 'Cannot generate sample from aggregate voter distribution ):'
            assert(False)

        voter_sample.append(rand_val)

    pol_csv.writerow(["voter_diff_means", difference_of_means(voter_sample,
      voter_std)])

    ## for candidates
    if type(cand_df) != pd.DataFrame:
      print '[error] cand_df could not be found! See pol_metrics.py'
      return
    pol_csv.writerow(["cand_mean", np.mean(cand_df['cf_score'])])
    cand_var = np.var(cand_df['cf_score'])
    pol_csv.writerow(["cand_variance", cand_var])
    cand_std = np.sqrt(cand_var)
    pol_csv.writerow(["cand_std", cand_std])
    pol_csv.writerow(["cand_kurtosis", kurtosis(cand_df['cf_score'])])
    pol_csv.writerow(["cand_diff_means", difference_of_means(cand_df['cf_score'],
      cand_std)])

    outfile.close()

def main():
    pol_metrics_state = raw_input('Pol Metrics Combined (0) or One State (1): ')
    if pol_metrics_state == "1":
        state = raw_input('State: ')
        year = raw_input('Year: ')
        prec_dist = raw_input('Prec Dist: ')
        num_clust = raw_input('Number of Clusters: ')

        outdir = '../results/%s/%s_clusters/mh_%s/%s/charts' % (state,
          num_clusters, year, precdist)

        pol_metrics_one_state(state, year, prec_dist, num_clust, '../', outdir)
    elif pol_metrics_state == "0":
        # for now, hard-coding
        state_arr = ['tx', 'ny']
        year = raw_input('Year: ')
        prec_dist = raw_input('Prec Dist: ')
        num_clust = raw_input('Number of Clusters: ')
        outdir = raw_input('Output filepath: ')

        pol_metrics_combined(state_arr, year, prec_dist, num_clust, '../',
            outdir)
    else:
        print 'Bad value.'


if __name__ == "__main__":
    main()

