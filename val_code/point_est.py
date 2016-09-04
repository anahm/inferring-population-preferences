"""
point_est.py

Uses inference results to compute precinct assignment variables, polarization
metrics, and district-level preference approximations
"""

from scipy.stats import binom, norm, laplace, uniform
from pol_metrics import *

import numpy as np
import os
import pandas as pd


"""
arr_to_df()
    Function that converts a dataframe row of parameter values into a pandas
    dataframe.

    @param: param_arr := array of parameters from mle_inferred_vals.csv
            [index, subdir, theta0, theta1, etc...NA, NA.1, index, loc0, loc1, etc]
"""
def arr_to_df(param_row, num_clusters):
    theta_arr = []
    loc_arr = []
    scale_arr = []

    for i in xrange(num_clusters):
        theta_arr.append(param_row['theta%d' % i])
        loc_arr.append(param_row['loc%d' % i])
        scale_arr.append(param_row['scale%d' % i])
        i += 3

    param_df = pd.DataFrame({
        'theta': theta_arr,
        'loc': loc_arr,
        'scale': scale_arr
      },
      columns=['theta', 'loc', 'scale'])

    return param_df


"""
init()
    Function that reads in data about the precinct and the final inferred
    parameters.

    @param: state_lst, year, outdir
    @ret: prec_df, param_df (rows of param per cluster), outdir (orig param dir)
"""
def init(state_lst, year, outdir, num_clusters, path_to_data):
    # gather all congdist info for states in consideration
    prec_df_lst = []
    for state in state_lst:
        precfile = '%s/data/%s_data/%s_%s/precline_%s_house_%s.csv' % \
                (path_to_data, state, state, year, state, year)
        prec_df = pd.read_csv(precfile)

        # now to merge the additional info (cong_dist) to prec_df
        prec_cd_file = '%s/data/%s_data/%s_%s/%s_%s_prec_cd.csv' % \
                (path_to_data, state, state, year, state, year)
        prec_df = pd.merge(prec_df, pd.read_csv(prec_cd_file),
                how='left', on='geoid')

        prec_df_lst.append((state, prec_df))

    # gathered all parameter info
    if os.path.exists('%s/inferred_param_df.csv' % outdir):
        param_df = pd.read_csv('%s/inferred_param_df.csv' % outdir)
    elif os.path.exists('%s/mle_inferred_vals.csv' % outdir):
        # mle_inferred_vals.csv format from data_results
        param_row_df = pd.read_csv('%s/mle_inferred_vals.csv' % outdir)
        param_df = arr_to_df(param_row_df.iloc[1], num_clusters)
        param_df.sort_values(by='loc', inplace=True)
        param_df.index = xrange(num_clusters)

        param_outfilename = '%s/inferred_param_df.csv' % outdir
        param_df.to_csv(param_outfilename, index=False)
        print 'Wrote converted param_df to: %s' % param_outfilename
    elif os.path.exists('%s/scipy_init.csv' % outdir):
        print 'Need to convert results.txt to inferred_param_df.csv!'
        assert(False)
    else:
        print 'File format not recognized. See init() function. ):'
        assert(False)

    return prec_df_lst, param_df


"""
compute_asst_lik
    Computes the likelihood of a given assignment variable i.

    @param: prec_df, param_df, i (assignment variable)
    @ret: DataFrame series of the computed likelihood of each precinct having
          assignment variable i
"""
def compute_asst_loglik(prec_df, param_df, i, precdist):
    theta = param_df.loc[i, 'theta']
    loc = param_df.loc[i, 'loc']
    scale = param_df.loc[i, 'scale']

    if precdist == 'uniform':
        cdf_val = uniform.cdf(prec_df['midpoint'], loc, scale)
    elif precdist == 'norm':
        cdf_val = norm.cdf(prec_df['midpoint'], loc, scale)
    elif precdist == 'laplace':
        cdf_val = laplace.cdf(prec_df['midpoint'], loc, scale)
    elif precdist == 'mixnorm':
        cdf_val = 0.5 * norm.cdf(prec_df['midpoint'], loc, scale) + \
            0.5 * norm.cdf(prec_df['midpoint'], -loc, scale)
    else:
        print 'Cannot find cdf given precinct distribution, see ' + \
                'compute_asst_loglik().'
        assert(False)

    return binom.logpmf(prec_df['num_votes_0'], prec_df['tot_votes'],
            cdf_val) + np.log(theta)


"""
select_asst
    Selects the assignment variable that is the argmax to the likelihood of the
    precinct given an assignment variable. Does this for all precincts.

    @param: prec_df (_with_ all the computed asst likelihoods)
    @ret: pandas series of the assignment variables
"""
def select_asst(prec_df, colnames):
    prec_df['asst'] = prec_df[colnames].idxmax(axis=1)

    # BUG: NY datasets have some rep vote share == 0, causes all rows to be -inf
    # also, idxmax returns 'nan' whenever it cannot find a max value
    prec_df.dropna(inplace=True)

    prec_df['asst'] = prec_df['asst'].astype(int)
    return


"""
weighted_means_bydist
    Function that computes the average of the means of all precincts in
    the same district weighted by the number of voters in each precinct.

    @param: prec_df, param_df
    @ret: dist_means_df
"""
def weighted_means_bydist(prec_df, param_df, state):
    prec_df['loc'] = prec_df['asst'].apply(lambda z: param_df.loc[z, 'loc'])
    prec_df['scale'] = prec_df['asst'].apply(lambda z: param_df.loc[z, 'scale'])

    grouped_df = prec_df.groupby('congdist')

    cd_arr = []
    weighted_mean_arr = []
    winner_arr = []
    losing_arr = []

    overall_weighted_mean = 0

    for cd, grp in grouped_df:
        # take weighted avg of inferred means
        weighted_mean = sum(grp['loc'] * grp['tot_votes']) / \
                float(sum(grp['tot_votes']))

        overall_weighted_mean += sum(grp['loc'] * grp['tot_votes'])
        cd_arr.append(cd)
        weighted_mean_arr.append(weighted_mean)

        winning_cf = grp['cf_score_0'].iloc[0] if grp['num_votes_0'].iloc[0] > \
                grp['num_votes_1'].iloc[0] else grp['cf_score_1'].iloc[0]
        losing_cf = grp['cf_score_0'].iloc[0] if grp['num_votes_0'].iloc[0] <= \
                grp['num_votes_1'].iloc[0] else grp['cf_score_1'].iloc[0]

        winner_arr.append(winning_cf)

    overall_weighted_mean = overall_weighted_mean / float(sum(prec_df['tot_votes']))
    print 'Overall Weighted Mean: %f' % overall_weighted_mean

    return pd.DataFrame({
        'congdist': cd_arr,
        'blargl_weighted_mean': weighted_mean_arr,
        'winner_score': winner_arr,
        'tot_votes': grouped_df['tot_votes'].agg(np.sum),
        'dem_votes': grouped_df['num_votes_0'].sum(),
        'state': state
    })


"""
write_to_file
    Function that writes the assignment variables to a file that will be in the
    same directory as where the inferred parameters were all found.

    @param: prec_df, outdir
"""
def write_to_file(prec_df_lst, dist_means_df_lst, outdir):
    asst_filename = '%s/prec_asst_cd.csv' % outdir
    prec_df = pd.concat(prec_df_lst)
    prec_df[['geoid', 'asst', 'congdist', 'state', 'loc', 'scale', 'num_votes_0',
        'tot_votes']].to_csv(asst_filename, index=False)
    print 'Stored precinct cluster asst and congdist to %s!' % asst_filename

    # dist_means_df
    distmeans_filename = '%s/dist_means.csv' % outdir
    dist_means_df = pd.concat(dist_means_df_lst)
    dist_means_df.to_csv(distmeans_filename, index=False)
    print 'Stored district means to %s!' % distmeans_filename

    return outdir


"""
point_est_main()
    Function to run all of the main code for the point_est.py file from the
    commandline.

    @param: state_lst (list of all states considered in data)
            year
            outdir (directory with inferred_vals.csv)
            num_clusters, precdist
"""
def point_est_main(state_lst, year, outdir, num_clusters, precdist,
        path_to_data):
    # pick files (prec, param)
    prec_df_lst, param_df = init(state_lst, year, outdir, num_clusters,
            path_to_data)
    print param_df
    print '-------------------------------------------------'

    # compute likelihood of each asst for all precs
    colnames = []
    first_prec = True

    final_prec_df_lst = []
    dist_means_df_lst = []

    for state, prec_df in prec_df_lst:
        for i in xrange(num_clusters):
            prec_df['%d' % i] = compute_asst_loglik(prec_df, param_df, i,
                    precdist)
            if first_prec:
                colnames.append('%d' % i)
        first_prec = False

        # mark asst based on largest likelihood
        select_asst(prec_df, colnames)
        prec_df['state'] = state
        final_prec_df_lst.append(prec_df)

        # some quick analysis
        print 'Frequency of Assignments'
        print (prec_df.groupby('asst').count())['0'] / len(prec_df.index)


        # compute weighted average of means per district
        dist_means_df_lst.append(weighted_means_bydist(prec_df, param_df,
            state))

    # write to file
    write_to_file(final_prec_df_lst, dist_means_df_lst, outdir)

    # polarization metrics
    pol_metrics_one_state(state, year, precdist, num_clusters, path_to_data,
        outdir)

    return outdir


"""
point_est_fake()
    Function to run all of the main code for the point_est.py file from the
    commandline for simulated datasets with slightly different file organization.

    @param: fake_dist (unimodal or bimodal)
            outdir (directory with inferred_vals.csv)
              eg: 'results/fake/4_clusters/extreme_{}/charts'
            num_clusters
            precdist
            path_to_data
"""
def point_est_fake(fake_dist, outdir, num_clusters, precdist, path_to_data):
    # pick files (prec, param)
    precfile = '{}/data/fake_data/{}_data.csv'.format(path_to_data, fake_dist)
    prec_df = pd.read_csv(precfile)
    # hard-coding all of the simulated precincts to be in the same district
    prec_df['congdist'] = 0

    param_df = None
    if os.path.exists('%s/inferred_param_df.csv' % outdir):
        param_df = pd.read_csv('%s/inferred_param_df.csv' % outdir)
    elif os.path.exists('%s/mle_inferred_vals.csv' % outdir):
        # mle_inferred_vals.csv format from data_results
        param_row_df = pd.read_csv('%s/mle_inferred_vals.csv' % outdir)
        param_df = arr_to_df(param_row_df.iloc[1], num_clusters)
        param_df.sort_values(by='loc', inplace=True)
        param_df.index = xrange(num_clusters)

        param_outfilename = '%s/inferred_param_df.csv' % outdir
        param_df.to_csv(param_outfilename, index=False)
        print 'Wrote converted param_df to: %s' % param_outfilename
    else:
        print 'File format not recognized. See init() function. ):'
        assert(False)
    print param_df
    print '-------------------------------------------------'

    # compute likelihood of each asst for all precs
    colnames = []

    for i in xrange(num_clusters):
        prec_df['%d' % i] = compute_asst_loglik(prec_df, param_df, i,
                precdist)
        colnames.append('%d' % i)

    # mark asst based on largest likelihood
    select_asst(prec_df, colnames)
    prec_df['state'] = None
    prec_df['geoid'] = prec_df.index

    # some quick analysis
    print 'Frequency of Assignments'
    print (prec_df.groupby('asst').count())['0'] / len(prec_df.index)

    # compute weighted average of means per district
    dist_means_df = weighted_means_bydist(prec_df, param_df, None)

    # write to file
    print prec_df.head()
    write_to_file([prec_df], [dist_means_df], outdir)

    # computing polarization metrics
    pol_metrics_fake(path_to_data, num_clusters, fake_dist, precdist, outdir)

    return outdir


def pol_metrics_fake(path_to_results, num_clust, fake_dist, prec_dist, outdir):
    # getting fake state_data
    ## for voters
    voter_dir = '%s/results/fake/%s_clusters/extreme_%s/charts' % (path_to_results,
            num_clust, fake_dist)

    prec_df = pd.read_csv('%s/prec_asst_cd.csv' % voter_dir).groupby('asst')
    clust_df = pd.DataFrame({
        'loc': prec_df['loc'].first(),
        'scale': prec_df['scale'].first(),
        'tot_votes': prec_df['tot_votes'].aggregate(np.sum)
    })

    param_df = pd.read_csv('%s/inferred_param_df.csv' % voter_dir)

    return pol_metrics_main(clust_df, param_df, None, prec_dist, 'fake',
        num_clust, -1, outdir)


def main():
    path_to_results = '../'

    # state
    state = raw_input('State: ')
    # prec_dist
    precdist = raw_input('Prec Dist: ')
    # num_cluster
    num_clusters = raw_input('Number of Clusters: ')

    if state != 'fake':
        state_lst = [state]

        # election year
        year = raw_input('Year: ')

        outdir = '%sresults/%s/%s_clusters/mh_%s/%s/charts' % (path_to_results,
                state, num_clusters, year, precdist)

        point_est_main(state_lst, year, outdir, int(num_clusters), precdist,
                '%sdata' % path_to_results)

    else:
        fake_dist = raw_input('Simulated dist (uni/bimodal): ')
        outdir = '{}results/fake/{}_clusters/extreme_{}/charts'.format(path_to_results,
            num_clusters, fake_dist)
        print outdir

        point_est_fake(fake_dist, outdir, int(num_clusters), precdist,
            path_to_results)



if __name__ == "__main__":
    main()

