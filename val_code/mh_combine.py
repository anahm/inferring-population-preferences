"""
mh_combine.py

Creates data tables from the MH data results from mh_alg.py.
"""

import csv
import datetime
import numpy as np
import os
import pandas as pd
import shutil


"""
store_param_rows()
    Function that writes all of the values of each parameter that are located
    at the same row/MH chain index into an output csv file.

    @param: dictionaries of all the dataframes of parameter values

    NOTE: creates a new file if one does not exist or appends
"""
def store_param_rows(theta_df_dict, loc_df_dict, scale_df_dict, dirName,
        filename, index, subdir_only=-1):
    outfilename = '%s/%s.csv' % (dirName, filename)

    header_written = os.path.isfile(outfilename)

    outfile = open(outfilename, 'a')
    outcsv = csv.writer(outfile, delimiter=',')

    for subdir in theta_df_dict.keys():
        if subdir_only != -1 and subdir != subdir_only:
            continue
        theta_series = theta_df_dict[subdir].iloc[index]
        loc_series = loc_df_dict[subdir].iloc[index]
        scale_series = scale_df_dict[subdir].iloc[index]

        param_arr = pd.concat([theta_series, loc_series, scale_series])
        if not header_written:
            # print the header
            outcsv.writerow(np.insert(param_arr.keys(), 0, ['index', 'subdir']))
            header_written = True
        outcsv.writerow(np.insert(param_arr.values, 0, [index, subdir]))

    outfile.close()


"""
make_param_header()
    Function that converts the param_arr keys (all numbers of the associated
    cluster) into a more user-friendly header (0, 1, 2 --> theta0, theta1, etc.)
    and also adds 'index' and 'subdir' column header names.
"""
def make_param_header(keys_arr, num_clusters):
    param_arr = ['theta', 'loc', 'scale']
    param_arr_i = -1

    ret_arr = ['index', 'subdir']

    num_keys = len(keys_arr)
    for i in xrange(num_keys):
        if keys_arr[i] == 'posterior':
            ret_arr.append(keys_arr[i])
            continue
        elif i % num_clusters == 0:
            param_arr_i += 1

        ret_arr.append('%s%s' % (param_arr[param_arr_i], i % num_clusters))

    return ret_arr


"""
    Function that infers parameters of model given multiple chains of data with
    random starting points. Parameters are selected as the set with the largest
    log-likelihood value.

    @param: df_dict - dictionary with all of the value dataframes and posterior
                      dataframes
    @ret: file of parameter values (outfilename)
"""
def mle_inference(post_df_dict, theta_df_dict, loc_df_dict, scale_df_dict,
        dirName, num_clusters):
    max_post = -np.inf
    max_post_key = -1
    max_post_index = -1

    # also want to output results of all potential mle candidates
    postfilename = 'highest_post_vals'

    for key, post_df in post_df_dict.iteritems():
        temp_post = (post_df.max()).iloc[0]
        temp_post_index = post_df.idxmax().iloc[0]
        store_param_rows(theta_df_dict, loc_df_dict, scale_df_dict, dirName,
                postfilename, temp_post_index, key)
        print temp_post

        if temp_post > max_post:
            max_post = temp_post
            max_post_key = key
            max_post_index = temp_post_index

    # outputting results
    outfilename = '%s/mle_inferred_vals.csv' % dirName
    outfile = open(outfilename, 'w+')
    outcsv = csv.writer(outfile, delimiter=',')


    for i in [0, max_post_index]:
        df = pd.DataFrame({
            'theta': (theta_df_dict[max_post_key].iloc[i]).values,
            'loc': (loc_df_dict[max_post_key].iloc[i]).values,
            'scale': (scale_df_dict[max_post_key].iloc[i]).values
        })
        post_series = pd.Series(post_df_dict[max_post_key].iloc[i])
        df.sort_values(by='loc', inplace=True)
        param_arr = pd.concat([df['theta'], df['loc'], df['scale'], post_series])
        if i == 0:
            # print the header
            outcsv.writerow(make_param_header(param_arr.keys(), num_clusters))

        outcsv.writerow(np.insert(param_arr.values, 0, [i, max_post_key]))

    outfile.close()
    return outfilename


"""
setup
    Function that creates the output directory for this script.
"""
def setup(input_dir):
    data_dir = '%s/charts' % input_dir

    # note: replaces old charts dir
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.mkdir(data_dir)

    # using subtitle for the readme file now
    f = open('%s/README.txt' % data_dir, 'w+')
    f.close()

    return data_dir


"""
examine_subdir()
    Function to examine the sub-directory with the results of one MH chain.

    @param: subdir_path, theta_df_dict, loc_df_dict, scale_df_dict,
            posterior_df_dict
"""
def examine_subdir(subdir_path, subdir, theta_df_dict, loc_df_dict, scale_df_dict,
        posterior_df_dict):

    # checking valid subdir
    if not os.path.exists('%s/theta_vals.csv' % subdir_path):
        print('skipping: ' + subdir)
        return

    theta_df = pd.read_csv('%s/theta_vals.csv' % subdir_path)
    theta_df_dict[subdir] = theta_df

    loc_df = pd.read_csv('%s/loc_vals.csv' % subdir_path)
    loc_df_dict[subdir] = loc_df

    scale_df = pd.read_csv('%s/scale_vals.csv' % subdir_path)
    scale_df_dict[subdir] = scale_df

    posterior_df = pd.read_csv('%s/posterior_vals.csv' % subdir_path)
    posterior_df_dict[subdir] = posterior_df


"""
mh_combine_one()
    Function to call the main function of mh_combine from the commandline when
    there is only one directory (used for testing).

    @param: orig_dir (and subdir), num_clusters
    @return: location of param_file
"""
def mh_combine_one(orig_dir, num_clusters):
    assert(num_clusters != None)

    dirName = setup(orig_dir)
    theta_df_dict = {}
    loc_df_dict = {}
    scale_df_dict = {}
    posterior_df_dict = {}

    dirname = orig_dir[-1]
    examine_subdir(orig_dir, dirname, theta_df_dict, loc_df_dict, scale_df_dict,
            posterior_df_dict)

    store_param_rows(theta_df_dict, loc_df_dict, scale_df_dict, dirName,
            'starting_points', 0)
    store_param_rows(theta_df_dict, loc_df_dict, scale_df_dict, dirName,
            'final_points', -1)

    param_file = mle_inference(posterior_df_dict, theta_df_dict, loc_df_dict,
            scale_df_dict, dirName, num_clusters)

    print 'mle_inferred_vals.csv written to %s' % (param_file)
    return dirName


"""
mh_combine_main()
    Function to call the main function of mh_combine from the commandline.

    @param: orig_dir, num_clusters
    @return: location of param_file
"""
def mh_combine_main(orig_dir, num_clusters):
    assert(num_clusters != None)

    dirName = setup(orig_dir)
    theta_df_dict = {}
    loc_df_dict = {}
    scale_df_dict = {}
    posterior_df_dict = {}

    # go through all subdirs
    for subdir in os.listdir(orig_dir):
        subdir_path = '%s/%s' % (orig_dir, subdir)
        examine_subdir(subdir_path, subdir, theta_df_dict, loc_df_dict,
                scale_df_dict, posterior_df_dict)

    store_param_rows(theta_df_dict, loc_df_dict, scale_df_dict, dirName,
            'starting_points', 0)
    store_param_rows(theta_df_dict, loc_df_dict, scale_df_dict, dirName,
            'final_points', -1)

    param_file = mle_inference(posterior_df_dict, theta_df_dict, loc_df_dict,
            scale_df_dict, dirName, num_clusters)

    print 'mle_inferred_vals.csv written to %s' % param_file
    return dirName


def main():
    state = raw_input('State: ')
    num_clust = raw_input('Number of Clusters: ')

    # directory containing info about inferred param (distmeans, asstvect, etc.)
    orig_dir = None

    if state == 'fake':
        fake_type = raw_input('Fake type (unimodal or bimodal): ')
        orig_dir = '../results/fake/{}_clusters/extreme_{}'.format(num_clust,
            fake_type)
    else:
        prec_dist = raw_input('Prec Dist: ')
        year = raw_input('Year: ')
        orig_dir = '../results/%s/%s_clusters/mh_%s/%s' % (state, num_clust, 
            year, prec_dist)

    mh_combine_main(orig_dir, int(num_clust))


if __name__ == "__main__":
    main()
