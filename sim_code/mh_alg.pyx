"""
mh_alg.pyx

Metropolis-Hastings algorithm to infer parameters

To compile Cython code to run:
> python setup.py build_ext --inplace

Helpful link to debug installing Cython:
http://stackoverflow.com/questions/19378037/how-to-make-python-aware-of-the-cython-module-being-installed-in-a-location-othe
"""

import csv
import math
import numpy as np
import os
import pandas as pd

from datetime import datetime
from scipy.misc import logsumexp
from scipy.stats import beta, norm, invgamma, uniform, laplace
from scipy.special import gammaln as gamln


"""
init_files
    Function that prepares reading/writing to file input and output.

    @param: state, year, number of clusters
    @ret: data_df, val_writer, avect_writer
"""
def init_files(num_clusters, infile, path_to_sim, dir):
    data_df = pd.read_csv(infile)

    # setup output directory / files
    if path_to_sim == None:
        outdir = '{}'.format(dir)
    else:
        outdir = '{}/{}'.format(path_to_sim, dir)

    # new directory name (auto-increments)
    dir_iter = int(max(os.listdir(outdir))) + 1
    while True:
        try:
            tempdir = "{}/{}".format(outdir, dir_iter)
            os.makedirs(tempdir)
            break
        except OSError:
            print 'Outdir %d already exists, boo.' % dir_iter
            assert(False)
    outdir = tempdir

    # create csv files to write to
    theta_lst = []
    loc_lst = []
    scale_lst = []
    for i in xrange(num_clusters):
        theta_lst.append('theta%d' % i)
        loc_lst.append('loc%d' % i)
        scale_lst.append('scale%d' % i)

    outfiles = {
            'dir': outdir
    }

    p_file = open('%s/posterior_vals.csv' % outdir, 'w+')
    p_writer = csv.writer(p_file, delimiter = ',')
    p_writer.writerow(['posterior'])
    outfiles['posterior'] = (p_writer, p_file)

    t_file = open('%s/theta_vals.csv' % outdir, 'w+')
    t_writer = csv.writer(t_file, delimiter = ',')
    t_writer.writerow(theta_lst)
    outfiles['theta'] = (t_writer, t_file)

    l_file = open('%s/loc_vals.csv' % outdir, 'w+')
    l_writer = csv.writer(l_file, delimiter = ',')
    l_writer.writerow(loc_lst)
    outfiles['loc'] = (l_writer, l_file)

    s_file = open('%s/scale_vals.csv' % outdir, 'w+')
    s_writer = csv.writer(s_file, delimiter = ',')
    s_writer.writerow(scale_lst)
    outfiles['scale'] = (s_writer, s_file)

    readme_file = open('%s/README.txt' % outdir, 'w+')
    readme_file.write(infile)
    readme_file.write('\nnumber of clusters: %d\n' % num_clusters)
    outfiles['readme'] = (None, readme_file)

    return data_df, outfiles


"""
init_vars
    Function that sets up the dictionary data structure that will hold all of
    the most up-to-date parameter values.

    NOTE: currently, each parameter can be initialized either randomly, using
    the specificed random distributions in get_rand_val() or by fixed values
    that were determined in the fake_data.py code to generate the simulated
    dataset (and are currently hard-coded below)

    @param: dataframe of precincts, number of clusters, number of prec,
            prec_dist - type of underlying prec distribution

    @ret: param_df
"""
def init_vars(num_clusters, prec_dist):
    theta_vals = get_rand_val('theta', num_clusters=num_clusters)

    loc_vals = []
    scale_vals = []
    for i in xrange(num_clusters):
        loc_vals.append(get_rand_val('loc', dist=prec_dist))
        scale_vals.append(get_rand_val('scale', dist=prec_dist))

    param_df = pd.DataFrame({
        'theta': theta_vals,
        'loc': loc_vals,
        'scale': scale_vals
    })
    return param_df


"""
get_rand_val
    Function that returns a random value for each variable based on the
    hyperparmeters set for each prior distribution.

    @param: datatype - string of what kind of param we need an rvs for
            [if 'theta' type] also need num_clusters int datatype
            dist - what type of dist we assume precinct dist follows
                   (eg: uniform, normal, laplace, mixture of two symmetric
                   normals)
"""
def get_rand_val(datatype, num_clusters=None, dist=None):
    if datatype == 'theta':
        assert(type(num_clusters) == int)
        return np.random.dirichlet([1] * num_clusters)

    if dist == 'uniform':
        if datatype == 'loc':
            return uniform.rvs(-10, 10)
        elif datatype == 'scale':
            return uniform.rvs(12, 8)
    elif dist == 'norm' or dist == 'laplace' or dist == 'mixnorm':
        if datatype == 'loc':
            return norm.rvs(0, 10)
        elif datatype == 'scale':
            return uniform.rvs(0, 10)

    # fallback
    print('Oops, cannot return a random value. See get_rand_val().')
    assert(False)


"""
prior_logpdf
    Function that computes the log pdf of a value given the specific prior
    distribution we have arbitrarily set for each type of parameter.

    @param: datatype, value
"""
def prior_logpdf(datatype, x, prec_dist):
    if datatype == 'loc':
        return(norm.logpdf(x, 0, 100))
    elif datatype == 'scale':
        if prec_dist == 'uniform':
            return(norm.logpdf(x, 0, 100))
        elif prec_dist == 'norm' or prec_dist == 'laplace' or \
                prec_dist == 'mixnorm':
            if x <= 0:
                # outside of the support of the inverse gamma dist
                return -np.inf
            return(invgamma.logpdf(x ** 2, 3, loc=0, scale=1))
        else:
            print 'Oops, cannot find a corresponding prior. See prior_logpdf().'
            assert(False)
    else:
        print('Oops, cannot return a pdf value. See prior_logpdf().')
        assert(False)


"""
prior_vect_logpdf
    Function that computes the log pdf of a value given the specific prior
    distribution we have arbitrarily set for each type of parameter.

    @param: datatype, value (as a pandas series!!)
"""
def prior_vect_logpdf(datatype, vect, prec_dist):
    if datatype == 'loc':
        return(norm.logpdf(vect, 0, 100))
    elif datatype == 'scale':
        if prec_dist == 'uniform':
            return(norm.logpdf(vect, 0, 100))
        elif prec_dist == 'norm' or prec_dist == 'laplace' or \
                prec_dist == 'mixnorm':
            var_vect = vect ** 2
            prior_vect = invgamma.logpdf(var_vect, 1, loc=0, scale=1)

            # outside of the support of the inverse gamma dist
            prior_vect[np.isnan(prior_vect)] = -np.inf

            return prior_vect
        else:
            print('Oops, no prior distribution defined. See prior_vect_logpdf().')
            assert(False)
    else:
        print('Oops, cannot return a pdf value. See prior_vect_logpdf().')
        assert(False)


"""
prop_dist_val
    Function that returns a new value based on the datatype and the pre-set
    proposal distributions around the previous value.

    @param: datatype, orig value
"""
def prop_dist_val(datatype, old_val, num_clusters=0):
    if datatype == 'theta':
        assert(type(old_val) == pd.core.series.Series)
        new_val = np.random.dirichlet(10 * old_val + 1)
        return new_val
    elif datatype == 'loc':
        pwidth = 1
    elif datatype == 'scale':
        pwidth = 1
    else:
        print('Oops, cannot return a new proposed value. See prop_dist_val().')
        assert(False)

    # uniform dist pwidth amount on either side
    return(uniform.rvs(old_val - pwidth, 2 * pwidth))


"""
use_next_val
    Function that selects whether to take the new or old value based on the
    computed posterior distribution ratio.

    @return: boolean value on whether to use new value
"""
cpdef bint use_next_val(double acc_ratio_numer, double acc_ratio_denom,
        int iter_num=0):
    # should only ever have accepted samples with nonzero likelihood
    assert(not(np.isnan(acc_ratio_denom)))
    if np.isnan(acc_ratio_numer):
        return False
    else:
        post_ratio = acc_ratio_numer - acc_ratio_denom

    # tempering step
    cdef int alpha = 1 # min(iter_num / 1000.0, 1)

    # r := acceptance probability
    cdef double r = post_ratio * alpha
    if np.log(uniform.rvs(0, 1)) < r:
        return True
    else:
        return False


"""
ldirichlet_pdf
    Function to compute the logpdf of the Dirichlet distribution. See
    notes/theta_update.pdf

    @param: alpha = vector parameter of the Dirichlet distribution
            theta = vector we are considering its pdf relative to this dist
"""
def ldirichlet_pdf(theta, alpha):
    try:
        kernel = sum((a - 1) * math.log(t) for a, t in zip(alpha, theta))
        lbeta_alpha = sum(math.lgamma(a) for a in alpha) - math.lgamma(sum(alpha))

        return kernel - lbeta_alpha
    except ValueError:
        return -np.Inf


"""
update_theta
    Function that updates one theta value at a time.

    @param: param_df, data_df, num_clusters, iter_num
"""
cpdef update_theta(param_df, data_df, int num_clusters, int iter_num, prec_dist):
    theta_old_vect = param_df['theta']
    theta_new_vect = prop_dist_val('theta', theta_old_vect, num_clusters)

    # prior + loglik = 0(no theta prior) + f(data | Z, loc_new, scale_old)
    cdef double acc_prob_denom = loglik(data_df, param_df, prec_dist) +\
            ldirichlet_pdf(theta_new_vect, 100 * theta_old_vect)

    param_df['theta'] = theta_new_vect
    cdef double acc_prob_numer = loglik(data_df, param_df, prec_dist) +\
            ldirichlet_pdf(theta_old_vect, 100 * theta_new_vect)

    if use_next_val(acc_prob_numer, acc_prob_denom, iter_num):
        return
    else:
        param_df['theta'] = theta_old_vect
        return


"""
loglik
    Function that computes the log-likelihood of the data given all of the
    inferred parameter values.

    @param: data_df, param_df
"""
def loglik(data_df, param_df, prec_dist):
    num_clusters = len(param_df.values)

    # list that will hold lists of logprob with each cluster values to be used
    # for the logsumexp computation later on
    cluster_logprob = []
    for i in xrange(num_clusters):
        logcdf = 0
        logsf = 0
        loc = param_df.loc[i, 'loc']
        scale = param_df.loc[i, 'scale']

        if prec_dist == 'uniform':
            logcdf = uniform.logcdf(data_df['midpoint'], loc, scale)
            logsf = uniform.logsf(data_df['midpoint'], loc, scale)
        elif prec_dist == 'norm':
            logcdf = norm.logcdf(data_df['midpoint'], loc, scale)
            logsf = norm.logsf(data_df['midpoint'], loc, scale)
        elif prec_dist == 'laplace':
            logcdf = laplace.logcdf(data_df['midpoint'], loc, scale)
            logsf = laplace.logsf(data_df['midpoint'], loc, scale)
        elif prec_dist == 'mixnorm':
            logcdf = np.log(0.5) + logsumexp([
                norm.logcdf(data_df['midpoint'], loc, scale),
                norm.logcdf(data_df['midpoint'], -loc, scale)
                ])
            logsf = np.log(0.5) + logsumexp([
                norm.logsf(data_df['midpoint'], loc, scale),
                norm.logsf(data_df['midpoint'], -loc, scale)
                ])
        else:
            print 'Cannot find logcdf given precinct distribution, see loglik().'
            assert(False)

        logp_succ = data_df['num_votes_0'] * logcdf
        logp_fail = data_df['num_votes_1'] * logsf
        logtheta = np.log(param_df.get_value(i, 'theta'))

        # cluster_logprob.append(np.exp(logp_succ + logp_fail + logtheta))
        cluster_logprob.append(logp_succ + logp_fail + logtheta)

    # logsumexp_temp = np.log(np.sum(cluster_logprob, axis=0))
    logsumexp_temp = logsumexp(cluster_logprob, axis=0)
    return np.sum(logsumexp_temp)


"""
posterior
    Function that computes the log-posterior of the data given all of the
    inferred parameter values.

    @param: data_df, param_df
"""
def compute_logposterior(data_df, param_df, prec_dist):
    return sum(prior_vect_logpdf('loc', param_df['loc'], prec_dist)) + \
            sum(prior_vect_logpdf('scale', param_df['scale'], prec_dist)) + \
            loglik(data_df, param_df, prec_dist)


"""
update_loc_scale
    Function that updates the loc and scale value for one cluster.

    @param: cluster, param_df, data_iternum
"""
cpdef update_loc_scale(int cluster, param_df, data_df, int iter_num, prec_dist):
    cdef double loc_old = param_df.loc[cluster, 'loc']
    cdef double loc_new = prop_dist_val('loc', loc_old)

    cdef double scale_old = param_df.loc[cluster, 'scale']
    cdef double scale_new = prop_dist_val('scale', scale_old)

    cdef double posterior_old = prior_logpdf('loc', loc_old, prec_dist) +\
            prior_logpdf('scale', scale_old, prec_dist) +\
            loglik(data_df, param_df, prec_dist)

    param_df.loc[cluster, 'loc'] = loc_new
    param_df.loc[cluster, 'scale'] = scale_new
    cdef double posterior_new = prior_logpdf('loc', loc_new, prec_dist) +\
            prior_logpdf('scale', scale_new, prec_dist) +\
            loglik(data_df, param_df, prec_dist)

    if use_next_val(posterior_new, posterior_old, iter_num):
        assert(scale_new > 0)
        return
    else:
        # revert to old value
        param_df.loc[cluster, 'loc'] = loc_old
        param_df.loc[cluster, 'scale'] = scale_old
        return


"""
run_mh_chain

    Function that is the backbone of the entire MH algorithm for my code.

    @param: num_iter
            num_clusters
            datafile - precline_ file, etc
            prec_dist - underlying precinct distribution, but same names as the
                        scipy.stats names ['norm', 'uniform', 'laplace', 'triang']
"""
def run_mh_chain(num_iter, num_clusters, datafile, prec_dist, year,
        results_dir=None, path_to_sim=None):
    starttime = datetime.now()

    outdir = results_dir if results_dir != None else '{}c_{}_{}'.format(
        num_clusters, prec_dist, year)

    data_df, outfiles = init_files(num_clusters, datafile, path_to_sim, outdir)

    # pdict := dictionary holding updated parameter values
    cdef int num_precs = len(data_df.index)

    outfiles['readme'][1].write('Number of Precs: %d\n' % num_precs)
    outfiles['readme'][1].write('Prec Distribution: %s\n\n' % prec_dist)

    param_df = init_vars(num_clusters, prec_dist)

    clust_arr = xrange(num_clusters)

    max_post_val = -np.inf

    for i in xrange(num_iter):
        # compute log likelihood
        posterior_val = compute_logposterior(data_df, param_df, prec_dist)

        if posterior_val > max_post_val:
            max_post_val = posterior_val

        if i % 100 == 0:
            # print new values to file every 100 iterations
            outfiles['posterior'][0].writerow([posterior_val])
            outfiles['theta'][0].writerow(param_df['theta'])
            outfiles['loc'][0].writerow(param_df['loc'])
            outfiles['scale'][0].writerow(param_df['scale'])

        # updating the values
        update_theta(param_df, data_df, num_clusters, i, prec_dist)
        map(lambda x: update_loc_scale(x, param_df, data_df, i, prec_dist),
                clust_arr)

        if i % 1000 == 0:
            outfiles['readme'][1].write('num_iter: %d\n' % i)
            outfiles['readme'][1].write('time elapsed: %s\n' %
                    (datetime.now() - starttime))


    outfiles['posterior'][1].close()
    outfiles['theta'][1].close()
    outfiles['loc'][1].close()
    outfiles['scale'][1].close()


    outfiles['readme'][1].write('time elapsed: %s\n' % (datetime.now() -
        starttime))
    outfiles['readme'][1].close()

    # returning where the files are all located
    return outfiles['dir']

