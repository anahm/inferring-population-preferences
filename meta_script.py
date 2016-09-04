"""
meta_script.py

Giant script to run through every piece of important code to re-generate the
results presented in our paper.

"""

import sys

sys.path.insert(0, 'shp_code')
from prec_reformat import prec_reformat_main
from oh_prep import oh_load_main

sys.path.insert(0, 'sim_code')
from mh_alg import run_mh_chain

sys.path.insert(0, 'val_code')
from mh_combine import mh_combine_one
from point_est import point_est_main
from compare_scores import compare_scores_main


""" Globals (change as needed) """
""" Params of election data source for inference """
state = 'tx'
year = '2006'
num_clusters = 4
prec_dist = 'norm'

""" Params of MH algorithm """
num_chains = 4
num_iter = 10000

""" Additional params """
combined_dir = "." # Directory for results of inferences of multiple states


""" Pre-processing the data files. """
# Summary: original data files --> "precline" file for inference
# data/<state>_data/orig_data/ includes:
    # shapefile/ - state shapefile (from http://cdmaps.polisci.ucla.edu/)
    # *_precvote.csv - precinct-level vote share (from Harvard Dataverse)
    # *_cand_byhand.csv - top dem/rep of each election (from NYTimes)

input_fname = ''
if state == 'tx' or state == 'ny':
  """
  shp_code/prec_reformat.py
    @param: state, year
    @ret: precline_* data file, eg: 'data/tx_data/tx_2006/precline_tx_house_2006.csv'
    test: shp_code/check_data.py
  """
  input_fname = prec_reformat_main(state, year)
elsif state == 'oh':
  """
  # shp_code/oh_prep.py
    # @param:  state, year
    # @ret: "precline" input data file
  """
  input_fname = oh_prep_main(state, year, False)


""" Metropolis-Hastings (MH) infernce. """
# Summary: running MH inference on the processed files
# Readable code in sim_code/mh_alg.pyx, but must run compile Cython code into
# mh_alg.so with the command 'python setup.py build_ext --inplace' to run below.
# Warning: this section of the code can take a long time to run depending on the
#          inference procedure parameters set (num_chains, num_iter)

# Preparing output directory
results_dir = '{}_{}c_{}_{}'.format(state, num_clusters, prec_dist, year)
if os.path.exists(results_dir):
  shutil.rmtree(results_dir)
os.mkdir(results_dir)
os.mkdir('{}/0'.format(results_dir))

# Running mh given num_chains / num_iter parameters
for x in xrange(num_chains):
  """
  sim_code/mh_alg.pyx
    # @param: num_iter, num_clust, input_fname must be precline_* file,
    #         prec_dist, results_dir
    # @ret: directory containing inferences (param_vals.csv, etc)
  """
  one_chain_dir = run_mh_chain(num_iter, num_clusters, input_fname, prec_dist,
      results_dir, 'sim_code')
  print 'Chain {} out of {} complete, output: {}'.format(x, num_chains,
      one_chain_dir)

print 'Inference results located in: {}'.format(results_dir)


""" Processing inferences. """
# Summary: Processing inference results of individual chains to be used for
#          validation and polarization application area
# After this section, <results_dir>/charts will include:
    # inferred_param_df.csv - Inferred parameters of all MH chains
    # prec_asst_cd.csv - Cluster assignment and cong. district of each precinct
    # dist_means.csv - Aggregated district-level estimate of preferences

"""
# val_code/mh_combine.py
# Find parameters of all MH chains derived from same data with highest posterior
    @param: directory with MH results (results_dir), num_clusters
    @ret: directory path with final inferred parameters (mle_inferred_vals.csv)
    Note: Also store information about each chain used to compute final set of
          parameters (eg: final_points.csv, highest_post_vals.csv,
          starting_points.csv)
"""
inferred_dir = mh_combine_one(results_dir, num_clusters)

"""
# val_code/point_est.py()
# Create point estimates for comparison (eg: cluster assignments for each
# precinct, aggregate district-level preferences, polarization metrics)
    @param: state, year, mh_combined results, num_clusters, prec_dist,
            path_to_data
    @ret: directory path with inferred assignment vector (prec_asst_cd.csv),
          aggregated district means (dist_means.csv), polarization metrics
          (pol_metrics.csv)

    test: frequency of most likely asst for precincts ~ inferred theta value
"""
point_est_main([state], year, inferred_dir, num_clusters, prec_dist, '.')
print 'Processed results stored in: {}'.format(inferred_dir)


""" Running validation tests. """
# Summary: Running validation tests comparing our results against related works
# After this section, <results_dir>/comparison_charts will include:
    # dist_comp.csv - District-level comparison to MRP estimates and CCES surveys
    # pred_error.csv - Standard error of each prediction task
    # pred_comp.csv - Predictions of next-cycle election by various tasks
    # alt_pred_comp.csv - Predictions of alternative election of same cycle,
    #                     again by various tasks

"""
# val_code/compare_scores.py
# Compare district-level estimates and run prediction validation tasks
    @param: state, year, inferred_dir, compare_dir, path to validation code
    @ret: dictionary of file paths to dist_comp.csv (dist), pred_comp.csv
    (pred), alt_pred_comp.csv (alt-pred), and pred_error.csv (error)
"""
compare_file_dict = compare_scores_main(state, year, inferred_dir, results_dir,
    'val_code')

"""
# val_code/val_plot.R
# Compare correlation between our results and related work
    @param: num_clusters, prec_dist, plots_dir (must update in the file)
    @ret: n/a - plots saved to specified directory
    Note: need to read script output for correlation and p-value statistics
"""
subprocess.call("/usr/bin/Rscript val_code/val_plot.r")


""" Running visualization code. """
# Summary: Code for the plots displayed in the paper

"""
# vis_code/plot_post_cand.R
# R visualization of overall posterior and candidate distribution
    @param: state, year, num_clust, prec_dist, outdir (must update in the file)
    @ret: n/a - plots saved to specified directory
"""
subprocess.call("/usr/bin/Rscript vis_code/plot_post_cand.r")

"""
# vis_code/pred_log_plot.R
# R visualization of prediction error
  # @param: prediction error csv (created by compare_scores.py or state_est.py)
  # @ret: n/a - plots saved to specified directory
"""
subprocess.call("/usr/bin/Rscript vis_code/plot_post_cand.r")


""" Validation of inferences of all states """
# Summary: Joining separate inferences made with data of each state for more
#          holistic comparison
# Note: Update combined_dir parameter set in the beginning to set where the
#       output files should be stored
# After this section, <combined_dir>/ will include:
    # combined_state_pred.csv - Standard error of each prediction task for
    #                           elections in multiple states

"""
# val_code/state_est.py
# Computing prediction error of inferences for all districts in all states
    @param: year, prec_dist, num_clust, path_to_results, outdir
    @ret: csv file path with prediction error terms
"""
combined_pred_file = state_est_main(year, prec_dist, num_clust, path_to_results,
    combined_dir)

# val_code/val_plot.R (see above for informtion)

