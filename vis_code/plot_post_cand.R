# plot_post_cand.R
#
# Plots the posterior predictive check based on a set of inferred parameter
# values.

require(ggplot2)
require(VGAM)

# NOTE: be sure that the inferred_param_df.csv is the most up-to-date!!

# Parameters that need to be set
state = 'oh'
year = '2008'
num_clust = '8'
precdist = 'norm'

# Outdir of the plot
outdir = paste('../results/', state, '/', num_clust, '_clusters/mh_', year, '/', precdist, '/charts', sep='')
# Candidate file to construct histograms
candfile = paste('../data/', state, '_data/data/', state, '_', year, '/', state, '_cand_', year, '.csv', sep='')

###########

inferred.dist = function(chart.dir, election.str, cand.file, precdist) {
  # NOTE: param.df assumes form of "inferred_param_df.csv"
  param.df <- read.csv(paste(chart.dir, 'inferred_param_df.csv', sep='/'))
  print(head(param.df))
  cand.df <- read.csv(cand.file)
  cand.df$x.vals <- as.numeric(cand.df$cf_score)
  print(head(cand.df))

  xlim = c(-10,10)
  x.vals = seq(xlim[1], xlim[2], length=1000)

  df <- data.frame(x.vals)
  for(i in 1:nrow(param.df)) {
    theta = param.df[i, 'theta']
    mu = param.df[i, 'loc']
    sigma = param.df[i, 'scale']
    if (precdist == 'norm') {
      df[paste('val', i, sep='')] = theta * dnorm(x.vals, mu, sigma)
    } else if (precdist == 'laplace') {
      df[paste('val', i, sep='')] = theta * dlaplace(x.vals, mu, sigma)
    } else {
      print('Error in precdist stuff.')
    }
  }
  df$y.vals = rowSums(df) - df$x.vals
  color_arr = c('black', 'red', 'green', 'blue', '#388E8E')
  values_arr = c('blargl'='black', '1'='red', '2'='green', '3'='blue', '4'='#388E8E',
                 '5'='orange', '6'='yellow', '7'='pink', '8'='brown')

  gg.var = ggplot(df, aes(x=x.vals, y=y.vals)) +
    geom_line() + # data=df, aes(x=x.vals, y=y.vals)) +
    geom_histogram(data=cand.df, aes(x=x.vals, y=..count../sum(..count..)), binwidth=0.2) +
    # scale_colour_manual('Cluster', values=color_arr) +
    xlab('Political Position') + ylab('Density') +
    ggtitle(paste('Distributions of Candidate vs. Voter Preferences (', election.str, ')', sep='')) +
    theme(panel.background = element_blank(), axis.line = element_line(colour = "black"),
          text = element_text(family = "Times", size=18))
  print(gg.var)
  ggsave(paste(chart.dir, 'cand_posterior_plot.jpg', sep='/'),width=8, height=4)
}

inferred.dist(outdir, paste(toupper(state), year), candfile, precdist)

