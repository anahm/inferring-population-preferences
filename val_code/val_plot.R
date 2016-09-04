# val_plot.py
#
# Code that plots scatterplots and computes correlations between our results
# and related works.

require(plyr)
require(ggplot2)

# Parameters that need to be set
num_clust = '8'
prec_dist = 'norm'
outdir = '.' # location of plots to be stored

# need to combine data for plotting tx, ny + 2006, 2008, 2010
read.comp.df = function(num_clust, prec_dist, state, year) {
  comp.dir = paste('../results/', state, '/', num_clust, '_clusters/mh_',
                    year, '/', prec_dist, '/comparison_charts', sep='')

  dist.df <- read.csv(paste(comp.dir, 'dist_comp.csv', sep='/'), header=T)
  dist.df$year <- year

  return(dist.df)
}



plot.comp = function(data, x, y, main, xlab, ylab, outfile, noYear) {
  lin.model = lm(paste(y, x, sep='~'), data=data)
  coef = coef(lin.model)

  if (noYear) {
    gg.var = ggplot(data, aes_string(x=x, y=y, colour='state')) +
              scale_colour_manual('State',
                      labels=c('TX', 'NY', 'OH'),
                      values=c('#762a83', '#1b7837', '#fc8d59'))
  } else {
    gg.var = ggplot(data, aes_string(x=x, y=y, colour='state', shape='year')) +
      scale_shape_manual('Cycle',
                      labels=c('2006', '2008', '2010'),
                      values=c(15, 19, 17)) +
      scale_colour_manual('State',
                      labels=c('TX', 'NY', 'OH'),
                      values=c('#762a83', '#1b7837', '#fc8d59'))
  }
  gg.var = gg.var +
    geom_point(size=9) +
    xlab(xlab) + ylab(ylab) +
    theme(panel.background = element_blank(),
      axis.line = element_line(colour = "black"),
      axis.line.x = element_line(color="black", size = 0.5),
      axis.line.y = element_line(color="black", size = 0.5),
      panel.grid.minor = element_line(colour = "grey95", size = 0.25),
      text = element_text(family = "Times", size=56),
      legend.key = element_rect(size = 1),
      legend.key.size = unit(3, 'lines'),
      axis.title.x=element_text(margin=margin(20,0,20,0)),
      axis.title.y=element_text(margin=margin(0,20,0,10))
    )

  ggsave(outfile, plot=gg.var, width=13, height=10.5)
  print(gg.var)
  return(lin.model)
}


###### plotting CCES

tx.2006.df <- read.comp.df(num_clust, prec_dist, 'tx', '2006')
tx.2008.df <- read.comp.df(num_clust, prec_dist, 'tx', '2008')
final.df <- rbind(tx.2006.df, tx.2008.df)

ny.2006.df <- read.comp.df(num_clust, prec_dist, 'ny', '2006')
final.df <- rbind(final.df, ny.2006.df)
ny.2008.df <- read.comp.df(num_clust, prec_dist, 'ny', '2008')
final.df <- rbind(final.df, ny.2008.df)

oh.2006.df <- read.comp.df(num_clust, prec_dist, 'oh', '2006')
final.df <- rbind(final.df, oh.2006.df)
oh.2008.df <- read.comp.df(num_clust, prec_dist, 'oh', '2008')
final.df <- rbind(final.df, oh.2008.df)

# Comparing with CCES Continuous Ideology (2006 and 2008 only)
main = paste('Comparison of CCES Response and District-Level Estimate', sep='')
xlab = 'Weighted Mean District Estimate (log)'
ylab = 'Ideology Self-Placement'
outfile = paste(outdir, '/', num_clust, 'c_', prec_dist, '_cces_self_p_filtered.pdf', sep='')

# scaling
final.df$scaled_mean = sign(final.df$blargl_weighted_mean) * log( abs(final.df$blargl_weighted_mean) + 1 )
lin.model = plot.comp(final.df, 'scaled_mean', 'self_p', main, xlab, ylab, outfile,
                      FALSE)
summary(lin.model)


# Adding in 2010 data for next comparison
tx.2010.df <- read.comp.df(num_clust, prec_dist, 'tx', '2010')
final.df <- rbind(final.df, tx.2010.df)

ny.2010.df <- read.comp.df(num_clust, prec_dist, 'ny', '2010')
final.df <- rbind(final.df, ny.2010.df)

oh.2010.df <- read.comp.df(num_clust, prec_dist, 'oh', '2010')
final.df <- rbind(final.df, oh.2010.df)


# Comparing with CCES Discrete Ideology
main = paste('Comparison of CCES Response and District-Level Estimate', sep='')
xlab = 'Weighted Mean District Estimate (log)'
ylab = 'Ideology Score (1-6)'
outfile = paste(outdir, '/', num_clust, 'c_', prec_dist, '_cces_ideo.pdf', sep='')

# scaling
final.df$scaled_mean = sign(final.df$blargl_weighted_mean) * log( abs(final.df$blargl_weighted_mean) + 1 )
lin.model = plot.comp(final.df, 'scaled_mean', 'ideo', main, xlab, ylab, outfile,
                      FALSE)
summary(lin.model)


###### plotting CW MRP results
# need to aggregate the inferred results in the same decade
read.mrp.df = function(state, num_clust, prec_dist, year) {
  comp.dir = paste('../results/', state, '/', num_clust, '_clusters/mh_',
                        year, '/', prec_dist, '/comparison_charts', sep='')
  tmp.df <- read.csv(paste(comp.dir, 'dist_comp.csv', sep='/'), header=T)
  dist.df <- tmp.df[2:3]
  dist.df$mrp_estimate <- tmp.df$mrp_estimate
  dist.df$state <- state
  return(dist.df)
}

dist.2006.df = read.mrp.df('tx', num_clust, prec_dist, '2006')
dist.2008.df = read.mrp.df('tx', num_clust, prec_dist, '2008')
dist.2010.df = read.mrp.df('tx', num_clust, prec_dist, '2010')
tx.tmp.df <- join(dist.2006.df, dist.2008.df, by='congdist', type='inner')
tx.tmp.df <- join(tx.tmp.df, dist.2010.df, by='congdist', type='inner')
tx.agg.df = data.frame(
    overall_mean = (rowSums(tx.tmp.df[, c(1, 5, 8)])) / 3,
    state = tx.tmp.df$state,
    mrp_estimate = tx.tmp.df$mrp_estimate
  )

dist.2006.df = read.mrp.df('ny', num_clust, prec_dist, '2006')
dist.2008.df = read.mrp.df('ny', num_clust, prec_dist, '2008')
dist.2010.df = read.mrp.df('ny', num_clust, prec_dist, '2010')
ny.tmp.df <- join(dist.2006.df, dist.2008.df, by='congdist', type='inner')
ny.tmp.df <- join(ny.tmp.df, dist.2010.df, by='congdist', type='inner')
ny.agg.df = data.frame(
    overall_mean = (rowSums(ny.tmp.df[, c(1, 5, 8)])) / 3,
    state = ny.tmp.df$state,
    mrp_estimate = ny.tmp.df$mrp_estimate
  )

dist.agg.df <- rbind(tx.agg.df, ny.agg.df)

dist.2006.df = read.mrp.df('oh', num_clust, prec_dist, '2006')
dist.2008.df = read.mrp.df('oh', num_clust, prec_dist, '2008')
dist.2010.df = read.mrp.df('oh', num_clust, prec_dist, '2010')
oh.tmp.df <- join(dist.2006.df, dist.2008.df, by='congdist', type='inner')
oh.tmp.df <- join(oh.tmp.df, dist.2010.df, by='congdist', type='inner')
oh.agg.df = data.frame(
  overall_mean = (rowSums(oh.tmp.df[, c(1, 5, 8)])) / 3,
  state = oh.tmp.df$state,
  mrp_estimate = oh.tmp.df$mrp_estimate
)

dist.agg.df <- rbind(dist.agg.df, oh.agg.df)

main = 'Comparison of MRP Estimates and Our District-Level Estimates'
xlab = 'Weighted Mean District Estimate (log)'
ylab = 'MRP Estimate'
outfile = paste(outdir, '/', num_clust, 'c_', prec_dist, '_mrp_estimate.pdf', sep='')

# scaling
dist.agg.df$scaled_mean = sign(dist.agg.df$overall_mean) * log( abs(dist.agg.df$overall_mean) + 1 )
lin.model = plot.comp(dist.agg.df, 'scaled_mean', 'mrp_estimate', main, xlab, ylab, outfile,
                      TRUE)
summary(lin.model)


###### plotting CW MRP vs. CCES
# need to aggregate the district-level values in the same decade
read.comp.df = function(state, num_clust, prec_dist, year) {
  comp.dir = paste('../results/', state, '/', num_clust, '_clusters/mh_',
                   year, '/', prec_dist, '/comparison_charts', sep='')
  dist.df <- read.csv(paste(comp.dir, 'dist_comp.csv', sep='/'), header=T)
  dist.df$state <- state
  return(dist.df)
}

year = '2010'
state = 'oh'
column = 'self_p'
dist.df = read.comp.df(state, num_clust, prec_dist, year)

main = paste('Comparison of MRP Estimates and CCES Self-P for ', state, sep='')
xlab = 'CCES Self-P'
ylab = 'MRP Estimate'
outfile = paste(outdir, '/', num_clust, 'c_', prec_dist, '_cces_', column, '_mrp.png', sep='')
lin.model = plot.comp(dist.df, column, 'mrp_estimate', main, xlab, ylab, outfile,
                      TRUE)
summary(lin.model)


## for all years and states combined
create.state.agg.df = function(state, num_clust, prec_dist, year, column, use.2010) {
  dist.2006.df = read.comp.df(state, num_clust, prec_dist, '2006')
  dist.2006.df[paste(column, '2006', sep='_')] = dist.2006.df[column]
  dist.2008.df = read.comp.df(state, num_clust, prec_dist, '2008')
  dist.2008.df[paste(column, '2008', sep='_')] = dist.2008.df[column]
  state.tmp.df <- join(dist.2006.df, dist.2008.df, by='congdist', type='inner')
  column_lst = c(paste(column, '2006', sep='_'), paste(column, '2008', sep='_'))

  if (use.2010 == TRUE) {
    dist.2010.df = read.comp.df(state, num_clust, prec_dist, '2010')
    dist.2010.df[paste(column, '2010', sep='_')] = dist.2010.df[column]
    state.tmp.df <- join(state.tmp.df, dist.2010.df, by='congdist', type='inner')
    column_lst = c(paste(column, '2006', sep='_'), paste(column, '2008', sep='_'),
                   paste(column, '2010', sep='_'))
  }
  state.agg.df = data.frame(
    overall_val = (rowSums(state.tmp.df[, column_lst])) / length(column_lst),
    state = state,
    mrp_estimate = state.tmp.df$mrp_estimate
  )
  return(state.agg.df)
}

column = 'ideo'
use.2010 = TRUE
tx.agg.df = create.state.agg.df('tx', num_clust, prec_dist, year, column, use.2010)
ny.agg.df = create.state.agg.df('ny', num_clust, prec_dist, year, column, use.2010)
dist.agg.df <- rbind(tx.agg.df, ny.agg.df)

oh.agg.df = create.state.agg.df('oh', num_clust, prec_dist, year, column, use.2010)
dist.agg.df <- rbind(dist.agg.df, oh.agg.df)

main = paste('Comparison of MRP Estimates and CCES ', column, ' for ', state, sep='')
xlab = 'MRP Estimate'
ylab = paste('Overall CCES', column)
outfile = paste(outdir, '/', 'overall_', num_clust, 'c_', prec_dist, '_cces_', column, '_mrp.png', sep='')

# scaling
lin.model = plot.comp(dist.agg.df, 'mrp_estimate', 'overall_val', main, xlab, ylab, outfile,
                      TRUE)
summary(lin.model)

