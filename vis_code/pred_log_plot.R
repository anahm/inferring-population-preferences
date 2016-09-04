# pred_log_plot.py
#
# Plots the different error rates of the prediction methods.

require(ggplot2)

# Parameters that need to be set
chart.dir <- '../results/pred_plots/' # Output directory

# Name of input file
input_file = paste(chart.dir, '4c_uniform_se.csv', sep='')
pred.df <- read.csv(intput_file, header=T)

# Name of output file (out.fname)
out.fname <- paste(paste(num_clusters, 'c', sep=''), filter_dist_type, 'pred_bar', sep='_')

#########

# Filter out mrp_basic because the error is too large and skews plots
pred.df <- pred.df[pred.df$method != 'mrp_basic', ]
pred.df <- pred.df[pred.df$dist == filter_dist_type, ]

pred.df$name <- as.character(pred.df$name)
pred.df$min.error <- pred.df$error - pred.df$se
pred.df$max.error <- pred.df$error + pred.df$se

gg.var <- ggplot(pred.df) +
  geom_bar(aes(x=name, y=error, fill=method), stat='identity', position=position_dodge()) +
  geom_errorbar(aes(x=name, ymin=min.error, ymax=max.error, color=method),
                width=0, position=position_dodge(width=0.9)) +
  facet_grid(. ~ pred_type, scales='free') +
  xlab('Election Cycle Being Predicted') +
  ylab('Mean Squared Error') +
  scale_y_continuous(expand=c(0,0)) +
  theme(panel.background = element_blank(),
        axis.line = element_line(colour = "black"),
        axis.title.x=element_text(margin=margin(25,0,10,0)),
        axis.title.y=element_text(margin=margin(0,25,0,10)),
        text = element_text(family = "Times", size=48),
        legend.key = element_rect(size = 6),
        legend.key.size = unit(3, 'lines')
      ) +
  # scale_color_brewer(palette="Set1")
  scale_fill_manual(name="Method",
                    labels=c('inferred' = 'Inferred Data',
                             'baseline' = 'Baseline Task',
                             'survey' = 'Survey Data',
                             'mrp_basic' = 'MRP Scale',
                             'mrp_cross' = 'MRP Cross-Val'),
                    values=c('baseline'='#762a83', 'inferred'='#af8dc3', 'mrp_basic'='#e7d4e8',
                             'mrp_cross'='#7fbf7b', 'survey'='#1b7837')) +
  scale_colour_manual(name='', values=c('baseline'='black', 'inferred'='black', 'mrp_basic'='black',
                             'mrp_cross'= 'black', 'survey'='black'), guide=FALSE)

print(gg.var)
ggsave(paste(chart.dir, out.fname, '.pdf', sep=''), width=20, height=7)
