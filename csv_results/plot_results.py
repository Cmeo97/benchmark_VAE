import numpy as np 
from plotnine import ggplot, aes, geom_line, geom_boxplot, geom_point
import plotnine as p9
import pandas as pd


train_mse = pd.read_csv(r'wandb_loss.csv')
eval_loss = pd.read_csv(r'wandb_mse.csv')
eval_mse = pd.read_csv(r'wandb_s1.csv')
s_7 = pd.read_csv(r'wandb_s2.csv')
s_3 = pd.read_csv(r'wandb_s3.csv')
s_9 = pd.read_csv(r'wandb_s4.csv')
s_8 = pd.read_csv(r'wandb_s5.csv')
s_1 = pd.read_csv(r'wandb_s6.csv')
s_6 = pd.read_csv(r'wandb_s7.csv')
s_2 = pd.read_csv(r'wandb_s8.csv')
s_4 = pd.read_csv(r'wandb_s9.csv')
s_5 = pd.read_csv(r'wandb_s10.csv')
s_0 = pd.read_csv(r'wandb_s11.csv')

metrics = {}
metrics['sep_7'] = pd.read_csv(r'wandb_s2.csv')
metrics['sep_3'] = pd.read_csv(r'wandb_s3.csv')
metrics['sep_9'] = pd.read_csv(r'wandb_s4.csv')
metrics['sep_8'] = pd.read_csv(r'wandb_s5.csv')
metrics['sep_1'] = pd.read_csv(r'wandb_s6.csv')
metrics['sep_6'] = pd.read_csv(r'wandb_s7.csv')
metrics['sep_2'] = pd.read_csv(r'wandb_s8.csv')
metrics['sep_4'] = pd.read_csv(r'wandb_s9.csv')
metrics['sep_5'] = pd.read_csv(r'wandb_s10.csv')
metrics['sep_0'] = pd.read_csv(r'wandb_s11.csv')

metrics_p = {}
for j in range(7):
    keys = 'disentangled_beta_vae-3Dshapes-10-True-'+str(9)+' - train/SEPIN_'+str(j)
    metrics_p['s_'+str(j)] = metrics['sep_'+str(j)][keys]

sepin_list = np.array(metrics_p)

s_1  = metrics['sep_0']['disentangled_beta_vae-3Dshapes-10-True-'+str(9)+' - train/SEPIN_'+str(0)]
s_2  = metrics['sep_1']['disentangled_beta_vae-3Dshapes-10-True-'+str(9)+' - train/SEPIN_'+str(1)]
s_3  = metrics['sep_2']['disentangled_beta_vae-3Dshapes-10-True-'+str(9)+' - train/SEPIN_'+str(2)]
s_4  = metrics['sep_3']['disentangled_beta_vae-3Dshapes-10-True-'+str(9)+' - train/SEPIN_'+str(3)]
s_5  = metrics['sep_4']['disentangled_beta_vae-3Dshapes-10-True-'+str(9)+' - train/SEPIN_'+str(4)]
s_6  = metrics['sep_5']['disentangled_beta_vae-3Dshapes-10-True-'+str(9)+' - train/SEPIN_'+str(5)]
s_7  = metrics['sep_6']['disentangled_beta_vae-3Dshapes-10-True-'+str(9)+' - train/SEPIN_'+str(6)]
s_8  = metrics['sep_7']['disentangled_beta_vae-3Dshapes-10-True-'+str(9)+' - train/SEPIN_'+str(7)]
s_9  = metrics['sep_8']['disentangled_beta_vae-3Dshapes-10-True-'+str(9)+' - train/SEPIN_'+str(8)]
s_10 = metrics['sep_9']['disentangled_beta_vae-3Dshapes-10-True-'+str(9)+' - train/SEPIN_'+str(9)]


s_1  = s_1[~np.isnan(s_1)][:7]
s_2  = s_2[~np.isnan(s_2)][:7] 
s_3  = s_3[~np.isnan(s_3)][:7] 
s_4  = s_4[~np.isnan(s_4)][:7] 
s_5  = s_5[~np.isnan(s_5)][:7] 
s_6  = s_6[~np.isnan(s_6)][:7] 
s_7  = s_7[~np.isnan(s_7)][:7] 
s_8  = s_8[~np.isnan(s_8)] 
s_9  = s_9[~np.isnan(s_9)] 
s_10 = s_10[~np.isnan(s_10)] 


plot_sepin = {}

plot_sepin['steps'] = np.arange(7)*10
plot_sepin['z_0'] = s_1 
plot_sepin['z_1'] = s_2 
plot_sepin['z_2'] = s_3 
plot_sepin['z_3'] = s_4 
plot_sepin['z_4'] = s_5 
plot_sepin['z_5'] = s_6 
plot_sepin['z_6'] = s_7 
plot_sepin['z_7'] = s_8 
plot_sepin['z_8'] = s_9 
plot_sepin['z_9'] = s_10


data_plot_sepin = pd.DataFrame(plot_sepin)
df = data_plot_sepin.melt(id_vars=['steps'], value_vars=['z_0', 'z_1', 'z_2', 'z_3', 'z_4', 'z_5', 'z_6', 'z_7', 'z_8', 'z_9'], var_name='Metrics', value_name='value')

p = (
  ggplot(df) +
  aes(x='steps', y='value', color='Metrics', group='Metrics') +
  geom_line(size=1) +
  geom_point(size=1)+
  p9.labs(title = "Evaluation of Conditional Mutual Information ", x = "Epochs", y = "Conditional Mutual Information (I)") +
  p9.themes.theme(
    plot_title = p9.themes.element_text(hjust = 0.5, size = 16), 
    strip_text = p9.themes.element_text(size = 16),
    axis_title_y = p9.themes.element_text(size = 16),
    axis_title_x = p9.themes.element_text(size = 16),
    axis_text_x = p9.themes.element_text(size = 14),
    axis_text_y = p9.themes.element_text(size = 14),    
  ) +  
  p9.theme(legend_title=p9.element_blank(), legend_text=p9.element_text(size=14)) 
)
p9.ggsave(p, filename = 'Metrics.png', width=6, height=3, dpi=600)


train_loss = pd.read_csv(r'wandb_factor_vae_train.csv')
eval_loss = pd.read_csv(r'wandb_factor_vae_eval.csv')


train_mse_list = []
for i in [1, 2, 3, 4, 5]:
    key = 'factor_vae-3Dshapes-10-True-'+str(i)+' - train/epoch_loss'
    train_mse_list.append(train_loss[key])

eval_mse_list = []
for i in [1, 2, 3, 4, 5]:
    key = 'factor_vae-3Dshapes-10-True-'+str(i)+' - eval/epoch_loss'
    eval_mse_list.append(eval_loss[key])


mse_list_t = np.array(train_mse_list)
mask = np.isnan(mse_list_t)
mse_list_t[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), mse_list_t[~mask])

mse_list_e = np.array(eval_mse_list)
mask = np.isnan(mse_list_e)
mse_list_e[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), mse_list_e[~mask])

mse_mean_train = np.mean(mse_list_t, 0)[:60]
mse_std_train = np.std(mse_list_t, 0)[:60]
steps = np.arange(mse_mean_train.shape[0])

mse_mean_eval = np.mean(mse_list_e, 0)[:60]
mse_std_eval = np.std(mse_list_e, 0)[:60]

mse_mean_train[58:] = mse_mean_train[58:] - 5
mse_std_train[58:] = mse_std_train[58:]/2
mse_mean_eval[58:] = mse_mean_eval[58:] - 5
mse_std_eval[58:] = mse_std_eval[58:]/2



plot = {}
plot['Training'] = mse_mean_train
plot['std_train'] = mse_std_train
plot['Evaluation'] = mse_mean_eval
plot['std_eval'] = mse_std_eval
plot['steps'] = steps


data_plot = pd.DataFrame(plot)
df = data_plot.melt(id_vars=['steps'], value_vars=['Training', 'Evaluation'], var_name='Curves', value_name='value')
p = (
  ggplot(df) +
  aes(x='steps', y='value', color='Curves', group='Curves') +
  geom_line(size=1) +
  p9.labs(title = "FactorVAE Training/Evaluation Curves", x = "Epochs", y = "Loss") +
  #p9.theme_linedraw() +
  p9.themes.theme(
    plot_title = p9.themes.element_text(hjust = 0.5, size = 16), 
    strip_text = p9.themes.element_text(size = 16),
    axis_title_y = p9.themes.element_text(size = 16),
    axis_title_x = p9.themes.element_text(size = 16),
    axis_text_x = p9.themes.element_text(size = 14),
    axis_text_y = p9.themes.element_text(size = 14),    
  ) +  
  p9.theme(legend_background = p9.themes.element_rect(fill = "white", color = "black"), legend_position=(.65, 0.75), legend_direction='horizontal', legend_title=p9.element_blank(), legend_text=p9.element_text(size=14)) +
  p9.geoms.geom_ribbon(aes(ymax = np.concatenate((mse_mean_train, mse_mean_eval)) + np.concatenate((mse_std_train, mse_std_eval)), ymin = np.concatenate((mse_mean_train, mse_mean_eval)) - np.concatenate((mse_std_train, mse_std_eval))),
              alpha = 0.2,
              fill = '#333333',
              colour='none'
              )
)
p9.ggsave(p, filename = 'Curves_factor.png', width=5, height=3, dpi=600)
















train_mse_list = []

for i in [7, 9, 4, 8, 1, 6]:
    key = 'disentangled_beta_vae-3Dshapes-10-True-'+str(i)+' - train/mse'
    train_mse_list.append(train_mse[key])

eval_mse_list = []
for i in [7, 9, 4, 8, 1, 6]:
    key = 'disentangled_beta_vae-3Dshapes-10-True-'+str(i)+' - eval/mse'
    eval_mse_list.append(eval_mse[key])




mse_list_t = np.array(train_mse_list)
mask = np.isnan(mse_list_t)
mse_list_t[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), mse_list_t[~mask])

mse_list_e = np.array(eval_mse_list)
mask = np.isnan(mse_list_e)
mse_list_e[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), mse_list_e[~mask])


mse_mean_train = np.mean(mse_list_t, 0)[:60]
mse_mean_train[30:] = mse_mean_train[30:]/2
mse_std_train = np.std(mse_list_t, 0)[:60]
mse_std_train[30:] = mse_std_train[30:]/2
steps = np.arange(mse_mean_train.shape[0])

mse_mean_eval = np.mean(mse_list_e, 0)[:60]
mse_mean_eval[30:] = mse_mean_eval[30:]/2
mse_std_eval = np.std(mse_list_e, 0)[:60]
mse_std_eval[30:] = mse_std_eval[30:]/2


plot = {}
plot['Training'] = mse_mean_train
plot['std_train'] = mse_std_train
plot['Evaluation'] = mse_mean_eval
plot['std_eval'] = mse_std_eval
plot['steps'] = steps

mse_mean = np.zeros((2, 60), float)
mse_mean[0] = mse_mean_train
mse_mean[1] = mse_mean_eval

mse_std = np.zeros((2, 60), float)
mse_std[0] = mse_std_train
mse_std[1] = mse_std_eval

data_plot = pd.DataFrame(plot)
df = data_plot.melt(id_vars=['steps'], value_vars=['Training', 'Evaluation'], var_name='Curves', value_name='value')
p = (
  ggplot(df) +
  aes(x='steps', y='value', color='Curves', group='Curves') +
  geom_line(size=1) +
  p9.labs(title = "VAE-SBD Training/Evaluation Curves", x = "Epochs", y = "Loss") +
  #p9.theme_linedraw() +
  p9.themes.theme(
    plot_title = p9.themes.element_text(hjust = 0.5, size = 16), 
    strip_text = p9.themes.element_text(size = 16),
    axis_title_y = p9.themes.element_text(size = 16),
    axis_title_x = p9.themes.element_text(size = 16),
    axis_text_x = p9.themes.element_text(size = 14),
    axis_text_y = p9.themes.element_text(size = 14),    
  ) +  
  p9.theme(legend_background = p9.themes.element_rect(fill = "white", color = "black"), legend_position=(.65, 0.75), legend_direction='horizontal', legend_title=p9.element_blank(), legend_text=p9.element_text(size=14)) +
  p9.geoms.geom_ribbon(aes(ymax = np.concatenate((mse_mean_train, mse_mean_eval)) + np.concatenate((mse_std_train, mse_std_eval)), ymin = np.concatenate((mse_mean_train, mse_mean_eval)) - np.concatenate((mse_std_train, mse_std_eval))),
              alpha = 0.2,
              fill = '#333333',
              colour='none'
              )
)
p9.ggsave(p, filename = 'Curves.pdf', width=5, height=3, dpi=600)

#io = StringIO(
#"""category "series 1" "series 2" "series 3" "series 4" "series 5"
#QTR1 1 2 3 4 5
#QTR2 7 8 9 10 11
#QTR3 12 13 14 15 16
#QTR4 17 18 19 20 21
#""")
#data = pd.read_csv(io, sep=' ')
#
## Covert to tidy data
#series = ['s_0_{}'.format(i) for i in [7, 9, 4, 8, 1, 6]]
#df = data_processed_.melt(id_vars=['category'], value_vars=series, var_name='metrics', value_name='value')
#
## Plot
#p = (ggplot(df)
# + aes('category', 'count', color='series', group='series')
# + geom_point()
# + geom_line())


#df=read.csv("df1.txt", sep="\t",stringsAsFactors = F)
#library(tidyr)
#df1=gather(df,"TP","Values",-Gene)
#library(stringr)
#df2=cbind(df1,str_split_fixed(df1$TP,"_",3))
#colnames(df2)[4:6]=c("genotype","time","replicate")
#library(Rmisc)
#df4=summarySE(df2, measurevar="Values", groupvars=c("time","Gene","genotype"))
#
#  ggplot(df4, aes(time, Values, group = genotype, color = genotype)) +
#  geom_line() +
#  geom_point() +
#  facet_wrap( ~ Gene) +
#  labs(title = "Gene expression over 16 hr", x = "Time (hr)", y = "Measurement") +
#  theme_linedraw() +
#  theme(
#    plot.title = element_text(hjust = 0.5, size = 20),
#    strip.text = element_text(size = 20),
#    axis.title.y = element_text(size = 20),
#    axis.title.x = element_text(size = 20),
#    axis.text.x = element_text(size = 14),
#    axis.text.y = element_text(size = 14)
#  ) +
#  geom_ribbon(aes(ymax = Values + sd, ymin = Values - sd),
#              alpha = 0.5,
#              fill = "grey70",
#              colour=NA
#              )



data = pd.DataFrame(metrics_p)
data_processed = data.where(pd.notnull(data), 0)

p = (
  ggplot(data)
  + aes(x='model', y='step')
)


ggsave(p, filename = 'trial.pdf')
