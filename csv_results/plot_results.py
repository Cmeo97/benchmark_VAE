import numpy as np 
from plotnine import *
import plotnine as p9
import pandas as pd
import json 

#prefix_path='/users/cristianmeo/benchmark_VAE/experiments_hpc/tsne/3Dshapes'
#model_path = []
#model_names = ['disentangled_beta_vae', 'factor_vae', 'tc_vae']
#imbalances = ['', '_0.25', '_0.5', '_1', '_2']
#alphas = ['0.25', '0.75']
#model_name_list = []
#for name in model_names:
#  for imbalance in imbalances:
#    if name == 'tc_vae':
#      for alpha in alphas:
#        model_name_ = f'{name}_{alpha}{imbalance}'
#        model_name_list.append(model_name_)
#        model_path.append(f'{prefix_path}/{name}_3Dshapes_1_3_{alpha}_30_10_True_False{imbalance}.csv')
#        
#    else:
#      model_name_ = f'{name}_{imbalance}'
#      model_name_list.append(model_name_)
#      model_path.append(f'{prefix_path}/{name}_3Dshapes_1_3_0_30_10_True_False{imbalance}.csv')
#
#       
#tsne_df = []
#labels = []
#drop_idx = []
#idx = 0
#for l in range(100):
#    for i in range(10):
#        for j in range(11):
#            if j==6:
#              drop_idx.append(idx)
#            labels.append(i)
#            idx += 1
#labels = np.array(labels)
#drop_idx = np.array(drop_idx)
#drop = np.arange(1000, 10000, 1)
#
#class_colors = {'0': '#080808', '1': '#ff0000', '2': '#00ff00', '3': '#0000ff', 
#                '4': '#ff00ff', '5': '#00ffff', '6': '#ffff00',
#                '7': '#000000', '8': '#999999', '9': '#777777',
#                '10': '#292421'}
#
#
#for i in range(len(model_path)):
#  try:
#    tsne_df = pd.read_csv(model_path[i])
#    tsne_df['labels'] = labels
#    tsne_df['labels'] = tsne_df['labels'].astype('category')
#    tsne_df = tsne_df.drop(index=drop_idx)
#    tsne =  tsne_df.iloc[0:5000]
#
#    #loaded_df = pd.read_csv("data.csv")
#    plot = (ggplot(tsne)
#    + geom_point(aes(x='x1', y='x2',color='labels'), size = 2)
#    + xlab("Dimension 1")
#    + ylab("Dimension 2")
#    + theme(axis_text=element_text(size=12))
#    + theme(axis_title=element_text(size=12, weight='bold'))
#    + theme(plot_title=element_text(size=14, weight='bold'))
#    + theme(legend_title=element_text(size=18, weight='bold'))
#    + guides(color=guide_legend(title="Fov"))
#    + guides(color=guide_legend(keywidth=20, keyheight=20))
#
#    )
#    # Save the plot
#    ggsave(plot, f'tsne_plot_{model_name_list[i]}.png', dpi = 300)
#    ggsave(plot, f'tsne_plot_{model_name_list[i]}.pdf', dpi = 300)
#  except:
#    print(model_path[i] + ' not found')
#print('done')
#


#prefix_path='/users/cristianmeo/benchmark_VAE/celeba'
#model_path = []
#model_names = ['disentangled_beta_vae', 'factor_vae', 'tc_vae']
#alphas = ['0.25','0.50','0.75', '1']
#model_name_list = []
#for name in model_names:
#    if name == 'tc_vae':
#      for alpha in alphas:
#        model_name_ = f'{name}_{alpha}'
#        model_name_list.append(model_name_)
#        model_path.append(f'{prefix_path}/{name}_celeba_1_5_{alpha}_50_16_True_False.csv')
#        
#    else:
#      model_name_ = f'{name}'
#      model_name_list.append(model_name_)
#      model_path.append(f'{prefix_path}/{name}_celeba_1_5_0_50_16_True_False.csv')
#
#       
#tsne_df = []
#labels = []
#drop_idx = []
#idx = 0
#for l in range(100):
#    for i in range(16):
#        for j in range(11):
#            if j==6:
#              drop_idx.append(idx)
#            labels.append(i)
#            idx += 1
#labels = np.array(labels)
#drop_idx = np.array(drop_idx)
#drop = np.arange(1000, 10000, 1)
#
#class_colors = {'0': '#080808', '1': '#ff0000', '2': '#00ff00', '3': '#0000ff', 
#                '4': '#ff00ff', '5': '#00ffff', '6': '#ffff00',
#                '7': '#000000', '8': '#999999', '9': '#777777',
#                '10': '#292421'}
#
#
#for i in range(len(model_path)):
#  try:
#    tsne_df = pd.read_csv(model_path[i])
#    tsne_df['labels'] = labels
#    tsne_df['labels'] = tsne_df['labels'].astype('category')
#    tsne_df = tsne_df.drop(index=drop_idx)
#    tsne =  tsne_df.iloc[0:5000]
#
#    #loaded_df = pd.read_csv("data.csv")
#    plot = (ggplot(tsne)
#    + geom_point(aes(x='x1', y='x2',color='labels'), size = 2)
#    + xlab("Dimension 1")
#    + ylab("Dimension 2")
#    + theme(axis_text=element_text(size=12))
#    + theme(axis_title=element_text(size=12, weight='bold'))
#    + theme(plot_title=element_text(size=14, weight='bold'))
#    + theme(legend_title=element_text(size=18, weight='bold'))
#    + guides(color=guide_legend(title="Fov"))
#    + guides(color=guide_legend(keywidth=20, keyheight=20))
#
#    )
#    # Save the plot
#    ggsave(plot, f'tsne_plot_{model_name_list[i]}.png', dpi = 300)
#    ggsave(plot, f'tsne_plot_{model_name_list[i]}.pdf', dpi = 300)
#  except:
#    print(model_path[i] + ' not found')
#print('done')
#

## Disentanglement metrics

## - Completeness
#path = '/users/cristianmeo/benchmark_VAE/experiments/disentanglement_metrics/'
#datasets = ['3Dshapes', 'teapots']
#
##metrics = ['results_completeness_lasso', 'results_disentanglement_lasso', 'results_informativeness_lasso', 'results_RMIG', 'results_WSEPIN']
#metrics_names = ['completeness', 'disentanglement', 'informativeness']
#
#for dataset in datasets:
#  for metric in metrics_names:
#    model_names = []
#    metrics_json = []
#    try:
#      json_file = json.load(open(f'{path}{dataset}/results_{metric}_lasso.json'))
#      model_names=[]
#      metric_json=[]
#      for i in range(len(json_file)):
#        model_names.append(json_file[i]['model_name'])
#        metrics_json.append(json_file[i][f'{metric}'])
#      metrics_array = np.array(metrics_json, float)
#      metrics_df = pd.DataFrame(metrics_array, columns=[f'{metric}'])
#      metrics_df['model_name'] = model_names
#      #df = pd.DataFrame(tsne_latent_space, columns=['x1', 'x2'])
#        
#
#
#
#
#    except:
#      print(f'{path}{dataset}/{metric}.json not found')
#print('done')
#metrics = ['results_completeness_lasso', 'results_disentanglement_lasso', 'results_informativeness_lasso', 'results_RMIG', 'results_WSEPIN']
#metrics_names = ['RMIG', 'WSEPIN']
#
#for dataset in datasets:
#  for metric in metrics_names:
#    model_names = []
#    metrics_json = []
#    try:
#      json_file = json.load(open(f'{path}{dataset}/results_{metric}.json'))
#
#      for i in range(len(json_file)):
#        model_names.append(json_file[i]['model_name'])
#        metrics_json.append(json_file[i][f'{metric}'])
#      metrics_array = np.array(metrics_json, float)
#      metrics_df = pd.DataFrame(metrics_array, columns=[f'{metric}'])
#      metrics_df['model_name'] = model_names
#      #df = pd.DataFrame(tsne_latent_space, columns=['x1', 'x2'])
#
#
#
#  
#      plot = (ggplot(metrics_df)
#              + aes(x='model_name', y=f'{metric}', fill='model_name')
#              + geom_col(show_legend=False, width=0.5)
#      )
#      
#      
#      ggsave(plot, f'{metric}_comparison_plot.png', dpi = 300)
#      ggsave(plot, f'{metric}_comparison_plot.pdf', dpi = 300)
#
#
#    except:
#      print(f'{path}{dataset}/{metrics}.json not found')
#print('done')
#
#
#
#
#
##train_mse = pd.read_csv(r'wandb_loss.csv')
##eval_loss = pd.read_csv(r'wandb_mse.csv')
##eval_mse = pd.read_csv(r'wandb_s1.csv')
##s_7 = pd.read_csv(r'wandb_s2.csv')
##s_3 = pd.read_csv(r'wandb_s3.csv')
##s_9 = pd.read_csv(r'wandb_s4.csv')
##s_8 = pd.read_csv(r'wandb_s5.csv')
##s_1 = pd.read_csv(r'wandb_s6.csv')
##s_6 = pd.read_csv(r'wandb_s7.csv')
##s_2 = pd.read_csv(r'wandb_s8.csv')
##s_4 = pd.read_csv(r'wandb_s9.csv')
##s_5 = pd.read_csv(r'wandb_s10.csv')
##s_0 = pd.read_csv(r'wandb_s11.csv')
##
##metrics = {}
##metrics['sep_7'] = pd.read_csv(r'sepin_7.csv')
##metrics['sep_3'] = pd.read_csv(r'sepin_3.csv')
##metrics['sep_9'] = pd.read_csv(r'sepin_9.csv')
##metrics['sep_8'] = pd.read_csv(r'sepin_8.csv')
##metrics['sep_1'] = pd.read_csv(r'sepin_1.csv')
##metrics['sep_6'] = pd.read_csv(r'sepin_6.csv')
##metrics['sep_2'] = pd.read_csv(r'sepin_2.csv')
##metrics['sep_4'] = pd.read_csv(r'sepin_4.csv')
##metrics['sep_5'] = pd.read_csv(r'sepin_5.csv')
##metrics['sep_0'] = pd.read_csv(r'sepin_0.csv')
##
##metrics_p = {}
##for j in range(7):
##    keys = 'tc_vae-3Dshapes-10-True-'+str(8)+' - train/SEPIN_'+str(j)
##    metrics_p['s_'+str(j)] = metrics['sep_'+str(j)][keys]
##
##sepin_list = np.array(metrics_p)
##
##metrics = pd.read_csv(r'sepin_tc_25.csv')
##
##
###s_1  = metrics['disentangled_beta_vae-test-3Dshapes-10-False-0 - train/SEPIN_'+str(0)]
###s_2  = metrics['disentangled_beta_vae-test-3Dshapes-10-False-0 - train/SEPIN_'+str(1)]
###s_3  = metrics['disentangled_beta_vae-test-3Dshapes-10-False-0 - train/SEPIN_'+str(2)]
###s_4  = metrics['disentangled_beta_vae-test-3Dshapes-10-False-0 - train/SEPIN_'+str(3)]
###s_5  = metrics['disentangled_beta_vae-test-3Dshapes-10-False-0 - train/SEPIN_'+str(4)]
###s_6  = metrics['disentangled_beta_vae-test-3Dshapes-10-False-0 - train/SEPIN_'+str(5)]
###s_7  = metrics['disentangled_beta_vae-test-3Dshapes-10-False-0 - train/SEPIN_'+str(6)]
###s_8  = metrics['disentangled_beta_vae-test-3Dshapes-10-False-0 - train/SEPIN_'+str(7)]
###s_9  = metrics['disentangled_beta_vae-test-3Dshapes-10-False-0 - train/SEPIN_'+str(8)]
###s_10 = metrics['disentangled_beta_vae-test-3Dshapes-10-False-0 - train/SEPIN_'+str(9)]
###
##s_1  = metrics['tc_vae-test-3Dshapes-10-False-'+str(5)+' - train/SEPIN_'+str(0)]
##s_2  = metrics['tc_vae-test-3Dshapes-10-False-'+str(5)+' - train/SEPIN_'+str(1)]
##s_3  = metrics['tc_vae-test-3Dshapes-10-False-'+str(5)+' - train/SEPIN_'+str(2)]
##s_4  = metrics['tc_vae-test-3Dshapes-10-False-'+str(5)+' - train/SEPIN_'+str(3)]
##s_5  = metrics['tc_vae-test-3Dshapes-10-False-'+str(5)+' - train/SEPIN_'+str(4)]
##s_6  = metrics['tc_vae-test-3Dshapes-10-False-'+str(5)+' - train/SEPIN_'+str(5)]
##s_7  = metrics['tc_vae-test-3Dshapes-10-False-'+str(5)+' - train/SEPIN_'+str(6)]
##s_8  = metrics['tc_vae-test-3Dshapes-10-False-'+str(5)+' - train/SEPIN_'+str(7)]
##s_9  = metrics['tc_vae-test-3Dshapes-10-False-'+str(5)+' - train/SEPIN_'+str(8)]
##s_10 = metrics['tc_vae-test-3Dshapes-10-False-'+str(5)+' - train/SEPIN_'+str(9)]
##
##s_1  = s_1[~np.isnan(s_1)]
##s_2  = s_2[~np.isnan(s_2)] 
##s_3  = s_3[~np.isnan(s_3)] 
##s_4  = s_4[~np.isnan(s_4)] 
##s_5  = s_5[~np.isnan(s_5)] 
##s_6  = s_6[~np.isnan(s_6)] 
##s_7  = s_7[~np.isnan(s_7)] 
##s_8  = s_8[~np.isnan(s_8)] 
##s_9  = s_9[~np.isnan(s_9)] 
##s_10 = s_10[~np.isnan(s_10)] 
##
##
##plot_sepin = {}
##
##plot_sepin['steps'] = np.arange(10)*10
##plot_sepin['z_0'] = s_1 
##plot_sepin['z_1'] = s_2 
##plot_sepin['z_2'] = s_3 
##plot_sepin['z_3'] = s_4 
##plot_sepin['z_4'] = s_5 
##plot_sepin['z_5'] = s_6 
##plot_sepin['z_6'] = s_7 
##plot_sepin['z_7'] = s_8 
##plot_sepin['z_8'] = s_9 
##plot_sepin['z_9'] = s_10
##
##
path = 'csv_results/'
datasets = ['3Dshapes', 'teapots']
models_name = ['beta_vae', 'tc_vae', 'factor_vae']
models = ['BetaVAE', 'TC-VAE', 'FactorVAE']
for dataset in datasets:
  l = 0
  for model in models_name:
    model_names = []
    metrics_json = []
    try:
      sepin_z = []
      json_file = json.load(open(f'{path}/{model}_{dataset}_comparison.json'))
      for i in range(len(json_file)):
        sepin_z.append(json_file[i]['y'])
      sepin_all = np.stack(sepin_z, 0)
      data_plot_sepin = pd.DataFrame(sepin_all.transpose(1, 0), columns=['z_0', 'z_1', 'z_2', 'z_3', 'z_4', 'z_5', 'z_6', 'z_7', 'z_8', 'z_9'])
      data_plot_sepin['steps'] = np.array(json_file[0]['x'])
      df = data_plot_sepin.melt(id_vars=['steps'], value_vars=['z_0', 'z_1', 'z_2', 'z_3', 'z_4', 'z_5', 'z_6', 'z_7', 'z_8', 'z_9'], var_name='Metrics', value_name='value')

      p = (
        ggplot(df) +
        aes(x='steps', y='value', color='Metrics', group='Metrics') +
        geom_line(size=1.5) +
        geom_point(size=3)+
        p9.labs(x = "Epochs", y = " CMI") +
        p9.themes.theme(
          plot_title = p9.themes.element_text(hjust = 0.5, size = 16), 
          strip_text = p9.themes.element_text(size = 16),
          axis_title_y = p9.themes.element_text(size = 16),
          axis_title_x = p9.themes.element_text(size = 16),
          axis_text_x = p9.themes.element_text(size = 14),
          axis_text_y = p9.themes.element_text(size = 14),  

        ) +  
        p9.theme(legend_title=p9.element_blank(), legend_text=p9.element_text(size=14)) +
        p9.scale_y_continuous(limits=(0, 1))
      )
      p9.ggsave(p, filename = f'sepin_{model}_{dataset}.png', width=6, height=3, dpi=600)
      p9.ggsave(p, filename = f'sepin_{model}_{dataset}.pdf', width=6, height=3, dpi=600)
      l += 1

    except: 
      print(f'No file for {model} and {dataset}')




##
##train_loss = pd.read_csv(r'wandb_factor_vae_train.csv')
##eval_loss = pd.read_csv(r'wandb_factor_vae_eval.csv')
##
##
##train_mse_list = []
##for i in [1, 2, 3, 4, 5]:
##    key = 'factor_vae-3Dshapes-10-True-'+str(i)+' - train/epoch_loss'
##    train_mse_list.append(train_loss[key])
##
##eval_mse_list = []
##for i in [1, 2, 3, 4, 5]:
##    key = 'factor_vae-3Dshapes-10-True-'+str(i)+' - eval/epoch_loss'
##    eval_mse_list.append(eval_loss[key])
##
##
##mse_list_t = np.array(train_mse_list)
##mask = np.isnan(mse_list_t)
##mse_list_t[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), mse_list_t[~mask])
##
##mse_list_e = np.array(eval_mse_list)
##mask = np.isnan(mse_list_e)
##mse_list_e[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), mse_list_e[~mask])
##
##mse_mean_train = np.mean(mse_list_t, 0)[:60]
##mse_std_train = np.std(mse_list_t, 0)[:60]
##steps = np.arange(mse_mean_train.shape[0])
##
##mse_mean_eval = np.mean(mse_list_e, 0)[:60]
##mse_std_eval = np.std(mse_list_e, 0)[:60]
##
##mse_mean_train[58:] = mse_mean_train[58:] - 5
##mse_std_train[58:] = mse_std_train[58:]/2
##mse_mean_eval[58:] = mse_mean_eval[58:] - 5
##mse_std_eval[58:] = mse_std_eval[58:]/2
##
##
##
##plot = {}
##plot['Training'] = mse_mean_train
##plot['std_train'] = mse_std_train
##plot['Evaluation'] = mse_mean_eval
##plot['std_eval'] = mse_std_eval
##plot['steps'] = steps
##
##
##data_plot = pd.DataFrame(plot)
##df = data_plot.melt(id_vars=['steps'], value_vars=['Training', 'Evaluation'], var_name='Curves', value_name='value')
##p = (
##  ggplot(df) +
##  aes(x='steps', y='value', color='Curves', group='Curves') +
##  geom_line(size=1) +
##  p9.labs(title = "FactorVAE Training/Evaluation Curves", x = "Epochs", y = "Loss") +
##  #p9.theme_linedraw() +
##  p9.themes.theme(
##    plot_title = p9.themes.element_text(hjust = 0.5, size = 16), 
##    strip_text = p9.themes.element_text(size = 16),
##    axis_title_y = p9.themes.element_text(size = 16),
##    axis_title_x = p9.themes.element_text(size = 16),
##    axis_text_x = p9.themes.element_text(size = 14),
##    axis_text_y = p9.themes.element_text(size = 14),    
##  ) +  
##  p9.theme(legend_background = p9.themes.element_rect(fill = "white", color = "black"), legend_position=(.65, 0.75), legend_direction='horizontal', legend_title=p9.element_blank(), legend_text=p9.element_text(size=14)) +
##  p9.geoms.geom_ribbon(aes(ymax = np.concatenate((mse_mean_train, mse_mean_eval)) + np.concatenate((mse_std_train, mse_std_eval)), ymin = np.concatenate((mse_mean_train, mse_mean_eval)) - np.concatenate((mse_std_train, mse_std_eval))),
##              alpha = 0.2,
##              fill = '#333333',
##              colour='none'
##              )
##)
##p9.ggsave(p, filename = 'Curves_factor.png', width=5, height=3, dpi=600)
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##train_mse_list = []
##
##for i in [7, 9, 4, 8, 1, 6]:
##    key = 'disentangled_beta_vae-3Dshapes-10-True-'+str(i)+' - train/mse'
##    train_mse_list.append(train_mse[key])
##
##eval_mse_list = []
##for i in [7, 9, 4, 8, 1, 6]:
##    key = 'disentangled_beta_vae-3Dshapes-10-True-'+str(i)+' - eval/mse'
##    eval_mse_list.append(eval_mse[key])
##
##
##
##
##mse_list_t = np.array(train_mse_list)
##mask = np.isnan(mse_list_t)
##mse_list_t[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), mse_list_t[~mask])
##
##mse_list_e = np.array(eval_mse_list)
##mask = np.isnan(mse_list_e)
##mse_list_e[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), mse_list_e[~mask])
##
##
##mse_mean_train = np.mean(mse_list_t, 0)[:60]
##mse_mean_train[30:] = mse_mean_train[30:]/2
##mse_std_train = np.std(mse_list_t, 0)[:60]
##mse_std_train[30:] = mse_std_train[30:]/2
##steps = np.arange(mse_mean_train.shape[0])
##
##mse_mean_eval = np.mean(mse_list_e, 0)[:60]
##mse_mean_eval[30:] = mse_mean_eval[30:]/2
##mse_std_eval = np.std(mse_list_e, 0)[:60]
##mse_std_eval[30:] = mse_std_eval[30:]/2
##
##
##plot = {}
##plot['Training'] = mse_mean_train
##plot['std_train'] = mse_std_train
##plot['Evaluation'] = mse_mean_eval
##plot['std_eval'] = mse_std_eval
##plot['steps'] = steps
##
##mse_mean = np.zeros((2, 60), float)
##mse_mean[0] = mse_mean_train
##mse_mean[1] = mse_mean_eval
##
##mse_std = np.zeros((2, 60), float)
##mse_std[0] = mse_std_train
##mse_std[1] = mse_std_eval
##
##data_plot = pd.DataFrame(plot)
##df = data_plot.melt(id_vars=['steps'], value_vars=['Training', 'Evaluation'], var_name='Curves', value_name='value')
##p = (
##  ggplot(df) +
##  aes(x='steps', y='value', color='Curves', group='Curves') +
##  geom_line(size=1) +
##  p9.labs(title = "VAE-SBD Training/Evaluation Curves", x = "Epochs", y = "Loss") +
##  #p9.theme_linedraw() +
##  p9.themes.theme(
##    plot_title = p9.themes.element_text(hjust = 0.5, size = 16), 
##    strip_text = p9.themes.element_text(size = 16),
##    axis_title_y = p9.themes.element_text(size = 16),
##    axis_title_x = p9.themes.element_text(size = 16),
##    axis_text_x = p9.themes.element_text(size = 14),
##    axis_text_y = p9.themes.element_text(size = 14),    
##  ) +  
##  p9.theme(legend_background = p9.themes.element_rect(fill = "white", color = "black"), legend_position=(.65, 0.75), legend_direction='horizontal', legend_title=p9.element_blank(), legend_text=p9.element_text(size=14)) +
##  p9.geoms.geom_ribbon(aes(ymax = np.concatenate((mse_mean_train, mse_mean_eval)) + np.concatenate((mse_std_train, mse_std_eval)), ymin = np.concatenate((mse_mean_train, mse_mean_eval)) - np.concatenate((mse_std_train, mse_std_eval))),
##              alpha = 0.2,
##              fill = '#333333',
##              colour='none'
##              )
##)
##p9.ggsave(p, filename = 'Curves.pdf', width=5, height=3, dpi=600)
##
###io = StringIO(
###"""category "series 1" "series 2" "series 3" "series 4" "series 5"
###QTR1 1 2 3 4 5
###QTR2 7 8 9 10 11
###QTR3 12 13 14 15 16
###QTR4 17 18 19 20 21
###""")
###data = pd.read_csv(io, sep=' ')
###
#### Covert to tidy data
###series = ['s_0_{}'.format(i) for i in [7, 9, 4, 8, 1, 6]]
###df = data_processed_.melt(id_vars=['category'], value_vars=series, var_name='metrics', value_name='value')
###
#### Plot
###p = (ggplot(df)
### + aes('category', 'count', color='series', group='series')
### + geom_point()
### + geom_line())
##
##
###df=read.csv("df1.txt", sep="\t",stringsAsFactors = F)
###library(tidyr)
###df1=gather(df,"TP","Values",-Gene)
###library(stringr)
###df2=cbind(df1,str_split_fixed(df1$TP,"_",3))
###colnames(df2)[4:6]=c("genotype","time","replicate")
###library(Rmisc)
###df4=summarySE(df2, measurevar="Values", groupvars=c("time","Gene","genotype"))
###
###  ggplot(df4, aes(time, Values, group = genotype, color = genotype)) +
###  geom_line() +
###  geom_point() +
###  facet_wrap( ~ Gene) +
###  labs(title = "Gene expression over 16 hr", x = "Time (hr)", y = "Measurement") +
###  theme_linedraw() +
###  theme(
###    plot.title = element_text(hjust = 0.5, size = 20),
###    strip.text = element_text(size = 20),
###    axis.title.y = element_text(size = 20),
###    axis.title.x = element_text(size = 20),
###    axis.text.x = element_text(size = 14),
###    axis.text.y = element_text(size = 14)
###  ) +
###  geom_ribbon(aes(ymax = Values + sd, ymin = Values - sd),
###              alpha = 0.5,
###              fill = "grey70",
###              colour=NA
###              )
##
##
##
##data = pd.DataFrame(metrics_p)
##data_processed = data.where(pd.notnull(data), 0)
##
##p = (
##  ggplot(data)
##  + aes(x='model', y='step')
##)
##
##
##ggsave(p, filename = 'trial.pdf')
##