import os
import socket

import pandas as pd
import numpy as np
import didipack as didi
import tqdm
from matplotlib import pyplot as plt
from source.deepsurrogate import *
import time
import tensorflow_probability as tfp
import tqdm
import os
from plot_function import DidiPlot
if socket.gethostname()=='work':
    start_dir='/home/antoinedidisheim/Dropbox/phd/Projects/fast_option_pricing/fop_code/'
else:
    start_dir='data/'
res_dir = 'res/delta_fig/'
os.makedirs(res_dir,exist_ok=True)

def plot(df,col,x_lab,y_lab,save_name):
    # fig = plt.figure(figsize=[6.4, 4.8 * 0.7])
    # t=df.loc[:,:].groupby('strike')[['mse_heston','mse_bate','mse_bench']].mean()
    label_ = 'Double Exponential Model'
    plt.plot(df[col],df['Surrogate'],label=label_+' Surrogate', color = DidiPlot.COLOR[0], linestyle=DidiPlot.LINE_STYLE[0])
    plt.plot(df[col],df['delta'], label=label_+' "True"', color = DidiPlot.COLOR[1], linestyle=DidiPlot.LINE_STYLE[1])
    plt.ylabel(y_lab)
    plt.xlabel(x_lab)
    # min_=df[['pred','iv']].min().min()
    # max_ = df[['pred','iv']].max().max()
    # if (max_-min_)<0.01:
    #     min_-=0.01
    #     max_+=0.01
    #
    #     plt.ylim([min_,max_])
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(res_dir+save_name+'.png')
    plt.show()


surogate = DeepSurrogate('bdjm')

## DELTA
df = pd.read_csv('/home/antoinedidisheim/Dropbox/phd/Projects/fast_option_pricing/fop_code/qt_cpp/delta/res_delta_bate_once.csv',index_col=False).rename(columns={'lambda':'lambda_parameter'})
df['dividend'] = df['rf']
h=0.01

COL_PLUS_BATE =['v0',
                'kappa',
                'theta',
                'sigma',
                'rho',
                'lambda_parameter',
                'nuUp',
                'nuDown',
                'p',
                'rf',
                'dividend',
                'strike',
                'T']
df['cp']=-1
df['Surrogate'] = surogate.get_price_delta(df[COL_PLUS_BATE+['cp']],var='S')



plot(df.loc[df['optionid']==1,:],col='strike',x_lab=r'$\hat{K}$',y_lab=r'$\Delta$',save_name='delta')


#### T
df = pd.read_csv('/home/antoinedidisheim/Dropbox/phd/Projects/fast_option_pricing/fop_code/qt_cpp/delta/res_theta_bate_once.csv',index_col=False).rename(columns={'lambda':'lambda_parameter'})
df['dividend'] = df['rf']
h=0.01

df['cp']=-1
df['Surrogate'] = surogate.get_price_delta(df[COL_PLUS_BATE+['cp']],var='T')



plot(df.loc[(df['optionid']==2) & (df['T']>2),:],col='T',x_lab=r'$T$',y_lab=r'$\tau$',save_name='tau')


#### V0
df = pd.read_csv('/home/antoinedidisheim/Dropbox/phd/Projects/fast_option_pricing/fop_code/qt_cpp/delta/res_vega_bate_once.csv',index_col=False).rename(columns={'lambda':'lambda_parameter'})
df['dividend'] = df['rf']
h=0.01

df['cp']=-1
df['Surrogate'] = surogate.get_price_delta(df[COL_PLUS_BATE+['cp']],var='v0')



plot(df.loc[df['optionid']==3,:],col='v0',x_lab=r'$v_t$',y_lab=r'$d v_t$',save_name='vega')

# #### rho
# df = pd.read_csv('/home/antoinedidisheim/Dropbox/phd/Projects/fast_option_pricing/fop_code/qt_cpp/delta/res_rho_bate_once.csv',index_col=False).rename(columns={'lambda':'lambda_parameter'})
# df['dividend'] = df['rf']
# h=0.01
#
# df['cp']=-1
# df['Surrogate'] = surogate.get_price_delta(df[COL_PLUS_BATE+['cp']],var='rf')
#
# df.loc[df['optionid']==4,:].groupby('rf')[['delta','Surrogate']].mean().plot()
# plt.show()