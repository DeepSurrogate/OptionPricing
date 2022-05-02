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
if socket.gethostname()=='work':
    start_dir='/home/antoinedidisheim/Dropbox/phd/Projects/fast_option_pricing/fop_code/'
else:
    start_dir='data/'


model_name = 'bdjm'

opt = pd.read_pickle(start_dir+'data/real_new/all_sp.p')

res_h = pd.read_pickle(start_dir+'res/real_merge/res_h.p').reset_index().rename(columns={'index': 't_ind'})
res_b = pd.read_pickle(start_dir+'res/real_merge/res_b.p').reset_index().rename(columns={'index': 't_ind'})

print('loaded', flush=True)

##################
# select init paramters and v0 values in training
##################


surogate = DeepSurrogate('bdjm')
surogate_heston = DeepSurrogate('heston')

t = 5
def load_and_pre_process_df(res):
    # for c in res.columns:
    #     if c not in ['v0','t_ind']:
    #         res[c] = res[c].mean()

    COL = list(res.columns[1:])


    df = opt.merge(res)


    df['strike_un'] = df['strike_un'] / 1000
    ind = (df['strike_un']) < df['S']
    df.loc[ind, 'mid_p'] = df.loc[ind, 'mid_p'] + (df.loc[ind, 'strike_un']) * np.exp(-df.loc[ind, 'rf'] * df.loc[ind, 'T'] / 365) - df.loc[ind, 'S'] * np.exp(-df.loc[ind, 'dividend'] * df.loc[ind, 'T'] / 365)


    df = df.loc[df['year']>=2004,:]
    df = df.dropna()


    df = df.sort_values(['optionid','date']).reset_index(drop=True)
    df['iv_future'] = df.groupby('optionid')['iv'].shift(-t)
    df['mid_p_future'] = df.groupby('optionid')['mid_p'].shift(-1)
    df['T_future'] = df.groupby('optionid')['T'].shift(-1)

    df['cp'] = 1
    df.loc[df['strike']<100,'cp'] = -1

    # df=df.loc[df['T']>t,:]
    df['T'] = df['T']-t ### WE SET t in the future for when we predict the prices!
    return df, COL

##################
# use the surrogate model to interpolate the model's predicted implied volatility
##################
df_h, COL_HESTON = load_and_pre_process_df(res_h)
df_b, COL_BATE = load_and_pre_process_df(res_b)


# gradient descent
COL_PLUS_BATE = COL_BATE+['rf','dividend','strike','T']
COL_PLUS_HESTON = COL_HESTON+['rf','dividend','strike','T']
COL_BATE.remove('v0')

# load the single paramters
x_params = np.load('res/pred/par.npy')


df_daily_bates = df_b[COL_PLUS_BATE+['cp']].copy()
df_daily_heston = df_h[COL_PLUS_HESTON+['cp']].copy()
df_once = df_daily_bates.copy()

df_once[COL_BATE] = x_params
df_once['nuUp'] = df_once['nuUp'].clip(0,0.4)

SPLIT = 10
D = []
for df_split in tqdm.tqdm(np.array_split(df_daily_bates,SPLIT),'price bdjm'):
    d = surogate.get_iv(df_split[COL_PLUS_BATE])
    D.append(d)
d=np.concatenate(D,axis=0)
df_b['pred_bdjm'] = d

D = []
for df_split in tqdm.tqdm(np.array_split(df_daily_heston,SPLIT),'price heston'):
    d = surogate_heston.get_iv(df_split[COL_PLUS_HESTON])
    D.append(d)
d=np.concatenate(D,axis=0)
df_b['pred_heston'] = d


D = []
for df_split in tqdm.tqdm(np.array_split(df_once,SPLIT),'once'):
    d = surogate.get_iv(df_split[COL_PLUS_BATE])
    D.append(d)
d=np.concatenate(D,axis=0)
df_b['pred_once'] = d


### delta
df_b['T'] = df_b['T']+t
df_h['T'] = df_h['T']+t
df_once['T'] = df_once['T']+t

D = []
for df_split in tqdm.tqdm(np.array_split(df_daily_bates,SPLIT),'bdjm'):
    d = surogate.get_price_delta(df_split[COL_PLUS_BATE+['cp']],var='S')
    D.append(d.values)
d=np.concatenate(D,axis=0)
df_b['delta_bdjm'] = d

D = []
for df_split in tqdm.tqdm(np.array_split(df_daily_heston,SPLIT),'heston'):
    d = surogate_heston.get_price_delta(df_split[COL_PLUS_HESTON+['cp']],var='S')
    D.append(d.values)
d=np.concatenate(D,axis=0)
df_b['delta_heston'] = d


D = []
for df_split in tqdm.tqdm(np.array_split(df_once,SPLIT),'once'):
    d = surogate.get_price_delta(df_split[COL_PLUS_BATE+['cp']],var='S')
    D.append(d.values)
d=np.concatenate(D,axis=0)
df_b['delta_once'] = d


##################
# save
##################

df_b.to_pickle('res/df.p')


