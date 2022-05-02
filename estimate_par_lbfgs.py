import socket

import pandas as pd
import numpy as np
import didipack as didi
from matplotlib import pyplot as plt
from source.deepsurrogate import *
import time
import tensorflow_probability as tfp

if socket.gethostname()=='work':
    start_dir='/home/antoinedidisheim/Dropbox/phd/Projects/fast_option_pricing/fop_code/'
else:
    start_dir='data/'


model_name = 'bdjm'

opt = pd.read_pickle(start_dir+'data/real_new/all_sp.p')

res_h = pd.read_pickle(start_dir+'res/real_merge/res_h.p').reset_index().rename(columns={'index': 't_ind'})
res_b = pd.read_pickle(start_dir+'res/real_merge/res_b.p').reset_index().rename(columns={'index': 't_ind'})

##################
# select init paramters and v0 values in training
##################
res = res_b.copy() if model_name == 'bdjm' else res_h.copy()
for c in res.columns:
    if c not in ['v0','t_ind']:
        res[c] = res[c].mean()

COL = list(res.columns[1:])

surogate = DeepSurrogate(model_name)

##################
# merge opt and res and select number of year
##################

df = opt.merge(res)
df = df.loc[df['year']<=2004,:]
df = df.dropna()
#keep only wednsday
ind=df['date'].dt.dayofweek==2
df = df.loc[ind,:]

##################
# use the surrogate model to interpolate the model's predicted implied volatility
##################

# gradient descent
COL_PLUS = COL+['rf','dividend','strike','T']
COL.remove('v0')
#
# s=time.time()
# grad, d = surogate.get_iv_delta_and_pred(df[COL_PLUS])
# t = np.round((time.time() - s) / 60, 2)
# print(t)
#
# grad[COL_PLUS].mean()




##################
# bfgs version (doesn't converge)
##################

def func(x_params):
    df[COL] = x_params
    iv = surogate.get_iv(df[COL_PLUS])
    v = np.mean(np.square(iv-df[['iv']]))
    # pred = self.c_model.model([par_est, state, true_opt])
    # v_call = tf_loss(pred, Y)
    # bnd = tf.reduce_sum(tf.nn.relu(x_params - bounds) + tf.nn.relu(-(x_params + bounds))) * bound_cost
    return v


def func_g(x_params):
    df[COL] = x_params.numpy()
    grad, iv = surogate.get_iv_delta_and_pred(df[COL_PLUS])
    loss_value = np.mean(np.square(iv-df[['iv']]))
    g = grad.mean()[COL].values

    g = tf.convert_to_tensor(g)
    loss_value = tf.convert_to_tensor(loss_value)
    # grad = np.mean(np.square(iv-df[['iv']]))

    # with tf.GradientTape() as tape:
    #     tape.watch(x_params)
    #     loss_value = func(x_params)
    # grads = tape.gradient(loss_value, [x_params])
    print('---',loss_value,flush=True)
    return loss_value, g

init_x = df[COL].mean().values
x_params = init_x

s = time.time()
soln = tfp.optimizer.lbfgs_minimize(value_and_gradients_function=func_g, initial_position=init_x, max_iterations=50, tolerance=1e-60)
soln_time = np.round((time.time() - s) / 60, 2)
pred_par = soln.position.numpy()
obj_value = soln.objective_value.numpy()
print(soln,flush=True)
print(soln_time)

print('init',func(init_x))
print('final',func(pred_par))
np.save('res/pred/par.npy',pred_par)



