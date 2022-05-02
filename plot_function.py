# plot_function
# Created by Antoine Didisheim, at 23.08.19
# job: 

import matplotlib.pylab as pylab
import pandas as pd

params = {'axes.labelsize': 14,
          'axes.labelweight': 'bold',
          'xtick.labelsize': 12,
          'ytick.labelsize': 12,
          'axes.titlesize': 12}
pylab.rcParams.update(params)



pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class DidiPlot:
    LINE_STYLE = [
        '-', '--', ':', '-.','-'
    ]
    # COLOR = [
    #     'black', 'gray', 'black', 'dimgray'
    # ]

    COLOR = [
        'black', 'blue', 'green', 'red','cyan'
    ]

    C_DICT = {'kappa': r'$\kappa$', 'sigma': r'$\sigma$', 'rho': r'$\rho$', 'theta': r'$\theta$', 'lambda_parameter': r'$\lambda$', 'nuUp': r'$\nu_1$', 'nuDown': r'$\nu_2$',
              'p': 'p', 'v0': r'$v_t$', 'strike': '$\hat{K}$', 'T': '$T$', 'rf': r'$rf$','dividend':'dividend'}
