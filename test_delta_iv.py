from matplotlib import pyplot as plt
from matplotlib import cm
from source.deepsurrogate import *
from scipy.interpolate import griddata



model_name = 'heston'
model_name = 'bdjm'

model_names = ['heston','bdjm']
for model_name in model_names:

   df = pd.read_csv(f'dat/{model_name}_ex.csv',index_col=False)

   ##################
   # load the surrogate model
   ##################
   surogate = DeepSurrogate(model_name)

   ##################
   # use the surrogate model to interpolate the model's predicted implied volatility
   ##################
   d = surogate.get_iv_delta(df)

   ##################
   # compare the volatility surface of the "original" quantlib model to that of the surrogate
   ##################

   def make_surf_plot(X,Y,Z,ax):
      XX,YY = np.meshgrid(np.linspace(min(X),max(X),230),np.linspace(min(Y),max(Y),230))
      ZZ = griddata(np.array([X,Y]).T,np.array(Z),(XX,YY), method='nearest')

      ax.plot_surface(XX, YY, ZZ, cmap=cm.coolwarm,linewidth=0)
      ax.view_init(25, 45)
      ax.set_xlabel('T')
      ax.set_ylabel('K')
      ax.set_zlabel(r'$\partial \sigma_{implied}$')
      return fig,ax

   nb_col = 3
   nb_row = int(np.ceil(d.shape[1] / nb_col))

   fig = plt.figure(figsize=[6.4 * nb_col, 4.8*nb_row])

   dict_tr = {'kappa': r'$\kappa$', 'theta': r'$\theta$', 'sigma': r'$\sigma$', 'rho': r'$\rho$', 'lambda_parameter': r'$\lambda$', 'nuUp': r'$\nu_1$',
              'nuDown': r'$\nu_1$', 'p': r'$p$', 'v0': r'$v_t$', 'strike': r'$\hat{K}$', 'T': r'$T$', 'rf': r'$r$', 'dividend': 'd','S':'S'}

   for i in range(d.shape[1]):
      ax = fig.add_subplot(nb_row, nb_col, i + 1, projection='3d')
      fig,ax = make_surf_plot(df['T'], df['strike'], d.iloc[:,i],ax)
      ax.set_title(dict_tr[d.columns[i]], fontsize=12, fontweight='bold')

   plt.show()

