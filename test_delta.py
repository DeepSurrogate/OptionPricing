from matplotlib import pyplot as plt
from matplotlib import cm
from source.deepsurrogate import *
from scipy.interpolate import griddata

model_name = 'heston'
model_name = 'bdjm'

model_names = ['heston', 'bdjm']
for model_name in model_names:
    df = pd.read_csv(f'dat/{model_name}_ex.csv', index_col=False)


    ##################
    # load the surrogate model
    ##################
    surogate = DeepSurrogate(model_name)

    ##################
    # use the surrogate model to interpolate the model's predicted implied volatility
    ##################
    df['cp']=1.0
    df['delta_call'] = surogate.get_price_delta(df,'S')
    df['cp']=-1.0
    df['delta_put'] = surogate.get_price_delta(df,'S')

    ##################
    # compare the volatility surface of the "original" quantlib model to that of the surrogate
    ##################

    def make_surf_plot(X, Y, Z, ax):
        """
        Create a volatility surface plot
        """
        XX, YY = np.meshgrid(np.linspace(min(X), max(X), 230), np.linspace(min(Y), max(Y), 230))
        ZZ = griddata(np.array([X, Y]).T, np.array(Z), (XX, YY), method='nearest')

        ax.plot_surface(XX, YY, ZZ, cmap=cm.coolwarm, linewidth=0)
        ax.view_init(25, 45)
        ax.set_xlabel('T')
        ax.set_ylabel('K',)
        return fig, ax


    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1, projection='3d')

    fig, ax = make_surf_plot(df['T'], df['strike'], df['delta_call'], ax)
    ax.set_title(r"$\Delta$ call", fontsize=12, fontweight='bold')
    ax = fig.add_subplot(1, 2, 2, projection='3d')

    fig, ax = make_surf_plot(df['T'], df['strike'], df['delta_put'], ax)
    ax.set_title(r"$\Delta$ put", fontsize=12, fontweight='bold')
    ax.invert_zaxis()
    t = fig.suptitle(f"{model_name.upper()} model", fontsize=14, fontweight='bold')
    plt.show()
