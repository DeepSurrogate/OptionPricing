from matplotlib import pyplot as plt
from matplotlib import cm
from source.deepsurrogate import *
from scipy.interpolate import griddata

model_name = 'heston'
model_name = 'bdjm'

model_names = ['heston', 'bdjm']
for model_name in model_names:
    df_call = pd.read_csv(f'dat/{model_name}_ex.csv', index_col=False)
    df_put = pd.read_csv(f'dat/{model_name}_ex.csv', index_col=False)

    ##################
    # using cp parity to transform call price into put price for ITM call
    ##################
    df_call['cp'] = 1.0  # the surogate class use call by default. To have put, add a columns 'cp' where 1 = call, and -1 = put
    df_call['S'] = 100.0

    df_put['cp'] = -1.0
    df_put['S'] = 100.0
    df_put.loc[:, 'price'] = df_put.loc[:, 'price'] - df_put.loc[:, 'S'] * np.exp(-df_put.loc[:, 'dividend'] * df_put.loc[:, 'T'] / 365) + df_put.loc[:, 'strike'] * np.exp(-df_put.loc[:, 'rf'] * df_put.loc[:, 'T'] / 365)

    ##################
    # load the surrogate model
    ##################
    surogate = DeepSurrogate(model_name)

    ##################
    # use the surrogate model to interpolate the model's predicted implied volatility
    ##################
    df_call['surrogate_price'] = surogate.get_price(df_call)
    df_put['surrogate_price'] = surogate.get_price(df_put)


    ##################
    # compare the volatility surface of the "original" quantlib model to that of the surrogate
    ##################

    def make_surf_plot(X, Y, Z, ax, rot_call=True):
        """
        Create a volatility surface plot
        """
        XX, YY = np.meshgrid(np.linspace(min(X), max(X), 230), np.linspace(min(Y), max(Y), 230))
        ZZ = griddata(np.array([X, Y]).T, np.array(Z), (XX, YY), method='nearest')

        ax.plot_surface(XX, YY, ZZ, cmap=cm.coolwarm, linewidth=0)
        if rot_call:
            ax.view_init(25, 45)
        else:
            # ax.view_init(25, -45)
            ax.view_init(25, 45)
        ax.set_xlabel('T')
        ax.set_ylabel('K')
        ax.set_zlabel(r'Price')
        return fig, ax


    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1, projection='3d')

    fig, ax = make_surf_plot(df_call['T'], df_call['strike'], df_call['price'], ax)
    ax.set_title("QuantLib", fontsize=12, fontweight='bold')
    ax = fig.add_subplot(1, 2, 2, projection='3d')

    fig, ax = make_surf_plot(df_call['T'], df_call['strike'], df_call['surrogate_price'], ax)
    ax.set_title("Surrogate", fontsize=12, fontweight='bold')

    t = fig.suptitle(f"{model_name.upper()} model | Call", fontsize=14, fontweight='bold')
    plt.show()

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1, projection='3d')

    fig, ax = make_surf_plot(df_put['T'], df_put['strike'], df_put['price'], ax, rot_call=False)
    ax.set_title("QuantLib", fontsize=12, fontweight='bold')

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    fig, ax = make_surf_plot(df_put['T'], df_put['strike'], df_put['surrogate_price'], ax, rot_call=False)
    ax.set_title("Surrogate", fontsize=12, fontweight='bold')

    t = fig.suptitle(f"{model_name.upper()} model | Put", fontsize=14, fontweight='bold')
    plt.show()
