import tensorflow as tf
import logging

tf.get_logger().setLevel(logging.ERROR)
from source.ml_model import NetworkModel
from source.parameters import *
import warnings
model_name = 'bdjm'


class DeepSurrogate:
    def __init__(self, model_name='heston', NB_DAYS=1, model_names=[]):
        model_list = ['heston', 'bdjm']
        self.model_name = model_name
        assert model_name in model_list, "Invalid model name, please select on of: ['heston','bdjm']"

        # define the name of the model to load
        if model_name == 'heston':
            call_model_name = Params().pricer.best_heston
        if model_name == 'bdjm':
            call_model_name = Params().pricer.best_bate

        par_c = Params()
        par_c.load(par_c.model.save_dir + call_model_name)
        par_c.update_process()
        self.par = par_c

        self.c_model = NetworkModel(par_c)
        self.c_model.load(par_c.name)

        self.first_time_numpy = True

        self.f = 1.0  # factor to standardize price K/S transform

    def get_iv(self, X):

        X = self.pre_process_X(X)
        return self.c_model.predict(X)

    def get_price(self, X):
        X = self.pre_process_X(X)
        return self.c_model.get_price(X) / self.f

    def get_iv_delta(self, X):
        X = self.pre_process_X(X)
        return self.c_model.get_grad_iv(X)

    def get_price_delta(self, X, var):
        if 'S' not in X.columns:
            X['S']=100.0

        h = 0.001
        X = X.copy()
        X[f'{var}'] += h
        p1 = self.get_price(X)
        X[f'{var}'] -= 2 * h
        p2 = self.get_price(X)

        return (p1 - p2) / (2 * h)

    def pre_process_X(self, X):
        X = X.copy()
        col = []
        for cc in self.par.process.__dict__.keys():
            col.append(cc)
        col.append('S')
        if (type(X) == np.array) & self.first_time_numpy:
            print(f'If you use a numpy dataset for a surrogate of the model {self.model_name}, please make sure that your columns respect the following order: ')
            print(col)
            print('The spot price "S", is optional.')
            print('If you use pandas with the correct columns name, the order of the columns does not matter')
            if X.shape[1] == len(col):
                X = pd.DataFrame(X, columns=col)
            elif X.shape[1] == len(col) - 1:
                col.remove('S')
                X = pd.DataFrame(X, columns=col)
            else:
                assert False, 'wrong number of columns'

        if 'S' not in X.columns:
            print('No S, we assume stock price = 100 and strike is normalize')
            X['S'] = 100.0
        else:
            self.f = 100.0 / X['S']
            X['strike'] = X['strike'] * self.f

        if (type(X) == pd.DataFrame):
            assert np.all([(x in X.columns) for x in col]), f'to use a DSE of the model {self.model_name} with a pandas input, your input should contain a columns for each of the states and parameters {col}'

        self.check_input_range(X)

        return X

    def check_input_range(self, X):
        check_dict = {'T':[1,380], 'rf':[0.0,0.75], 'dividend':[0.0,0.05], 'v0':[0.01,0.9],
         'kappa':[0.1,50.0], 'sigma':[0.1,5.0], 'rho':[-1.0,0.0], 'theta':[0.01,0.9], 'lambda_parameter':[0.05,4.0],
         'nuUp':[0.0,0.4], 'nuDown':[0.0,0.4], 'p':[0.0,1.0]}
        raise_warning =False
        warning_text = "\n WARNING: some of your input are outside the surrogate's training range: \n"
        for i, k in enumerate(check_dict):
            if k in X.columns:
                if np.all((X[k] <= check_dict[k][1]) & (X[k] >= check_dict[k][0])):
                    pass
                else:
                    warning_text+= f"    - {k} ouside the surrogate range {check_dict[k]} \n"
                    raise_warning = True
        if raise_warning:
            warnings.warn(warning_text)



##################
# plot standardizing
##################
import pylab

params = {'axes.labelsize': 14,
          'axes.labelweight': 'bold',
          'xtick.labelsize': 12,
          'ytick.labelsize': 12,
          'axes.titlesize': 12}
pylab.rcParams.update(params)

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
