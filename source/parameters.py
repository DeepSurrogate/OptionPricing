# parameters
# Created by Antoine Didisheim, at 06.08.19
# job: store default parameters used throughout the projects in single .py

import itertools
from enum import Enum
import numpy as np
import pandas as pd

import sys
sys.path.append('./source/')


##################
# params Enum
##################

class Optimizer(Enum):
    SGD = 1
    SGS_DECAY = 2
    ADAM = 3
    RMS_PROP = 4
    ADAMAX = 5
    NADAM = 5
    ADAGRAD = 5


class OptionType(Enum):
    CALL_EUR = 1
    PUT_EUR = 2


class Process(Enum):
    GBM = 1
    HESTON_MODEL = 2
    BATES = 3
    DOUBLE_EXP = 4


class Sampling(Enum):
    RANDOM = 1
    LATIN_HYPERCUBE = 2
    SPARSE_GRID = 3
    GRID_10 = 4
    GIRD_20 = 5
    GRID_LOCALP = 6
    GRID_ZERO = 7
    GRID_15 = 8
    GRID_20_LOCALP = 9
    GRID_20_ZERO = 10
    GRID_30 = 30
    GRID_50 = 12
    GRID_50_bis = 13


class Loss(Enum):
    MSE = 1
    MAE = 2

class CostFunction(Enum):
    VANILLA = 1
    VOL_LASSO = 2


##################
# params classes
##################


class ParamsProcess:
    def draw_values(self, nb=1, smaller_range=False, mean_value=False):
        d = self.__dict__.copy()
        for k in d.keys():
            if smaller_range:
                min_ = d[k][0]
                max_ = d[k][1]
                delt = max_ - min_
                min_ = min_ + delt * 0.1
                max_ = max_ - delt * 0.1
            else:
                min_ = d[k][0]
                max_ = d[k][1]

            if (type(d[k][0]) == int) & (type(d[k][1]) == int):
                d[k] = np.random.randint(int(np.ceil(min_)), int(np.ceil(max_)) + 1, nb)
            else:
                if mean_value:
                    d[k] = np.array([(min_ + max_) / 2])
                else:
                    d[k] = np.random.uniform(min_, max_, nb)

        return d


class ParamsGBM(ParamsProcess):
    def __init__(self):
        self.strike = [60.0, 140.0] # the strike value here is important only to define the m and std to normalize the input
        self.rf = [0.0, 0.075]
        self.dividend = [0.0, 0.05]
        self.v0 = [0.01, 0.9]
        self.T = [1, 380]

class ParamsHeston(ParamsGBM):
    def __init__(self):
        super().__init__()
        self.kappa = [0.1, 50.0]
        self.theta = [0.01, 0.9]
        self.sigma = [0.1, 5.0]  # vol of vol
        self.rho = [-1.0, -0.0]


class ParamsDoubleExp(ParamsHeston):
    def __init__(self):
        super().__init__()
        self.lambda_parameter = [0.00, 4.0]
        self.nuUp = [0.0, 0.4]
        self.nuDown = [0.0, 0.4]
        self.p = [0.0, 1.0]


class ParamsOption:
    def __init__(self):
        self.option_type = OptionType.PUT_EUR
        self.process = Process.HESTON_MODEL


class ParamsPricer:
    def __init__(self):
        self.save_name = ''
        self.fit_process = Process.BATES
        self.cost_function = CostFunction.VANILLA
        self.lbda = 0.05
        self.nb_init = 30


        self.best_heston = 'PARITY_M5_CALLEURLayer_5L400_swish_Lr0_0001_ADAMoMAE_BATCH_256tr_size_1CM'
        self.best_bate = 'PARITY_M5_CALLEURLayer_7L400_swish_Lr5e_05_ADAMoMAE_BATCH_256tr_size_30CM'

class ParamsData:
    def __init__(self):
        self.path_sim_save = './records'
        self.train_size = 3
        self.test_size = 10000
        self.cross_vary_list = ['strike', 'T', 'rf', 'dividend']
        self.bsiv_vd = True
        self.parallel = False
        self.samplingTraing = Sampling.RANDOM
        self.batch_system = -1


class ParamsBayesianActiveLearning:
    def __init__(self):
        self.sigma_m = 1
        self.sigma_v = 100
        self.N = 25 * 1000
        self.xi = 100
        self.nb_round = 60
        self.is_bae = True


class ParamsModels:
    def __init__(self):
        # self.kernel = RBF(length_scale=100)
        self.save_dir = './model_save/'
        self.res_dir = './res/'

        # self.kernel = RBF(10, (1e-2, 1e2))
        self.kernel_name = 'mattern'
        self.active_subspace = -1
        self.normalize = True
        self.normalize_range = False
        # self.model_type = 'deep'
        self.model_type = 'nnet'
        self.E = 10

        self.layers = [64, 32, 16]
        self.batch_size = 512
        self.activation = 'swish'
        self.opti = Optimizer.ADAM
        self.loss = Loss.MSE
        self.learning_rate = 0.001


# store all parameters into a single object
class Params:
    def __init__(self):
        self.name_detail = ''
        self.name = ''
        self.seed = 12345
        self.model = ParamsModels()
        self.opt = ParamsOption()
        self.data = ParamsData()
        self.bae = ParamsBayesianActiveLearning()
        self.pricer = ParamsPricer()

        self.process = None
        self.update_process()
        self.update_model_name()

    def update_process(self, process=None):
        if process is not None:
            self.opt.process = process
        if self.opt.process.name == Process.HESTON_MODEL.name:
            self.process = ParamsHeston()
        if self.opt.process.name == Process.GBM.name:
            self.process = ParamsGBM()
        if self.opt.process.name == Process.DOUBLE_EXP.name:
            self.process = ParamsDoubleExp()

    def update_model_name(self):
        n = self.name_detail + self.opt.option_type.name.replace('_', '')
        n += 'Layer_'
        # for l in self.model.layers:
        #     n = n + str(l)+'_'
        if np.all(self.model.layers[0] == np.array(self.model.layers)):
            n = n + str(len(self.model.layers)) + 'L' + str(self.model.layers[0]) + '_'
        else:
            for l in self.model.layers:
                n = n + str(l) + '_'

        n += self.model.activation + '_Lr'
        n += str(self.model.learning_rate) + '_'
        # n += str(self.model.opti)+'_'

        n += self.model.opti.name + 'o' + self.model.loss.name + '_'
        n += 'BATCH_' + str(self.model.batch_size)

        n = n + 'tr_size_' + str(self.data.train_size) + 'CM'
        # for k in self.process.__dict__.keys():
        #     n = n + str(k) + str(self.process.__dict__[k][0]) + str(self.process.__dict__[k][1]) + '_'
        # n = n + 'tr_size_' + str(self.data.train_size) + '_te_size_' + str(self.data.test_size)

        n = n.replace('.', '_')
        n = n.replace('-', '_')
        self.name = n

    def print_values(self):
        """
        Print all parameters used in the model
        """
        for key, v in self.__dict__.items():
            try:
                print('########', key, '########')
                for key2, vv in v.__dict__.items():
                    print(key2, ':', vv)
            except:
                print(v)

    def update_param_grid(self, grid_list, id_comb):
        ind = []
        for l in grid_list:
            t = np.arange(0, len(l[2]))
            ind.append(t.tolist())
        combs = list(itertools.product(*ind))
        print('comb', str(id_comb + 1), '/', str(len(combs)))
        c = combs[id_comb]

        for i, l in enumerate(grid_list):
            self.__dict__[l[0]].__dict__[l[1]] = l[2][c[i]]

    def save(self, save_dir, file_name='/parameters.p'):
        # simple save function that allows loading of deprecated parameters object
        df = pd.DataFrame(columns=['key', 'value'])

        for key, v in self.__dict__.items():
            try:
                for key2, vv in v.__dict__.items():
                    temp = pd.DataFrame(data=[str(key) + '_' + str(key2), vv], index=['key', 'value']).T
                    df = df.append(temp)

            except:
                temp = pd.DataFrame(data=[key, v], index=['key', 'value']).T
                df = df.append(temp)
        df.to_pickle(save_dir + file_name,protocol=4)

    def load(self, load_dir, file_name='/parameters.p'):
        # simple load function that allows loading of deprecated parameters object
        df = pd.read_pickle(load_dir + file_name)
        # First check if this is an old pickle version, if so transform it into a df
        if type(df) != pd.DataFrame:
            loaded_par = df
            df = pd.DataFrame(columns=['key', 'value'])
            for key, v in loaded_par.__dict__.items():
                try:
                    for key2, vv in v.__dict__.items():
                        temp = pd.DataFrame(data=[str(key) + '_' + str(key2), vv], index=['key', 'value']).T
                        df = df.append(temp)

                except:
                    temp = pd.DataFrame(data=[key, v], index=['key', 'value']).T
                    df = df.append(temp)

        no_old_version_bug = True

        for key, v in self.__dict__.items():
            try:
                for key2, vv in v.__dict__.items():
                    t = df.loc[df['key'] == str(key) + '_' + str(key2), 'value']
                    if t.shape[0] == 1:
                        tt = t.values[0]
                        self.__dict__[key].__dict__[key2] = tt
                    else:
                        if no_old_version_bug:
                            no_old_version_bug = False
                            # print('#### Loaded parameters object is depreceated, default version will be used')
                        # print('Parameter', str(key) + '.' + str(key2), 'not found, using default: ',
                        #       self.__dict__[key].__dict__[key2])

            except:
                t = df.loc[df['key'] == str(key), 'value']
                if t.shape[0] == 1:
                    tt = t.values[0]
                    self.__dict__[key] = tt
                else:
                    if no_old_version_bug:
                        no_old_version_bug = False
                    #     print('#### Loaded parameters object is depreceated, default version will be used')
                    # print('Parameter', str(key), 'not found, using default: ', self.__dict__[key])

        self.update_process()

        # to ensure backward compatibility, we update her the cross_vary_list
        self.data.cross_vary_list = ['strike', 'T', 'dividend', 'rf']
