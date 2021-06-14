# ml_model
# Created by Antoine Didisheim, at 30.01.20
# job: 

import pickle
import socket
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from source.parameters import *
import os
from scipy.stats import norm

##################
# custom layer to mix parse the stat, par, and data input of the model
##################
class FirstLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs, par):
        super(FirstLayer, self).__init__(dtype='float64')
        self.num_outputs = num_outputs
        self.par = par

        c = []
        y = ['iv']
        state = ['v0']
        opt_data = self.par.data.cross_vary_list
        for cc in self.par.process.__dict__.keys():
            if (cc not in state) & (cc not in opt_data):
                c.append(cc)
        self.l1 = len(c)
        self.l2 = len(state)
        self.l3 = len(opt_data)

    def build(self, input_shape):
        # print(input_shape[2][-1])
        self.kernel_par = self.add_weight("kernel_par",
                                          shape=[self.l1,
                                                 self.num_outputs], dtype=tf.float64)
        self.kernel_state = self.add_weight("kernel_state",
                                            shape=[self.l2,
                                                   self.num_outputs], dtype=tf.float64)
        self.kernel_data = self.add_weight("kernel_data",
                                           shape=[self.l3,
                                                  self.num_outputs], dtype=tf.float64)

        # self.kernel_par = self.add_weight("kernel_par",
        #                                   shape=[self.num_outputs, self.l1], dtype=tf.float64)
        # self.kernel_state = self.add_weight("kernel_state",
        #                                     shape=[self.num_outputs, self.l2], dtype=tf.float64)
        # self.kernel_data = self.add_weight("kernel_data",
        #                                    shape=[self.num_outputs,self.l3], dtype=tf.float64)

    def call(self, input):
        r = tf.matmul(input[0], self.kernel_par) + tf.matmul(input[1], self.kernel_state) + tf.matmul(input[2], self.kernel_data)
        # r = tf.matmul(tf.transpose(input[0]), self.kernel_par)+tf.matmul(tf.transpose(input[1]), self.kernel_state)+tf.matmul(tf.transpose(input[2]), self.kernel_data)
        r = tf.nn.swish(r)
        return r


class NetworkModel:
    def __init__(self, par: Params()):
        self.par = par
        self.model = None
        # self.gp = neural_network.MLPRegressor(hidden_layer_sizes=(100,))
        self.history_training = pd.DataFrame(data={'nope': [1, 2, 3]})

        self.m = None
        self.std = None
        self.m_y = None
        self.std_y = None

        if socket.gethostname() in ['work', 'workstation']:
            self.par.model.save_dir = self.par.model.save_dir.replace('/scratch/snx3000/adidishe/fop/', './')

        self.save_dir = self.par.model.save_dir + '/' + self.par.name


        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def normalize(self, X=None, y=None):
        if self.par.model.normalize:
            if X is not None:
                X = (X - self.m) / self.std
            if y is not None:
                pass
                # y = (y-self.m_y)/self.std_y
        return X, y

    def unnormalize(self, X=None, y=None):
        if self.par.model.normalize:
            if X is not None:
                if self.par.model.normalize_range:
                    X = X * self.x_range
                else:
                    X = (X * self.std) + self.m
            if y is not None:
                if type(y) == np.ndarray:
                    y = y * self.std_y.values + self.m_y.values
                else:
                    y = (y * self.std_y) + self.m_y
                # y = y+self.m_y

        return X, y

    def split_state_data_par(self, df):

        state = df[['v0']]

        opt_data = df[['strike', 'T', 'rf', 'dividend']]
        c = []
        for cc in self.par.process.__dict__.keys():
            # for cc in df.columns:
            if (cc not in state.columns) & (cc not in opt_data.columns):
                c.append(cc)

        par_est = df[c]

        # to put back if the __dict__ trick above does not do the trick
        # if self.par.opt.process == Process.DOUBLE_EXP:
        #     # print('double exp')
        #     par_est = par_est[['dividend', 'kappa', 'lambda_parameter', 'nuDown', 'nuUp', 'p', 'rho', 'sigma', 'theta']]
        # else:
        #     par_est = par_est[['dividend','kappa', 'rho', 'sigma', 'theta']]
        #     # print('COLUMNS', par_est.columns)

        return [par_est, state, opt_data]

    def train(self):
        ##################
        # first get the data_split col size
        ##################
        c = []
        y = ['iv']
        state = ['v0']
        opt_data = self.par.data.cross_vary_list
        for cc in self.par.process.__dict__.keys():
            if (cc not in state) & (cc not in opt_data):
                c.append(cc)
        c1 = len(c)
        c2 = len(c) + len(state)
        c3 = len(c) + len(state) + len(opt_data)

        # in training we both look at mean and std for future normalization and norm the intput
        # self.m = X.mean()
        # self.std = X.std() + 0.0000001

        d = self.par.process.__dict__
        m = {}
        std = {}
        for i, k in enumerate(d):
            m[k] = np.mean(d[k])
            std[k] = (((max(d[k]) - min(d[k])) ** 2) / 12) ** (1 / 2)
        self.m = pd.Series(m)
        self.std = pd.Series(std)

        ##################
        # prepare data sets
        ##################
        def tr(x):
            y = (((x[:c1], x[c1:c2], x[c2:c3]), x[c3:]))
            return y

        if self.par.opt.option_type == OptionType.CALL_EUR:
            data_dir = self.par.data.path_sim_save + '/call'
        else:
            data_dir = self.par.data.path_sim_save + '/put'

        train_list = [data_dir + '/' + x for x in os.listdir(data_dir) if 'test' not in x]
        # train_list = [data_dir +'/'+x for x in os.listdir(data_dir) if 'test' not in x]
        test_list = [data_dir + '/' + x for x in os.listdir(data_dir) if 'test' in x]
        # data_train = tf.data.TFRecordDataset(train_list,buffer_size=self.par.model.batch_size)
        data_train = tf.data.TFRecordDataset(train_list, num_parallel_reads=tf.data.experimental.AUTOTUNE)
        data_train = data_train.map(lambda x: tf.ensure_shape(tf.io.parse_tensor(x, tf.float64), (c3 + 1)), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        data_train = data_train.map(lambda x: tr(x), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        data_train = data_train.batch(batch_size=self.par.model.batch_size * 1)
        #
        # for d in data_train:
        #     print(f'\n{d}')

        # data_train = data_train.apply(lambda x: tf.io.parse_tensor(x, tf.float64))

        # i = 0
        # for x in data_train.take(1000):
        #     i+=1
        #     print(x)

        data_test = tf.data.TFRecordDataset(test_list, num_parallel_reads=tf.data.experimental.AUTOTUNE)
        data_test = data_test.map(lambda x: tf.ensure_shape(tf.io.parse_tensor(x, tf.float64), (c3 + 1)), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        data_test = data_test.map(lambda x: tr(x), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        data_test = data_test.batch(batch_size=self.par.model.batch_size)

        self.save()
        if self.model is None:
            self.create_nnet_model()

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.save_dir + '/', save_weights_only=True, verbose=0, save_best_only=True)
        print('start training for', self.par.model.E, 'epochs', flush=True)
        self.history_training = self.model.fit(x=data_train, epochs=self.par.model.E, validation_data=data_test, callbacks=[cp_callback], verbose=1)  # Pass callback to training
        # self.history_training = self.model.fit(x=data_train, epochs=self.par.model.E, callbacks=[cp_callback], verbose=1)  # Pass callback to training

        self.history_training = pd.DataFrame(self.history_training.history)


    def score(self, X, y):
        X, y = self.normalize(X, y)
        X = self.split_state_data_par(X)
        loss, mae, mse, r2 = self.model.evaluate(X, y, verbose=0)
        return r2, mae, mse




    def _wip_get_grad_price(self, X):
        # extract bs input
        K = X['strike'].values
        # S_old = X['S'].values
        # S = np.full_like(K,100.0)
        S = X['S'].values
        sig = self.predict(X).flatten()
        T = (X['T'].values) / 365
        div = X['dividend'].values
        rf = X['rf'].values

        d = self.get_grad_iv(X)

        # get vega
        # d1 = (1 / (sig * np.sqrt(T))) * (
        #         np.log(100 / K) + (rf - div + ((sig ** 2) / 2)) * T
        # )
        # d2  = d1 - sig*np.sqrt(T)
        d1 = (1 / (sig * np.sqrt(T))) * (
                np.log(100 / K) + (rf - div + (sig ** 2) / 2) * T
        )
        d2 = d1 - sig * np.sqrt(T)


        vega = 100*norm.cdf(d1)*np.sqrt(T)

        # do s


        dk = -(100.0 * K) / (100 ** 2)
        d['S'] = vega * d['strike'] * dk
        delta = norm.pdf(d1)
        delta[K < S] = delta[K < S] - 1
        d['S'] += delta

        ## do all but s
        for c in d.columns:
            if c != 'S':
                d[c] = vega * d[c]
                if c =='strike':
                    dk = -100 * K / (100 ** 2)
                    delta = norm.pdf(d1)
                    delta[K < S] = delta[K < S] - 1
                    d[c]=-d[c]*dk+delta
                if c == 'T':
                    theta = - (S*norm.pdf(d1)*sig)/(2*np.sqrt(T)) - rf*K*np.exp(-rf*T)*norm.cdf(d2)
                    theta_neg = - (S*norm.pdf(d1)*sig)/(2*np.sqrt(T)) - rf*K*np.exp(-rf*T)*norm.cdf(-d2)
                    theta[K < S] = theta_neg[K < S]
                    theta = theta
                    d[c]+=-theta


                if c == 'rf':
                    dRf = K*T*np.exp(-rf*T)*norm.cdf(d2)
                    dRf_neg = -K*T*np.exp(-rf*T)*norm.cdf(-d2)
                    dRf[K < S] = dRf_neg[K < S]
                    d[c]+=dRf


        return d

    def get_grad_iv(self, X):
        # extract bs input
        K = X['strike'].values
        S = X['S'].values
        sig = self.predict(X).flatten()
        T = (X['T'].values) / 365
        div = X['dividend'].values
        rf = X['rf'].values


        X, y = self.normalize(X, y=None)
        X = self.split_state_data_par(X)

        xx = [tf.convert_to_tensor(x.values) for x in X]
        with tf.GradientTape(persistent=True) as g:
            g.watch(xx[0])
            g.watch(xx[1])
            g.watch(xx[2])
            pred = self.model(xx)
        d = g.gradient(pred, xx[0])
        d1 = g.gradient(pred, xx[1])
        d2 = g.gradient(pred, xx[2])
        del g

        d = np.concatenate([d.numpy(), d1.numpy(), d2.numpy()], axis=1)
        col = list(X[0].columns) + list(X[1].columns) + list(X[2].columns)

        d = pd.DataFrame(d, columns=col, index=X[0].index)
        d = d*(1/self.std)

        kk = K * S / 100
        d['S'] = d['strike']*(-(kk*100)/(S**2))




        return d


    def get_price(self, X):
        # get bs input
        sig = self.predict(X).flatten()
        T = (X['T'].values) / 365
        div = X['dividend'].values
        K = X['strike'].values
        rf = X['rf'].values
        S = 100.0


        # get price
        d1 = (1 / (sig * np.sqrt(T))) * (
                np.log(100 / K) + (rf - div + (sig ** 2) / 2) * T
        )
        d2 = d1 - sig * np.sqrt(T)
        F = 100 * np.exp((rf - div) * T)
        p = np.exp(-rf * T) * (norm.cdf(d1) * F - norm.cdf(d2) * K)



        # df.loc[ind, 'mid_p'] = df.loc[ind, 'S'] * np.exp(-df.loc[ind, 'dividend'] * df.loc[ind, 'T'] / 365) + df.loc[ind, 'mid_p'] - (df.loc[ind, 'strike_un'] / 1000) * np.exp(-df.loc[ind, 'rf'] * df.loc[ind, 'T'] / 365)
        if 'cp' in X.columns:
            ind = X['cp']==-1
            ind = np.where(ind)
            p[ind] = p[ind] - S * np.exp(-div[ind] * T[ind]) + K[ind] * np.exp(-rf[ind] * T[ind])


        return p

    def predict(self, X, vol=False):
        X, y = self.normalize(X, y=None)
        X = self.split_state_data_par(X)
        pred = self.model.predict(X)
        return pred

    def save(self, other_save_dir=None):
        self.par.save(save_dir=self.save_dir)

        with open(self.save_dir + '/m' + '.p', 'wb') as handle:
            pickle.dump(self.m, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(self.save_dir + '/std' + '.p', 'wb') as handle:
            pickle.dump(self.std, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(self.save_dir + '/m_y' + '.p', 'wb') as handle:
            pickle.dump(self.m_y, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.save_dir + '/std_y' + '.p', 'wb') as handle:
            pickle.dump(self.std_y, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.history_training.to_pickle(self.save_dir + '/history.p')

    def load(self, n, other_save_dir=None):

        self.par.name = n
        if other_save_dir is None:
            temp_dir = self.par.model.save_dir + '' + self.par.name
        else:
            temp_dir = other_save_dir + '' + self.par.name

        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        par = Params()
        par.load(load_dir=temp_dir)
        self.par = par

        with open(temp_dir + '/m' + '.p', 'rb') as handle:
            self.m = pickle.load(handle)

        with open(temp_dir + '/std' + '.p', 'rb') as handle:
            self.std = pickle.load(handle)

        with open(temp_dir + '/m_y' + '.p', 'rb') as handle:
            self.m_y = pickle.load(handle)
        with open(temp_dir + '/std_y' + '.p', 'rb') as handle:
            self.std_y = pickle.load(handle)

        self.history_training = pd.read_pickle(self.save_dir + '/history.p')

        if self.model is None:
            self.create_nnet_model()
        self.model.load_weights(self.save_dir + '/')


    def create_nnet_model(self):
        L = []
        # self = Params()
        # self.par = Params()

        for i, l in enumerate(self.par.model.layers):
            if i == 0:
                L.append(FirstLayer(l, self.par))
            else:
                L.append(layers.Dense(l, activation=self.par.model.activation, dtype=tf.float64))
        L.append(layers.Dense(1, dtype=tf.float64))
        self.model = keras.Sequential(L)

        # optimizer = tf.keras.optimizers.RMSprop(0.05)
        if self.par.model.opti == Optimizer.SGD:
            optimizer = tf.keras.optimizers.SGD(self.par.model.learning_rate)
        if self.par.model.opti == Optimizer.RMS_PROP:
            optimizer = tf.keras.optimizers.RMSprop(self.par.model.learning_rate)
        if self.par.model.opti == Optimizer.ADAM:
            optimizer = tf.keras.optimizers.Adam(self.par.model.learning_rate)
        if self.par.model.opti == Optimizer.NADAM:
            optimizer = tf.keras.optimizers.Nadam(self.par.model.learning_rate)
        if self.par.model.opti == Optimizer.ADAMAX:
            optimizer = tf.keras.optimizers.Adamax(self.par.model.learning_rate)
        if self.par.model.opti == Optimizer.ADAGRAD:
            optimizer = tf.keras.optimizers.Adamax(self.par.model.learning_rate)

        # optimizer = tf.keras.optimizers.Adam(0.00005/2)

        def r_square(y_true, y_pred):
            SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
            SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
            return (1 - SS_res / (SS_tot + tf.keras.backend.epsilon()))

        if self.par.model.loss == Loss.MAE:
            self.model.compile(loss='mae', optimizer=optimizer, metrics=['mae', 'mse', r_square])
        if self.par.model.loss == Loss.MSE:
            self.model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse', r_square])
