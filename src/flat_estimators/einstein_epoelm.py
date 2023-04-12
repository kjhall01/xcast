from scipy.special import expit, logit
import datetime as dt
import numpy as np
import scipy.linalg.lapack as la
from collections.abc import Iterable
import copy
import matplotlib.pyplot as plt
import dask

def find_b(x, y):
    if x.dtype == np.float64 and y.dtype == np.float64:
        _, B, info = la.dposv(x, y)
    elif x.dtype == np.float32 and y.dtype == np.float32:
        _, B, info = la.sposv(x, y)
    else:
        assert False, 'x: {} and y: {} not matching or good dtypes for lapack'.format(
            x.dtype, y.dtype)
    if info > 0:
        print('info > 0!')
        x = x + np.triu(x, k=1)
        B = np.linalg.lstsq(x, y, rcond=None)[0]
    return B

class epoelm:
    """Ensemble Extreme Learning Machine using Einstein Notation"""

    def __init__(self, encoding='nonexceedance', initialization='normal', activation='relu', hidden_layer_size=5, regularization=0, regularize_lambda=True, preprocessing='minmax', n_estimators=25, eps=np.finfo('float').eps, activations=None, quantiles=[0.2, 0.4, 0.6, 0.8], standardize_y=False, save_y=True):
        assert isinstance(hidden_layer_size, int) and hidden_layer_size > 0, 'Invalid hidden_layer_size {}'.format(hidden_layer_size)
        #assert type(c) is int, 'Invalid C {}'.format(c)
        assert type(preprocessing) is str and preprocessing in ['std', 'minmax', 'none'], 'Invalid preprocessing {}'.format(preprocessing)
        self.activation = activation
        #self.c = c
        self.save_y = save_y
        self.quantiles=quantiles
        self.regularize_lambda = regularize_lambda
        self.encoding=encoding
        assert encoding.lower() in ['nonexceedance', 'binary'], 'invalid encoding for epoelm - must be "nonexceedance" or "binary"'
        self.hidden_layer_size = hidden_layer_size
        self.preprocessing = preprocessing
        self.initialization = initialization
        self.n_estimators=n_estimators
        self.eps2 = eps
        self.eps=0.01
        #regularizations = [0, 0.000976562, 0.0078125, 0.0625, 0.25, 0.5, 1, 4, 16, 256, 1028  ]
        #regularization = max( regularization, 5)
        self.regularization = 2**regularization if regularization is not None else None  #regularizations[regularization] if r
        self.activations = {
            'sigm': expit,
            'tanh': np.tanh,
            'relu': lambda ret: np.maximum(0, ret),
            'lin': lambda ret: ret,
            'softplus': lambda ret: np.logaddexp(ret, 0),
            'leaky': lambda ret: np.where(ret > 0, ret, 0.1*ret ),
            'elu': lambda ret: np.where(ret > 0, ret, 0.1* (np.exp(ret) - 1) ),
        } if activations is None else activations
        self.standardize_y = standardize_y
        assert activation in self.activations.keys(), 'invalid activation function {}'.format(activation)

    def set_params(self, **params):
        for key in params.keys():
            setattr(self, key, params[key])
        return self

    def get_params(self, deep=False):
        return vars(self)

    def fit(self, x, y):
        if self.n_estimators == 0:
            self.n_estimators = 30
        x, y = x.astype(np.float64), y.astype(np.float64)
        # first, take care of preprocessing
        if self.preprocessing == 'std':
            self.mean, self.std = x.mean(axis=0), x.std(axis=0)
            x = (x - self.mean) / self.std  # scales to std normal dist

        if self.preprocessing == 'minmax':
            self.min, self.max = x.min(axis=0), x.max(axis=0)
            x = ((x - self.min) / (self.max - self.min)) * 2 - 1  # scales to [-1, 1]

        self.ymean = y.mean()
        self.ystd = y.std()
        self.quantile_of_mean = (y < self.ymean).mean()
        if self.ystd <= np.finfo(float).eps**19:
            print(y)
        assert self.ystd > np.finfo(float).eps**19, 'standard deviation of y is too close to zero! ... use a drymask? {}'.format(y)

        if self.standardize_y:
            y = ( y - self.ymean) / self.ystd

        x_features, y_features = x.shape[1], y.shape[1]
        if self.initialization == 'normal':
            self.w = np.random.randn(self.n_estimators, x_features, self.hidden_layer_size)
            self.b = np.random.randn(self.n_estimators, 1, self.hidden_layer_size)
        elif self.initialization == 'uniform':
            self.w = np.random.rand(self.n_estimators*x_features*self.hidden_layer_size).reshape(self.n_estimators, x_features, self.hidden_layer_size) * 2 - 1
            self.b = np.random.rand(self.n_estimators*self.hidden_layer_size).reshape(self.n_estimators, 1, self.hidden_layer_size) * 2 - 1
        else:
            self.w = np.random.rand(self.n_estimators*x_features*self.hidden_layer_size).reshape(self.n_estimators, x_features, self.hidden_layer_size) * (2 / np.sqrt(x_features)) - (1 / np.sqrt(x_features))
            self.b = np.random.rand(self.n_estimators*self.hidden_layer_size).reshape(self.n_estimators, 1, self.hidden_layer_size) * (2 / np.sqrt(x_features)) - (1 / np.sqrt(x_features))

        act = np.einsum('ij,kjn->kin', x, self.w) + self.b
        h = self.activations[self.activation]( act )
        qtemplate = np.ones_like(h[:,:, 0], dtype=float)

        if self.save_y:
            self.y = y.copy()

        quantiles = [ np.nanquantile(y, i, method='midpoint') for i in self.quantiles ]
        self.iqr = quantiles[-1] - quantiles[0]
        self.q33, self.q67 = np.nanquantile(y, 1.0/3.0, method='midpoint'), np.nanquantile(y, 2.0/3.0, method='midpoint')
        #thresholds = [ ( i - self.ymean ) / self.ystd for i in quantiles] # need to check that ystd > 0!!!!
        thresholds = quantiles

        hqs, ts = [], []
        for i, q in enumerate(thresholds):
            hq = np.concatenate( [h, np.expand_dims(qtemplate * q, axis=-1) ], axis=2)
            hqs.append(hq)
            if self.encoding.lower() == 'binary':
                t = np.zeros_like(y, dtype=float)
                t[y <= quantiles[i]] = 1
            else:
                t =  quantiles[i] - y #/ (y.max() - y.min()))
            ts.append(t)
        hqs = np.concatenate(hqs, axis=1)
        ts = np.vstack(ts)

        if self.encoding.lower() == 'binary':
            ts = np.abs(ts - self.eps2)
            logs = logit(ts)
        else:
            logs = ts

        qhth = np.stack([hqs[i, :, :].T.dot(hqs[i,:,:]) for i in range(self.n_estimators)], axis=0)
        qeye = np.zeros(qhth.shape)
        np.einsum('jii->ji', qeye)[:] = 1.0
        if not self.regularize_lambda:
            qeye[:, -1, -1] = 0.0
        qhth_plus_ic = qhth + ( ( qeye * self.regularization ) if self.regularization is not None else 0 )

        qht_logs = np.einsum('kni,ij->knj', np.transpose(hqs, [0, 2, 1]), logs)
        self.gammas = [ find_b(qhth_plus_ic[i,:,:], qht_logs[i,:,:]) for i in range(self.n_estimators) ]#[dask.delayed(find_b)(qhth_plus_ic[i,:,:], qht_logs[i,:,:]) for i in range(self.n_estimators)]
        #self.gammas = dask.compute(*self.gammas)
        self.gamma = np.stack(self.gammas, axis=0)
        # enforce gamma > 0

        self.lambdas = self.gamma[:, -1, :].mean(axis=-1)
        gamma = self.gamma[self.lambdas > self.eps, :, :]
       # self.w = self.w[self.lambdas > np.finfo(float).eps**19, :, :]
       # self.b = self.b[self.lambdas > np.finfo(float).eps**19, :, :]
        self.n_estimators = gamma.shape[0]



    def crps(self, x, y):
        x, y = x.astype(np.float64), y.astype(np.float64)
        # first, take care of preprocessing
        if self.preprocessing == 'std':
            self.mean, self.std = x.mean(axis=0), x.std(axis=0)
            x = (x - self.mean) / self.std  # scales to std normal dist

        if self.preprocessing == 'minmax':
            self.min, self.max = x.min(axis=0), x.max(axis=0)
            x = ((x - self.min) / (self.max - self.min)) * 2 - 1  # scales to [-1, 1]

        if self.n_estimators == 0:
            return np.ones_like(y) * np.nan
            #act = x.dot(self.beta[:-1].reshape(-1,1))
            #ret =  np.logaddexp(act.mean(axis=0), 0) -1 + np.logaddexp( -1*act.mean(axis=0), 0)
            #return act / self.beta[-1]

        h = self.activations[self.activation]( np.einsum('ij,kjn->kin', x, self.w) + self.b )
        qtemplate = np.ones_like(h[:,:, 0], dtype=float)
        if self.standardize_y:
            y = ( y - self.ymean) / self.ystd
        act = np.concatenate( [ h, np.expand_dims(qtemplate * y.squeeze(), axis=-1) ], axis=2)
        act = np.einsum('kin,knj->kij', act, self.gamma)

        ret = np.logaddexp(act.mean(axis=0), 0) -1 + np.logaddexp( -1*act.mean(axis=0), 0)
        gam = np.expand_dims(self.gamma[:, -1, :], axis=1)
        return ret / gam.mean(axis=0)

    def predict(self, x, quantile=None, preprocessing='asis'):
        x = x.astype(np.float64)

        #if self.n_estimators == 0:
        #    print(f'lost all estimators returning {self.ymean}')
        #    return np.ones( x.shape[0], dtype=float ).reshape(-1,1) * self.ymean

        if quantile is None:
            quantile = self.quantile_of_mean
        # first, take care of preprocessing

        if self.preprocessing == 'std' and preprocessing == 'asis':
            x = (x - self.mean) / self.std  # scales to std normal dist

        if self.preprocessing == 'minmax' and preprocessing == 'asis':
            x = ((x - self.min) / (self.max - self.min)) * 2 - 1  # scales to [-1, 1]


        h = self.activations[self.activation]( np.einsum('ij,kjn->kin', x, self.w) + self.b )
        hb_without_q = np.einsum('kin,knj->kij', h, self.gamma[:, :-1, :])
        ret=  ( logit(quantile) - hb_without_q ) / np.expand_dims(self.gamma[:, -1, :], axis=1)
        ret[ self.lambdas <= self.eps, :, : ] = self.ymean if not self.standardize_y else 0
        ret = ret.mean(axis=0)
       # if ( ret * self.ystd + self.ymean if self.standardize_y else ret ).sum() >  50:
        #    print('return: {} - lambda: {}, ystd: {}'.format( ret * self.ystd + self.ymean if self.standardize_y else ret, self.lambdas, self.ystd) )
        #    print("{}".format( { k:v for k,v in vars(self).items() if k in ['encoding', 'initialization', 'regularization', 'n_estimators', 'activation', 'standardize_y', 'preprocessing', 'quantiles' ] } ) )
        #    print()
        return ret * self.ystd + self.ymean if self.standardize_y else ret

    def probability_distribution_function(self, x, preprocessing='asis', quantile=None, threshold=None, mean=True):
        x = x.astype(np.float64)
        # first, take care of preprocessing
        if self.preprocessing == 'std' and preprocessing == 'asis':
            x = (x - self.mean) / self.std  # scales to std normal dist

        if self.preprocessing == 'minmax' and preprocessing == 'asis':
            x = ((x - self.min) / (self.max - self.min)) * 2 - 1  # scales to [-1, 1]

        assert quantile is None or threshold is None, "Cannot pass both quantile and deterministic thresholds simultaneously!"
        assert quantile is not None or threshold is not None, "Must pass either quantile or thresholds for predict-pdf"

        if threshold is not None:
            if not isinstance(threshold, Iterable):
                threshold = [threshold]
            ret = []
            for q in threshold:
                if self.standardize_y:
                    q = (q - self.ymean) / self.ystd
                h = self.activations[self.activation]( np.einsum('ij,kjn->kin', x, self.w) + self.b )
                qtemplate = np.ones_like(h[:,:, 0], dtype=float)
                act = np.concatenate( [ h, np.expand_dims(qtemplate * q, axis=-1) ], axis=2)
                act = np.einsum('kin,knj->kij', act, self.gamma)
                ret1 = expit( act )
                ret1 = ret1 *  (1 - ret1) * np.expand_dims(self.gamma[:, -1, :], axis=1)
                ret.append(ret1.mean(axis=0))
            return np.hstack(ret)
        else: # quantile is not None:
            assert self.save_y, 'You must pass save_y=True in the constructor if you want to use quantile thresholds!'
            if not isinstance(quantile, Iterable):
                quantile = [quantile]
            ret = []
            for q in quantile:
                assert q > 0 and q < 1, 'quantiles must be on (0, 1)'
                q = np.nanquantile(self.y, q, method='midpoint')
                h = self.activations[self.activation]( np.einsum('ij,kjn->kin', x, self.w) + self.b )
                qtemplate = np.ones_like(h[:,:, 0], dtype=float)
                act = np.concatenate( [ h, np.expand_dims(qtemplate * q, axis=-1) ], axis=2)
                act = np.einsum('kin,knj->kij', act, self.gamma)
                ret1 = expit( act )
                ret1 = ret1 *  (1 - ret1) * np.expand_dims(self.gamma[:, -1, :], axis=1)
                ret.append(ret1.mean(axis=0))
            return np.hstack(ret)

    def predict_ne(self, x, preprocessing='asis', quantile=0.5, threshold=None, mean=True):
        x = x.astype(np.float64)
        # first, take care of preprocessing
        if self.preprocessing == 'std' and preprocessing == 'asis':
            x = (x - self.mean) / self.std  # scales to std normal dist

        if self.preprocessing == 'minmax' and preprocessing == 'asis':
            x = ((x - self.min) / (self.max - self.min)) * 2 - 1  # scales to [-1, 1]

        assert quantile is None or threshold is None, "Cannot pass both quantile and deterministic thresholds simultaneously!"
        # options are one or the other, or neither

        if quantile is not None:
            if not isinstance(quantile, Iterable):
                threshold = [quantile]
            ret = []
            for q in threshold:
                q = np.nanquantile(self.y, q, method='midpoint')
                if self.standardize_y:
                    q = (q - self.ymean) / self.ystd
                h = self.activations[self.activation]( np.einsum('ij,kjn->kin', x, self.w) + self.b )
                qtemplate = np.ones_like(h[:,:, 0], dtype=float)
                act = np.concatenate( [ h, np.expand_dims(qtemplate * q, axis=-1) ], axis=2)
                act = np.einsum('kin,knj->kij', act, self.gamma)
                ret1 = act.mean(axis=0) #/ self.n_estimators)
                ret.append(ret1)
            return h, np.hstack(ret)

    def predict_proba(self, x, preprocessing='asis', quantile=None, threshold=None, mean=True):
        x = x.astype(np.float64)
        # first, take care of preprocessing
        if self.preprocessing == 'std' and preprocessing == 'asis':
            x = (x - self.mean) / self.std  # scales to std normal dist

        if self.preprocessing == 'minmax' and preprocessing == 'asis':
            x = ((x - self.min) / (self.max - self.min)) * 2 - 1  # scales to [-1, 1]

        assert quantile is None or threshold is None, "Cannot pass both quantile and deterministic thresholds simultaneously!"
        # options are one or the other, or neither

        if threshold is not None:
            if not isinstance(threshold, Iterable):
                threshold = [threshold]
            if self.n_estimators == 0:
                return np.ones(x.shape[0]*len(threshold), dtype=float).reshape(x.shape[0], len(threshold)) / len(threshold)
            ret = []
            for q in threshold:
                if self.standardize_y:
                    q = (q - self.ymean) / self.ystd
                h = self.activations[self.activation]( np.einsum('ij,kjn->kin', x, self.w) + self.b )
                qtemplate = np.ones_like(h[:,:, 0], dtype=float)
                act = np.concatenate( [ h, np.expand_dims(qtemplate * q, axis=-1) ], axis=2)
                act = np.einsum('kin,knj->kij', act, self.gamma)
                ret1 = expit( act )[ self.lambdas > self.eps, :, : ].mean(axis=0) #/ self.n_estimators)
                ret.append(ret1)
            return np.hstack(ret)
        elif quantile is not None:
            assert self.save_y, 'You must pass save_y=True in the constructor if you want to use quantile thresholds!'
            if not isinstance(quantile, Iterable):
                quantile = [quantile]
            if self.n_estimators == 0:
                return np.ones(x.shape[0]*len(quantile), dtype=float).reshape(x.shape[0], len(quantile)) / len(quantile)
            ret = []
            for q in quantile:
                assert q > 0 and q < 1, 'quantiles must be on (0, 1)'
                q = np.nanquantile(self.y, q, method='midpoint')
                h = self.activations[self.activation]( np.einsum('ij,kjn->kin', x, self.w) + self.b )
                qtemplate = np.ones_like(h[:,:, 0], dtype=float)
                act = np.concatenate( [ h, np.expand_dims(qtemplate * q, axis=-1) ], axis=2)
                act = np.einsum('kin,knj->kij', act, self.gamma)
                ret1 = expit( act )[ self.lambdas > self.eps, :, : ].mean(axis=0) #/ self.n_estimators)
                ret.append(ret1)
            return np.hstack(ret)
        else:
            ret = []
            if self.n_estimators == 0:
                return np.ones(x.shape[0]*3, dtype=float).reshape(x.shape[0], 3) * 0.333
            for q in [self.q33, self.q67]:
                #q = (q - self.ymean) / self.ystd
                h = self.activations[self.activation]( np.einsum('ij,kjn->kin', x, self.w) + self.b )
                qtemplate = np.ones_like(h[:,:, 0], dtype=float)
                act = np.concatenate( [ h, np.expand_dims(qtemplate * q, axis=-1) ], axis=2)
                act = np.einsum('kin,knj->kij', act, self.gamma)
                ret1 = expit( act)[ self.lambdas > self.eps, :, : ].mean(axis=0)
                ret.append(ret1)
            bnan = np.stack(ret, axis=1)
            nn = bnan[:, 1] - bnan[:, 0]
            ret = np.hstack( [ bnan[:,0].reshape(-1,1), nn.reshape(-1,1), 1 - bnan[:,1].reshape(-1,1)   ]   )
            return ret / np.nansum(ret, axis=-1).reshape(-1,1)
