from scipy.special import expit, logit
import datetime as dt
import numpy as np
import scipy.linalg.lapack as la
from collections.abc import Iterable
import copy
import matplotlib.pyplot as plt

class EPOELM:
    """Ensemble Extreme Learning Machine using Einstein Notation"""

    def __init__(self, encoding='nonexceedance', activation='relu', hidden_layer_size=5, c=-10, preprocessing='minmax', n_estimators=50, eps=np.finfo('float').eps, activations=None, quantiles=[0.2, 0.4, 0.6, 0.8]):
        assert isinstance(hidden_layer_size, int) and hidden_layer_size > 0, 'Invalid hidden_layer_size {}'.format(hidden_layer_size)
        assert type(c) is int, 'Invalid C {}'.format(c)
        assert type(preprocessing) is str and preprocessing in ['std', 'minmax', 'none'], 'Invalid preprocessing {}'.format(preprocessing)
        self.activation = activation
        self.c = c
        self.quantiles=quantiles
        self.encoding=encoding
        assert encoding.lower() in ['nonexceedance', 'binary'], 'invalid encoding for epoelm - must be "nonexceedance" or "binary"'
        self.hidden_layer_size = hidden_layer_size
        self.preprocessing = preprocessing
        self.n_estimators=n_estimators
        self.eps=eps
        self.activations = {
            'sigm': expit,
            'tanh': np.tanh,
            'relu': lambda ret: np.maximum(0, ret),
            'lin': lambda ret: ret
        } if activations is None else activations
        assert activation in self.activations.keys(), 'invalid activation function {}'.format(activation)

    def set_params(self, **params):
        for key in params.keys():
            setattr(self, key, params[key])
        return self

    def get_params(self, deep=False):
        return vars(self)

    def fit(self, x, y):
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

        x_features, y_features = x.shape[1], y.shape[1]
        self.w = np.random.randn(self.n_estimators, x_features, self.hidden_layer_size)
        self.b = np.random.randn(self.n_estimators, 1, self.hidden_layer_size)

        act = np.einsum('ij,kjn->kin', x, self.w) + self.b
        h = self.activations[self.activation]( act )
        qtemplate = np.ones_like(h[:,:, 0], dtype=float)

        quantiles = [ np.nanquantile(y, i, method='midpoint') for i in self.quantiles ]
        thresholds = [ ( i - self.ymean ) / self.ystd for i in quantiles]
        self.q33, self.q67 = np.nanquantile(y, 1.0/3.0, method='midpoint'), np.nanquantile(y, 2.0/3.0, method='midpoint')

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
            ts = np.abs(ts - self.eps)
            logs = logit(ts)
        else:
            logs = ts

        qhth = np.stack([hqs[i, :, :].T.dot(hqs[i,:,:]) for i in range(self.n_estimators)], axis=0)
        qeye = np.zeros(qhth.shape)
        np.einsum('jii->ji', qeye)[:] = 1.0
        qhth_plus_ic = qhth + qeye / (2**self.c)

        qht_logs = np.einsum('kni,ij->knj', np.transpose(hqs, [0, 2, 1]), logs)
        self.gammas = []
        for i in range(self.n_estimators):
            if x.dtype == np.float64 and y.dtype == np.float64:
                _, B, info = la.dposv(qhth_plus_ic[i,:,:], qht_logs[i,:,:])
            elif x.dtype == np.float32 and y.dtype == np.float32:
                _, B, info = la.sposv(qhth_plus_ic[i,:,:], qht_logs[i,:,:])
            else:
                assert False, 'x: {} and y: {} not matching or good dtypes for lapack'.format(
                    x.dtype, y.dtype)
            if info > 0:
                qhth_plus_ic = qhth_plus_ic + np.triu(qhth_plus_ic, k=1)
                B = np.linalg.lstsq(qhth_plus_ic[i,:,:], qht_logs[i,:,:], rcond=None)[0]
            self.gammas.append(B)
        self.gamma = np.stack(self.gammas, axis=0)

    def predict(self, x, quantile=0.5, preprocessing='asis'):
        x = x.astype(np.float64)
        # first, take care of preprocessing
        if self.preprocessing == 'std' and preprocessing == 'asis':
            x = (x - self.mean) / self.std  # scales to std normal dist

        if self.preprocessing == 'minmax' and preprocessing == 'asis':
            x = ((x - self.min) / (self.max - self.min)) * 2 - 1  # scales to [-1, 1]

        h = self.activations[self.activation]( np.einsum('ij,kjn->kin', x, self.w) + self.b )
        hb_without_q = np.einsum('kin,knj->kij', h, self.gamma[:, :-1, :])
        return ( ( logit(quantile) - hb_without_q ) / np.expand_dims(self.gamma[:, -1, :], axis=1) ).mean(axis=0) * self.ystd + self.ymean


    def predict_proba(self, x, preprocessing='asis', threshold=None):
        x = x.astype(np.float64)
        # first, take care of preprocessing
        if self.preprocessing == 'std' and preprocessing == 'asis':
            x = (x - self.mean) / self.std  # scales to std normal dist

        if self.preprocessing == 'minmax' and preprocessing == 'asis':
            x = ((x - self.min) / (self.max - self.min)) * 2 - 1  # scales to [-1, 1]

        if threshold is not None:
            if not isinstance(threshold, Iterable):
                threshold = [threshold]
            ret = []
            for q in threshold:
                q = (q - self.ymean) / self.ystd
                h = self.activations[self.activation]( np.einsum('ij,kjn->kin', x, self.w) + self.b )
                qtemplate = np.ones_like(h[:,:, 0], dtype=float)
                act = np.concatenate( [ h, np.expand_dims(qtemplate * q, axis=-1) ], axis=2)
                act = np.einsum('kin,knj->kij', act, self.gamma)
                ret1 = expit( act ).mean(axis=0) #/ self.n_estimators)
                ret.append(ret1)
            return np.hstack(ret)
        else:
            ret = []
            for q in [self.q33, self.q67]:
                q = (q - self.ymean) / self.ystd
                h = self.activations[self.activation]( np.einsum('ij,kjn->kin', x, self.w) + self.b )
                qtemplate = np.ones_like(h[:,:, 0], dtype=float)
                act = np.concatenate( [ h, np.expand_dims(qtemplate * q, axis=-1) ], axis=2)
                act = np.einsum('kin,knj->kij', act, self.gamma)
                ret1 = expit( act).mean(axis=0)
                ret.append(ret1)
            bnan = np.stack(ret, axis=1)
            nn = bnan[:, 1] - bnan[:, 0]
            ret = np.hstack( [ bnan[:,0].reshape(-1,1), nn.reshape(-1,1), 1 - bnan[:,1].reshape(-1,1)   ]   )
            return ret / np.nansum(ret, axis=-1).reshape(-1,1)
