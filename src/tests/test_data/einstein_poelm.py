from scipy.special import expit, logit
import datetime as dt
import numpy as np
import scipy.linalg.lapack as la

class POELM:
    """Ensemble Probabilistic-Output Extreme Learning Machine using Einstein Notation"""

    def __init__(self, activation='lin', hidden_layer_size=5, c=3, preprocessing='minmax', n_estimators=30, eps=np.finfo('float').eps, activations=None):
        assert isinstance(hidden_layer_size, int) and hidden_layer_size > 0, 'Invalid hidden_layer_size {}'.format(hidden_layer_size)
        assert type(c) is int, 'Invalid C {}'.format(c)
        assert type(preprocessing) is str and preprocessing in ['std', 'minmax', 'none'], 'Invalid preprocessing {}'.format(preprocessing)
        self.activation = activation
        self.c = c
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

    def fit(self, x, y_terc):
        x, y = x.astype(np.float64), y_terc.astype(np.float64)

        # first, take care of preprocessing
        if self.preprocessing == 'std':
            self.mean, self.std = x.mean(axis=0), x.std(axis=0)
            x = (x - self.mean) / self.std  # scales to std normal dist

        if self.preprocessing == 'minmax':
            self.min, self.max = x.min(axis=0), x.max(axis=0)
            x = ((x - self.min) / (self.max - self.min)) * 2 - 1  # scales to [-1, 1]

        # after transformation, check feature dim
        x_features, y_features = x.shape[1], y_terc.shape[1]

        # now, initialize weights & do all repeated stochastic initializations
        self.w = np.random.randn(self.n_estimators, x_features, self.hidden_layer_size)
        self.b = np.random.randn(self.n_estimators, 1, self.hidden_layer_size)

        h = self.activations[self.activation]( np.einsum('ij,kjn->kin', x, self.w) + self.b )
        hth = np.stack([h[i, :, :].T.dot(h[i,:,:]) for i in range(self.n_estimators)], axis=0)#np.einsum('kni,kin->knn', np.transpose(h, [0, 2, 1]), h)
        eye = np.zeros(hth.shape)
        np.einsum('jii->ji', eye)[:] = 1.0
        hth_plus_ic = hth + eye / (2**self.c)



        y_terc -= self.eps
        y_terc = np.abs(y_terc)
        logs = logit(y_terc)
        #logs = logs - 0.333 * (logs.max() - logs.min())
        ht_logs = np.einsum('kni,ij->knj', np.transpose(h, [0, 2, 1]), logs)

        self.gammas = []
        for i in range(self.n_estimators):
            if x.dtype == np.float64 and y.dtype == np.float64:
                _, B, info = la.dposv(hth_plus_ic[i,:,:], ht_logs[i,:,:])
            elif x.dtype == np.float32 and y.dtype == np.float32:
                _, B, info = la.sposv(hth_plus_ic[i,:,:], ht_logs[i,:,:])
            else:
                assert False, 'x: {} and y: {} not matching or good dtypes for lapack'.format(
                    x.dtype, y.dtype)
            if info > 0:
                hth_plus_ic = hth_plus_ic + np.triu(hth_plus_ic, k=1)
                B = np.linalg.lstsq(hth_plus_ic[i,:,:], ht_logs[i,:,:], rcond=None)[0]
            self.gammas.append(B)
        self.gamma = np.stack(self.gammas, axis=0)


    def predict(self, x, preprocessing='asis'):
        return np.argmax(self.predict_proba(x), axis=-1)

    def predict_proba(self, x, preprocessing='asis'):
        x = x.astype(np.float64)
        # first, take care of preprocessing
        if self.preprocessing == 'std' and preprocessing == 'asis':
            x = (x - self.mean) / self.std  # scales to std normal dist

        if self.preprocessing == 'minmax' and preprocessing == 'asis':
            x = ((x - self.min) / (self.max - self.min)) * 2 - 1  # scales to [-1, 1]

        h = self.activations[self.activation]( np.einsum('ij,kjn->kin', x, self.w) + self.b )
        act = np.einsum('kin,knj->ij', h, self.gamma)
        ret = expit( act / self.n_estimators)
        return ret / ret.sum(axis=-1).reshape(-1,1)
