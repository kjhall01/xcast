
from sklearn.model_selection import KFold
import numpy as np
import scipy.linalg.lapack as la
from collections.abc import Iterable

import xarray as xr
import numpy as np
from scipy.linalg import svd
from scipy.stats import spearmanr, t
from sklearn.model_selection import KFold
from collections.abc import Iterable
import scipy.linalg.lapack as la
from matplotlib.backends.backend_pdf import PdfPages
import datetime
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import dask.array as da


class linear_regression:
    def __init__(self, fit_intercept=True, crossvalsplits=5, probability_method='error_variance'):
        self.fit_intercept = fit_intercept
        self.splits = crossvalsplits
        self.probability_method = probability_method
        assert self.probability_method in ['error_variance', 'adjusted_error_variance'], 'Invalid probability method'

    def fit(self, x, y):
        self.y = y.copy()
        self.xmean = x.mean(axis=0).reshape(1, -1)
        self.xstd = x.std(axis=0).reshape(1, -1)
        assert x.shape[0] == y.shape[0], 'X and Y must have the same number of samples!'
        x, y = x.astype(np.float64), y.astype(np.float64)
        if self.fit_intercept:
            x = np.hstack([np.ones((x.shape[0], 1)), x])
            self.xmean = np.hstack([np.ones((1,1)), self.xmean])
            self.xstd = np.hstack([np.ones((1,1)), self.xstd])
        kf = KFold(n_splits=self.splits)
        indices  = np.arange(x.shape[0])

        predictions, obs = [], []
        for train_ndxs, test_ndxs in kf.split(indices):
            xtrain, ytrain = x[train_ndxs, :], y[train_ndxs, :]
            self._fit(xtrain, ytrain)
            preds = self.predict(x[test_ndxs,:], do_pre=False)
            predictions.append(preds)
            obs.append(y[test_ndxs, :].reshape(-1, 1))
        predictions = np.vstack(predictions)
        obs = np.vstack(obs)
        self.hcst_err_var = ((predictions - obs)**2).sum(axis=0) / (self.y.shape[0] - x.shape[1] - 1)


    def _fit(self, x, y):
        if x.dtype == np.float64 and y.dtype == np.float64:
            _, B, info = la.dposv(x.T.dot(x), x.T.dot(y))
        elif x.dtype == np.float32 and y.dtype == np.float32:
            _, B, info = la.sposv(x.T.dot(x), x.T.dot(y))
        else:
            assert False, 'x: {} and y: {} not matching or good dtypes for lapack'.format(
                x.dtype, y.dtype)
        if info > 0:
            xTx = x.T.dot(x)+ np.triu(x.T.dot(x), k=1).T
            B = np.linalg.lstsq(xTx, x.T.dot(y), rcond=None)[0]
        self.coef_ = B

    def predict(self, x, do_pre=True):
        if do_pre and self.fit_intercept:
            x = np.hstack([np.ones((x.shape[0], 1)), x])
        return x.dot(self.coef_)

    def predict_proba(self, x, quantile=None):
        if self.fit_intercept:
            x = np.hstack([np.ones((x.shape[0], 1)), x])
        mu = x.dot(self.coef_)
        xvp = ((x - self.xmean)**2 / ( self.xstd**2 * self.y.shape[0] )).sum(axis=1).reshape(-1, 1)
        pred_err_var = (1+xvp).dot( self.hcst_err_var.reshape(1, -1) )
        if self.probability_method == 'error_variance':
            pred_err_var = np.ones_like(pred_err_var) * self.hcst_err_var.reshape(1, -1)
        if quantile is not None:
            if not isinstance(quantile, Iterable):
                quantile = [quantile]
            ret = []
            for q in quantile:
                assert q > 0 and q < 1, 'quantile must be float between 0 and 1'
                threshold = np.nanquantile(self.y, q, axis=0)
                xx = 1 - t.sf(threshold, self.y.shape[0] - x.shape[1] - 1 ,  loc=mu, scale=np.sqrt( pred_err_var))
                ret.append(xx)
            return np.hstack(ret)
        else:
            bn = np.nanquantile(self.y, (1.0/3.0), axis=0)
            an = np.nanquantile(self.y, (2.0/3.0), axis=0, )
            bn_prob = 1 - t.sf(bn, self.y.shape[0] - x.shape[1] - 1,  loc=mu, scale=np.sqrt(pred_err_var))
            an_prob = t.sf(an, self.y.shape[0]- x.shape[1] - 1 ,  loc=mu, scale=np.sqrt(pred_err_var))
            nn_prob = 1 - bn_prob - an_prob
            return np.hstack([bn_prob, nn_prob, an_prob])