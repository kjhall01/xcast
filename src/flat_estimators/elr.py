import copy
import numpy as np
from sklearn.decomposition import PCA
import statsmodels.api as sm
from collections.abc import Iterable

class extended_logistic_regression:
    def __init__(self, pca=-999, preprocessing='none', verbose=False, **kwargs):
        self.kwargs = kwargs
        self.an_thresh = 0.67
        self.bn_thresh = 0.33
        self.pca = pca
        self.preprocessing = preprocessing
        self.verbose = verbose

    def fit(self, x, y):
        self.models = [multivariate_extended_logistic_regression(
            pca=self.pca, preprocessing=self.preprocessing, verbose=self.verbose, **self.kwargs) for i in range(x.shape[1])]
        for i in range(x.shape[1]):
            self.models[i].bn_thresh = self.bn_thresh
            self.models[i].an_thresh = self.an_thresh
            self.models[i].fit(x[:, i].reshape(-1, 1), y)

    def predict_proba(self, x, quantile=None):
        res = []
        for i in range(x.shape[1]):
            res.append(self.models[i].predict_proba(
                x[:, i].reshape(-1, 1), quantile=quantile))
        res = np.stack(res, axis=0)
        return np.nanmean(res, axis=0)
    
    def predict(self, x, quantile=None):
        raise NotImplementedError


class multivariate_extended_logistic_regression:
    def __init__(self, pca=-999, preprocessing='none', verbose=False, thresholds=[0.33, 0.67], **kwargs):
        self.kwargs = kwargs
        self.thresholds = thresholds
        self.preprocessing = preprocessing
        self.verbose = verbose

    def fit(self, x, y):
        y = y[:, 0].reshape(-1, 1)
        # first, take care of preprocessing
        x2 = copy.deepcopy(x)
        y2 = copy.deepcopy(y)
        if self.preprocessing == 'std':
            self.mean, self.std = x.mean(axis=0), x.std(axis=0)
            x2 = (copy.deepcopy(x) - self.mean) / \
                self.std  # scales to std normal dist
        if self.preprocessing == 'minmax':
            self.min, self.max = x.min(axis=0), x.max(axis=0)
            x2 = ((copy.deepcopy(x) - self.min) /
                  (self.max - self.min)) * 2 - 1  # scales to [-1, 1]
        self.y = y2
        bs = [np.quantile(y, thresh, method='midpoint')
                for thresh in self.thresholds]
        y = np.vstack([np.where(y < b, np.ones((y.shape[0], 1)).astype(
            np.float64)*0.999, np.ones((y.shape[0], 1)).astype(np.float64)*0.001) for b in bs])
        v = []
        for b in bs:
            x_bn = np.hstack(
                [x2, np.ones((x2.shape[0], 1), dtype=np.float64)*b])
            v.append(x_bn)
        x3 = np.vstack(v)
        model = sm.GLM(y, sm.add_constant(
            x3, has_constant='add'), family=sm.families.Binomial())
        self.model = model.fit()


    def nonexceedance(self, x, quantile=0.5):
        if not isinstance(quantile, Iterable):
            quantile = np.asarray([quantile])
        x2 = copy.deepcopy(x)
        if self.preprocessing == 'std':
            x2 = (copy.deepcopy(x) - self.mean) / \
                self.std  # scales to std normal dist
        if self.preprocessing == 'minmax':
            x2 = ((copy.deepcopy(x) - self.min) /
                  (self.max - self.min)) * 2 - 1  # scales to [-1, 1]
        thresh = np.quantile(self.y, quantile, method='midpoint')
        x_an = np.hstack([x2, np.ones((x.shape[0], 1)) * thresh])
        # self.x_an.append(x_an)
        x_an = sm.add_constant(x_an, has_constant='add')
        return self.model.predict(x_an).reshape(-1, 1)


    def exceedance(self, x, quantile=0.5):
        return 1 - self.nonexceedance(x, quantile=quantile)

    def predict_proba(self, x, quantile=None):
        if quantile is None:
            bn = self.nonexceedance(x, quantile=(1/3))
            an = self.exceedance(x, quantile=(2/3))
            nn = 1 - (bn + an)
            return np.hstack([bn, nn, an])
        else:
            if not isinstance(quantile, Iterable):
                quantile = np.asarray([quantile])
            return np.hstack([self.nonexceedance(x, quantile=q) for q in quantile])


    def predict(self, x, quantile=None):
        raise NotImplementedError
