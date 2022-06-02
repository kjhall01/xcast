import numpy as np
from sklearn.linear_model import GammaRegressor, PoissonRegressor
from sklearn.ensemble import RandomForestRegressor


class NanRegression:
    def __init__(self, **kwargs):
        self.model = None

    def fit(self, x, y=None):
        self.x_features = x.shape[1]

    def transform(self, x):
        assert self.x_features == x.shape[1]
        ret = np.empty((x.shape[0], 1))
        ret[:] = np.nan
        return ret

    def predict(self, x):
        assert self.x_features == x.shape[1]
        ret = np.empty((x.shape[0], 1))
        ret[:] = np.nan
        return ret


class GammaRegressionOne:
    def __init__(self, **kwargs):
        self.model = GammaRegressor(**kwargs)

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x).reshape(-1, 1)


class PoissonRegressionOne:
    def __init__(self, **kwargs):
        self.model = PoissonRegressor(**kwargs)

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x).reshape(-1, 1)


class RandomForestRegressorOne:
    def __init__(self, **kwargs):
        if 'n_estimators' not in kwargs.keys():
            kwargs['n_estimators'] = 5
        self.model = RandomForestRegressor(**kwargs)

    def fit(self, x, y):
        self.model.fit(x, np.squeeze(y))

    def predict(self, x):
        return self.model.predict(x).reshape(-1, 1)
