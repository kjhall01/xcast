import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from scipy.special import softmax
from sklearn.linear_model import GammaRegressor, PoissonRegressor


def insert_zeros(x, i):
    cols = x.shape[1]
    if cols == 1:
        if i == 0:
            return np.hstack([np.zeros(x.shape), x])
        elif i == 1:
            return np.hstack([x, np.zeros(x.shape)])
        else:
            return np.hstack([x, np.zeros(x.shape)])
    else:
        if i == 0:
            return np.hstack([np.zeros((x.shape[0], 1)), x])
        elif i == cols:
            return np.hstack([x, np.zeros((x.shape[0], 1))])
        elif i == 1:
            if cols > 2:
                return np.hstack([x[:, 0].reshape(-1, 1), np.zeros((x.shape[0], 1)), x[:, 1:]])
            else:
                return np.hstack([x[:, 0].reshape(-1, 1), np.zeros((x.shape[0], 1)), x[:, 1:].reshape(-1, 1)])
        elif i == cols-1:
            if cols > 2:
                return np.hstack([x[:, :cols-1], np.zeros((x.shape[0], 1)), x[:, cols-1:].reshape(-1, 1)])
            else:
                return np.hstack([x[:, :cols-1].reshape(-1, 1), np.zeros((x.shape[0], 1)), x[:, cols-1:].reshape(-1, 1)])
        else:
            return np.hstack([x[:, :i], np.zeros((x.shape[0], 1)), x[:, i:]])


class rf_classifier:
    def __init__(self, **kwargs):
        self.model = RandomForestClassifier(**kwargs)

    def fit(self, x, y):
        self.model.fit(x, np.argmax(y, axis=1))
        self.classes = [i for i in range(y.shape[1])]

    def predict_proba(self, x):
        ret = self.model.predict_proba(x)
        for i in self.classes:
            if i not in self.model.classes_:
                ret = insert_zeros(ret, i)
        return ret

    def predict(self, x):
        ret = self.predict_proba(x)
        return np.argmax(ret, axis=1)


class nan_classifier:
    def __init__(self, **kwargs):
        self.model = None
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

    def fit(self, x, *args, **kwargs):
        self.x_features = x.shape[1]
        if len(args) > 0:
            y = args[0]
            self.y_features = y.shape[1]

    def transform(self, x, **kwargs):
        assert self.x_features == x.shape[1]
        ret = np.empty((x.shape[0], self.n_components))
        ret[:] = np.nan
        return ret

    def predict(self, x, n_out=1, **kwargs):
        assert self.x_features == x.shape[1]
        ret = np.empty((x.shape[0], n_out))
        ret[:] = np.nan
        return ret

    def predict_proba(self, x, n_out=3, **kwargs):
        assert self.x_features == x.shape[1]
        ret = np.empty((x.shape[0], n_out))
        ret[:] = np.nan
        return ret


class naive_bayes_classifier:
    def __init__(self, **kwargs):
        self.model = MultinomialNB(**kwargs)

    def fit(self, x, y):
        self.model.partial_fit(x, np.argmax(y, axis=-1),
                               classes=[i for i in range(y.shape[1])])

    def predict_proba(self, x):
        return self.model.predict_proba(x)

    def predict(self, x):
        ret = self.model.predict_proba(x)
        return np.argmax(ret, axis=1)



class nan_regression:
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


class gamma_regression:
    def __init__(self, **kwargs):
        self.model = GammaRegressor(**kwargs)

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x).reshape(-1, 1)


class poisson_regression:
    def __init__(self, **kwargs):
        self.model = PoissonRegressor(**kwargs)

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x).reshape(-1, 1)


class random_forest_regressor:
    def __init__(self, **kwargs):
        if 'n_estimators' not in kwargs.keys():
            kwargs['n_estimators'] = 5
        self.model = RandomForestRegressor(**kwargs)

    def fit(self, x, y):
        self.model.fit(x, np.squeeze(y))

    def predict(self, x):
        return self.model.predict(x).reshape(-1, 1)
