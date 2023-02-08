from ..flat_estimators.classifiers import *
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from .base_estimator import BaseEstimator
from ..preprocessing.onehot import OneHotEncoder, quantile
from ..preprocessing.spatial import regrid
from ..core.utilities import guess_coords, check_all
import xarray as xr


class cMemberCount:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, Y, method='midpoint', x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None, y_lat_dim=None, y_lon_dim=None, y_sample_dim=None, y_feature_dim=None, an_thresh=0.67, bn_thresh=0.33,  explicit=False):
        x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(
            X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

        y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim = guess_coords(
            Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
        check_all(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)

        X1  = X.stack(member=(x_feature_dim, x_sample_dim)).expand_dims({'MTemp': [0]})
        self.high_threshold = quantile(X1, (2/3.0), method=method,  x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim='member', x_feature_dim='MTemp').mean('MTemp')
        self.low_threshold = quantile(X1, (1/3.0), method=method,  x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim='member', x_feature_dim='MTemp').mean('MTemp')


    def predict_proba(self, X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None):
        x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(
            X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

        bn = xr.ones_like(X).where(X <= self.low_threshold, other=0).sum(x_feature_dim) / X.shape[list(X.dims).index(x_feature_dim)]
        an = xr.ones_like(X).where(X > self.high_threshold, other=0).sum(x_feature_dim) / X.shape[list(X.dims).index(x_feature_dim)]
        nn = 1 - an - bn
        return xr.concat([bn, nn, an], x_feature_dim).assign_coords({x_feature_dim: ['BN', 'NN', 'AN']})


class cProbabilityAnomalyCorrelation:
    def __init__(self, remove_climatology=True, **kwargs):
        self.kwargs = kwargs
        self.remove_climatology = remove_climatology

    def fit(self, X, Y, method='midpoint', x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None, y_lat_dim=None, y_lon_dim=None, y_sample_dim=None, y_feature_dim=None, an_thresh=0.67, bn_thresh=0.33,  explicit=False):
        x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(
            X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

        y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim = guess_coords(
            Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
        check_all(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)

        X1  = X.stack(member=(x_feature_dim, x_sample_dim)).expand_dims({'MTemp': [0]})
        self.high_threshold = quantile(X1, (2/3.0), method=method,  x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim='member', x_feature_dim='MTemp').mean('MTemp')
        self.low_threshold = quantile(X1, (1/3.0), method=method,  x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim='member', x_feature_dim='MTemp').mean('MTemp')

        probs = self._predict_proba(X, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim)
        ohc = OneHotEncoder()
        ohc.fit(Y, x_lat_dim=y_lat_dim, x_lon_dim=y_lon_dim, x_sample_dim=y_sample_dim, x_feature_dim=y_feature_dim)
        T = ohc.transform(Y, x_lat_dim=y_lat_dim, x_lon_dim=y_lon_dim, x_sample_dim=y_sample_dim, x_feature_dim=y_feature_dim)

        self.sdratio = T.std(y_sample_dim) / probs.std(x_sample_dim)
        oprime = T.swap_dims({y_sample_dim: x_sample_dim}).assign_coords({x_sample_dim: probs.coords[x_sample_dim].values})
        pprime = probs
        if self.remove_climatology:
            oprime = oprime - 0.333
            probs = probs - 0.333
        num = (  oprime * probs ).mean( x_sample_dim)
        denom = np.sqrt( (oprime ** 2 * pprime **2 ).mean(x_sample_dim))
        self.pac = num / denom

    def _predict_proba(self, X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None):
        x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(
            X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

        bn = xr.ones_like(X).where(X <= self.low_threshold, other=0).sum(x_feature_dim) / X.shape[list(X.dims).index(x_feature_dim)]
        an = xr.ones_like(X).where(X > self.high_threshold, other=0).sum(x_feature_dim) / X.shape[list(X.dims).index(x_feature_dim)]
        nn = 1 - an - bn
        return xr.concat([bn, nn, an], x_feature_dim).assign_coords({x_feature_dim: ['BN', 'NN', 'AN']})

    def predict_proba(self, X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None):
        x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(
            X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        probs = self._predict_proba(X, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim)
        probs = probs * self.sdratio * self.pac
        featcoord = probs.coords[x_feature_dim].values


        for iteration in range(3):
            # adjust negative anomalies
            feats = []
            for i in range(probs.shape[list(probs.dims).index(x_feature_dim)]):
                feats.append( probs.isel(**{x_feature_dim: i}).copy().drop(x_feature_dim))

            for i in range(probs.shape[list(probs.dims).index(x_feature_dim)]):
                prob = probs.isel(**{x_feature_dim: i})
                negative_discrepancy = prob.where(prob < 0, other=0.01) - 0.01
                prob = prob.where(prob > 0, other=0.01)
                for j in range(probs.shape[list(probs.dims).index(x_feature_dim)]):
                    if i!=j:
                        feats[j] = feats[j] + negative_discrepancy / 2.0
            probs = xr.concat(feats, x_feature_dim).assign_coords({x_feature_dim: featcoord})

            #adjust positive anomalies
            feats = []
            for i in range(probs.shape[list(probs.dims).index(x_feature_dim)]):
                feats.append( probs.isel(**{x_feature_dim: i}).copy())

            for i in range(probs.shape[list(probs.dims).index(x_feature_dim)]):
                prob = probs.isel(**{x_feature_dim: i})
                positive_discrepancy = prob.where(prob > 1, other=0.99) - 0.99
                prob = prob.where(prob <= 1, other=0.99)
                for j in range(probs.shape[list(probs.dims).index(x_feature_dim)]):
                    if i!=j:
                        feats[j] = feats[j] + positive_discrepancy / 2.0

            probs = xr.concat(feats, x_feature_dim).assign_coords({x_feature_dim: featcoord})

            third_discrepancy = probs.sum(x_feature_dim) - 1
            probs = probs - third_discrepancy / 3.0
        return probs




class cMultivariateLogisticRegression(BaseEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = MultivariateELRClassifier


class cExtendedLogisticRegression(BaseEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = ELRClassifier


class cMultiLayerPerceptron(BaseEstimator):
    def __init__(self, hidden_layer_sizes=None, **kwargs):
        if hidden_layer_sizes is not None:
            kwargs['hidden_layer_sizes'] = hidden_layer_sizes
        else:
            kwargs['hidden_layer_sizes'] = (5,)
        super().__init__(**kwargs)
        self.model_type = MLPClassifier


class cNaiveBayes(BaseEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = NaiveBayesClassifier


class cRandomForest(BaseEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = RFClassifier


class cPOELM(BaseEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = POELMClassifier
