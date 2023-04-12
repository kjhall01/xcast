from ..preprocessing.onehot import OneHotEncoder, quantile
from ..core.utilities import guess_coords, check_all, check_xyt_compatibility
import xarray as xr



class Ensemble:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, Y, method='midpoint', x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None, y_lat_dim=None, y_lon_dim=None, y_sample_dim=None, y_feature_dim=None, an_thresh=0.67, bn_thresh=0.33,  explicit=False):
        x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(
            X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

        y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim = guess_coords(
            Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
        check_all(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)

        check_xyt_compatibility(X, Y, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)

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
    
    def predict(self, X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None):
        x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(
            X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        return X.mean(x_feature_dim).expand_dims({'M': [0]})