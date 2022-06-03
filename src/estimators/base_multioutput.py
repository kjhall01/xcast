from sklearn.neural_network import MLPRegressor
import xarray as xr
import numpy as np
from ..core.utilities import guess_coords, check_all


class MultiOutputRegressor:
    def __init__(self, model, **kwargs):
        self.kwargs = kwargs
        self.model = model

    def predict(self, X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None):
        x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim = guess_coords(
            X, x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim)
        check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

        flat_x = X.transpose(x_sample_dim, x_lat_dim, x_lon_dim,  x_feature_dim).stack(
            point=(x_lat_dim, x_lon_dim,  x_feature_dim)).dropna('point')
        x_data = flat_x.values

        preds = self.model.predict(x_data)
        ret = xr.DataArray(name='predicted_values', data=preds, dims=[x_sample_dim, 'point'], coords={'point': self.ypoint_flat, x_sample_dim: X.coords[x_sample_dim]}, attrs={
                           'generated_by': 'Pure-Python CPT CCA Regressor Predicted Valuse'}).unstack('point').sortby(x_lat_dim).sortby(x_lon_dim)
        ret.attrs.update(X.attrs)
        return ret

    def fit(self, X, Y, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None, y_lat_dim=None, y_lon_dim=None, y_sample_dim=None, y_feature_dim=None):
        x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim = guess_coords(
            X, x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim)
        y_lat_dim, y_lon_dim, y_sample_dim,  y_feature_dim = guess_coords(
            Y, y_lat_dim, y_lon_dim, y_sample_dim,  y_feature_dim)
        check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        check_all(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)

        # extract 2D [ T x Stacked(X, Y, M)] datasets from 4D [T X Y M] DataArrays, removing points in space with missing values
        flat_x = X.transpose(x_sample_dim, x_lat_dim, x_lon_dim,  x_feature_dim).stack(
            point=(x_lat_dim, x_lon_dim,  x_feature_dim)).dropna('point')

        flat_y = Y.transpose(y_sample_dim, y_lat_dim, y_lon_dim,  y_feature_dim).stack(
            point=(y_lat_dim, y_lon_dim,  y_feature_dim)).dropna('point')

        x_data, y_data = flat_x.values, flat_y.values
        self.ypoint_flat = flat_y.point

        self.model = self.model(**self.kwargs)
        self.model.fit(x_data, y_data)
