import scipy.stats as ss
from ..core.utilities import *
import numpy as np
import dask.array as da


def drymask(X, dry_threshold=0.001, quantile_threshold=0.33, method='midpoint', x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None):
    x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(
        X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
    check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

    if X.dims[0] != x_lat_dim or X.dims[1] != x_lon_dim or X.dims[2] != x_sample_dim or X.dims[3] != x_feature_dim:
        X = X.transpose(x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

    bn = quantile(X, quantile_threshold, method=method, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim)
    mask = bn.where(bn >= dry_threshold, other=np.nan)
    return mask.where(np.isnan(mask), other=1)

def quantile(X, threshold, method='midpoint', x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None):
    x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(
        X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
    check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

    X1 = X.transpose(x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
    x_data = X1.data

    def _nanquantile(x):
        pct_nans = np.isnan(x).sum(axis=-2) / x.shape[-2]
        nans = np.where(pct_nans == 1.0)
        x[nans[0], nans[1], :, nans[2]] = -999999
        ret = np.asarray(np.nanquantile(x, threshold, axis=-2, method=method))
        ret[nans] = np.nan
        return ret

    results = da.blockwise(_nanquantile, 'ijl', x_data,
                           'ijkl', dtype=float, concatenate=True).persist()
    coords = {
        x_lat_dim: X1.coords[x_lat_dim].values,
        x_lon_dim: X1.coords[x_lon_dim].values,
        x_feature_dim: [i for i in range(results.shape[-1])],
    }

    dims = [x_lat_dim, x_lon_dim, x_feature_dim]
    attrs = X1.attrs
    attrs.update(
        {'generated_by': 'XCast One-Hot Encoded'})
    return xr.DataArray(data=results, coords=coords, dims=dims, attrs=attrs)


class RankedTerciles:
    def __init__(self, low_thresh=None, high_thresh=None, explicit=False):
        self.low_thresh, self.high_thresh = low_thresh, high_thresh
        self.explicit = explicit

    def fit(self, X, quantile_method='midpoint', x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None):
        x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(
            X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        iseldict = {x_feature_dim: 0}
        self.feature_dim, self.lat_dim, self.lon_dim = x_feature_dim, x_lat_dim, x_lon_dim
        X1 = X.isel()  # **iseldict)
        if self.low_thresh is None:
            self.low_thresh = 0.33
        if self.high_thresh is None:
            self.high_thresh = 0.67

        if self.explicit:
            self.high_threshold = quantile(
                X1, 0.33, method=quantile_method, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim)
            self.low_threshold = quantile(
                X1, 0.66, method=quantile_method, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim)
            self.high_threshold = xr.ones_like(
                self.high_threshold)*self.high_thresh
            self.low_threshold = xr.ones_like(
                self.low_threshold) * self.low_thresh
        else:
            self.high_threshold = quantile(
                X1, self.high_thresh, method=quantile_method,  x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim).isel(**{x_feature_dim: 0})
            self.low_threshold = quantile(
                X1, self.low_thresh, method=quantile_method,  x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim).isel(**{x_feature_dim: 0})

        self.nanmask = X1.mean(x_sample_dim).mean(x_feature_dim)
        self.nanmask = self.nanmask.where(np.isnan(self.nanmask), other=1)

    def transform(self, X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None):
        x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(
            X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        #iseldict = {x_feature_dim: 0}
        #X1 = X.isel(**iseldict)
        self.high_threshold = self.high_threshold.swap_dims(**{ self.lat_dim: x_lat_dim, self.lon_dim: x_lon_dim})
        self.low_threshold = self.low_threshold.swap_dims(**{ self.lat_dim: x_lat_dim, self.lon_dim: x_lon_dim})

        self.feature_dim, self.lat_dim, self.lon_dim = x_feature_dim, x_lat_dim, x_lon_dim
        X_BN = X.where(X < self.low_threshold, other=-999)
        X_BN = X_BN.where(X_BN == -999, other=1.0)
        X_BN = X_BN.where(X_BN == 1.0, other=0)

        X_AN = X.where(X > self.high_threshold, other=-998)
        X_AN = X_AN.where(X_AN == -998, other=1.0)
        X_AN = X_AN.where(X_AN == 1.0, other=0)

        X_N = X.where(self.low_threshold <= X, other=0.0)
        X_N = X_N.where(X_N <= self.high_threshold, other=0.0)
        X_N = X_N.where(X_N == 0.0, other=1.0)
        X1 = xr.concat([X_BN, X_N, X_AN], x_feature_dim)
        attrs = X1.attrs

        r = X1.assign_coords({x_feature_dim: ['BN', 'NN', 'AN']}) * self.nanmask
        r.attrs['generated_by'] = attrs['generated_by'] + \
            '\n  XCAST Ranked Tercile One-Hot Encoded' if 'generated_by' in attrs.keys(
        ) else '\n  XCAST Ranked Tercile One-Hot Encoded '
        return r
