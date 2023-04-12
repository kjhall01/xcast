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

from . import BaseEstimator
from ..core.utilities import guess_coords, check_all
from ..core.chunking import align_chunks

def svd_flip_v(v):
    signs = []
    for i in range(v.shape[0]):
        if np.abs(np.max(v[i, :])) < np.abs(np.min(v[i, :])):
            v[i, :] *= -1
            signs.append(-1)
        else:
            signs.append(1)
    return v, np.asarray(signs)


def std(X, method='midpoint', x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None):
    x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(
        X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
    check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

    X1 = X.transpose(x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
    x_data = X1.data

    def _nanquantile(x):
        pct_nans = np.isnan(x).sum(axis=-2) / x.shape[-2]
        nans = np.where(pct_nans == 1.0)
        x[nans[0], nans[1], :, nans[2]] = -999999
        ret = np.asarray(np.nanstd(x, axis=-2))
        ret[nans] = np.nan
        return ret

    results = da.blockwise(_nanquantile, 'ijl', x_data,
                           'ijkl', dtype=float, concatenate=True).persist()
    coords = {
        x_lat_dim: X1.coords[x_lat_dim].values,
        x_lon_dim: X1.coords[x_lon_dim].values,
        x_feature_dim: X1.coords[x_feature_dim].values,
    }

    dims = [x_lat_dim, x_lon_dim, x_feature_dim]
    attrs = X1.attrs
    attrs.update(
        {'generated_by': 'XCast StdDev'})
    return xr.DataArray(data=results, coords=coords, dims=dims, attrs=attrs)



class EOF:
    def __init__(self, modes=None, latitude_weighting=False, separate_members=True, crossvalidation_splits=5):
        self.eof_modes = modes
        self.latitude_weighting = latitude_weighting
        self.separate_members=separate_members

    def fit(self, X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None):
        x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim = guess_coords(X, x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim)
        check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

        self.x_lat_dim, self.x_lon_dim, self.x_feature_dim = x_lat_dim, x_lon_dim, x_feature_dim
        self.x_sample_dim = x_sample_dim

        xstd = std(X, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim)
        X = X.where(xstd != 0, other=np.nan)

        xmask =  X.mean(x_sample_dim, skipna=False).mean(x_feature_dim, skipna=False)
        self.xmask = xr.ones_like( xmask ).where(~np.isnan(xmask), other=np.nan)

        if not self.separate_members:

            if self.latitude_weighting:
                self.xweights = np.cos(np.deg2rad(getattr(X, x_lat_dim)))
                self.xweights.name = "weights"
                self.xweights = np.sqrt(self.xweights)
                xweights_flat = (X.where(np.isnan(X), other=1).transpose(x_sample_dim, x_lat_dim, x_lon_dim,  x_feature_dim) * self.xweights).mean(x_sample_dim, skipna=False).stack(point=(x_lat_dim, x_lon_dim,  x_feature_dim))

            flat_x = X.transpose(x_sample_dim, x_lat_dim, x_lon_dim,  x_feature_dim).stack(point=(x_lat_dim, x_lon_dim,  x_feature_dim)).dropna('point', how='any')
            self.xpoint = flat_x.point
            if self.latitude_weighting:
                xweights_flat = xweights_flat.sel(point=self.xpoint).values
            else:
                xweights_flat = None

            x_data = flat_x.values
            self.eof = eof_(modes=self.eof_modes, latitude_weights=xweights_flat  if xweights_flat is not None else None)
            self.eof.fit(x_data)
            scores = self.eof.transform(x_data)

            self.eof_loadings = xr.DataArray(name='eof_loadings', data=self.eof.eof_loadings, dims=['mode', 'point'], coords={'mode': [i+1 for i in range(self.eof.eof_loadings.shape[0])], 'point': self.xpoint}, attrs={'generated_by': 'XCast CCA Regressor X EOF modes'}).unstack('point').sortby(x_lat_dim).sortby(x_lon_dim).sortby(x_feature_dim)
            self.eof_loadings.name = 'eof_loadings'
            dct = { self.x_lat_dim: self.xmask.coords[self.x_lat_dim],  self.x_lon_dim: self.xmask.coords[self.x_lon_dim]  }
            self.eof_loadings = self.eof_loadings.reindex(**dct)
            self.eof_variance_explained = xr.DataArray(name='percent_variance_explained', data=self.eof.percent_variance_explained, dims=['mode'], coords={'mode': [i+1 for i in range(self.eof.eof_scores.shape[1])]}, attrs={'generated_by': 'XCast CCA Regressor X EOF Scores'})
            self.eof_scores = xr.DataArray(name='eof_scores', data=self.eof.eof_scores, dims=[x_sample_dim, 'mode'], coords={'mode': [i+1 for i in range(self.eof.eof_loadings.shape[0])], x_sample_dim: X.coords[x_sample_dim]}, attrs={'generated_by': 'XCast CCA Regressor X EOF Scores'})
            new_x = xr.ones_like(X).mean(x_feature_dim)#.swap_dims(**{ y_lat_dim: x_lat_dim, y_lon_dim: x_lon_dim})#.assign_coords(**{ y_sample_dim: getattr(Y, y_sample_dim)})
        else:
            self.eofs = []
            self.eof_loadings = []
            self.eof_scores = []
            self.pct_variances = []
            self.xpoints = []

            flat_x = X.transpose(x_sample_dim, x_lat_dim, x_lon_dim,  x_feature_dim).stack(point=(x_lat_dim, x_lon_dim,  x_feature_dim)).dropna('point', how='any')
            self.xpoint = flat_x.point

            if self.latitude_weighting:
                self.xweights = np.cos(np.deg2rad(getattr(X, x_lat_dim)))
                self.xweights.name = "weights"
                self.xweights = np.sqrt(self.xweights)
                xweights_flat = (X.where(np.isnan(X), other=1).mean(x_sample_dim).transpose( x_lat_dim, x_lon_dim, x_feature_dim) * self.xweights).stack(point=(x_lat_dim, x_lon_dim, x_feature_dim)).sel(point=self.xpoint)
            else:
                xweights_flat = None

            for member, mcoord in enumerate(X.coords[x_feature_dim].values):
                flat_x2 = flat_x.sel(**{x_feature_dim: mcoord})
                self.xpoints.append(flat_x2.point)
                x_data = flat_x2.values
                self.eofs.append(eof_(modes=self.eof_modes, latitude_weights=xweights_flat.sel(**{x_feature_dim: mcoord}).values if xweights_flat is not None else None))
                self.eofs[member].fit(x_data)
                #self.eofs[member].transform(x_data)
                self.eof_scores.append(self.eofs[member].eof_scores)
                self.eof_loadings.append(  xr.DataArray(name='eof_loadings', data=self.eofs[member].eof_loadings, dims=[ 'mode', 'point'], coords={'mode': [i+1 for i in range(self.eofs[member].eof_loadings.shape[0])], 'point': self.xpoints[member]}, attrs={'generated_by': 'XCast CCA Regressor X EOF modes'}).unstack('point').sortby(x_lat_dim).sortby(x_lon_dim) )
                self.eof_loadings[member].name = 'eof_loadings'
                dct = { self.x_lat_dim: self.xmask.coords[self.x_lat_dim],  self.x_lon_dim: self.xmask.coords[self.x_lon_dim]  }
                self.eof_loadings[member] = self.eof_loadings[member].reindex(**dct)

                self.pct_variances.append(self.eofs[member].percent_variance_explained)
            self.eof_scores = np.stack(self.eof_scores, axis=0)
            self.pct_variances = np.stack(self.pct_variances, axis=0)

            self.eof_loadings = xr.concat(self.eof_loadings, x_feature_dim).assign_coords({x_feature_dim: X.coords[x_feature_dim].values})
            self.eof_scores = xr.DataArray(name='eof_scores', data=self.eof_scores, dims=[x_feature_dim, x_sample_dim, 'mode'], coords={'mode': [i+1 for i in range(self.eofs[0].eof_scores.shape[1])], x_sample_dim: X.coords[x_sample_dim],  x_feature_dim: getattr(X, x_feature_dim)}, attrs={'generated_by': 'XCast CCA Regressor X EOF Scores'})
            self.eof_variance_explained = xr.DataArray(name='percent_variance_explained', data=self.pct_variances, dims=[x_feature_dim, 'mode'], coords={'mode': [i+1 for i in range(self.eofs[0].eof_scores.shape[1])],   x_feature_dim: getattr(X, x_feature_dim)}, attrs={'generated_by': 'XCast CCA Regressor X EOF Scores'})

    def transform(self, X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None):
        x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim = guess_coords(X, x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim)
        check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        if not self.separate_members:
            flat_x = X.transpose(x_sample_dim, x_lat_dim, x_lon_dim,  x_feature_dim).stack(point=(x_lat_dim, x_lon_dim,  x_feature_dim)).sel(point=self.xpoint)
            x_data = flat_x.values
            scores = self.eof.transform(x_data)
            eof_scores = xr.DataArray(name='eof_scores', data=scores, dims=[self.x_sample_dim, 'mode'], coords={'mode': [i+1 for i in range(self.eof.eof_loadings.shape[0])], self.x_sample_dim: X.coords[x_sample_dim].values}, attrs={'generated_by': 'XCast CCA Regressor X EOF Scores'})
            #new_x = xr.concat([ xr.ones_like(self.xmask) for i in range(x_data.shape[0]) ], self.x_sample_dim).assign_coords({self.x_sample_dim: X.coords[x_sample_dim].values}).swap_dims(**{ self.x_lat_dim: x_lat_dim, self.x_lon_dim: x_lon_dim})
        else:
            flat_x = X.transpose(x_sample_dim, x_lat_dim, x_lon_dim,  x_feature_dim).stack(point=(x_lat_dim, x_lon_dim,  x_feature_dim)).sel(point=self.xpoint)
            eof_scores = []
            for member, mcoord in enumerate(X.coords[x_feature_dim].values):
                flat_x2 = flat_x.sel(**{x_feature_dim: mcoord})
                x_data = flat_x2.values
                scores = self.eofs[member].transform(x_data)
                eof_scores.append(scores)

            eof_scores = np.stack(eof_scores, axis=0)
            eof_scores = xr.DataArray(name='eof_scores', data=eof_scores, dims=[x_feature_dim, self.x_sample_dim, 'mode'], coords={'mode': [i+1 for i in range(self.eofs[0].eof_scores.shape[1])], self.x_sample_dim: X.coords[x_sample_dim].values,  x_feature_dim: getattr(X, x_feature_dim)}, attrs={'generated_by': 'XCast CCA Regressor X EOF Scores'})

            #new_x = xr.concat([ xr.ones_like(self.xmask) for i in range(x_data.shape[0]) ], self.x_sample_dim)
           # new_x = new_x#.assign_coords({self.x_sample_dim: X.coords[x_sample_dim].values}).swap_dims(**{ self.y_lat_dim: x_lat_dim, self.y_lon_dim: x_lon_dim})
        return eof_scores

    def inverse_transform(self, X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None):
        assert 'mode' in X.dims, 'X needs to have a "mode" dimension of {} for inverse_transform to work'.format(self.eof_modes)
        X2 = X.isel(mode=0)
        x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim = guess_coords(X2.expand_dims({'lat': [0], 'lon':[0]}), 'lat', 'lon', x_sample_dim,  x_feature_dim)
        check_all(X2.expand_dims({'lat': [0], 'lon':[0]}), x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        if self.separate_members:
            ret = []
            for member, mcoord in enumerate(X.coords[x_feature_dim].values):
                scores = X.sel(**{x_feature_dim: mcoord}).transpose(x_sample_dim, 'mode').values
                features = self.eofs[member].inverse_transform(scores)
                ret.append(  xr.DataArray(name='full_field', data=features, dims=[ x_sample_dim, 'point'], coords={x_sample_dim: X.coords[x_sample_dim].values, 'point': self.xpoints[member]}, attrs={'generated_by': 'XCast CCA Regressor X EOF modes'}).unstack('point').sortby(self.x_lat_dim).sortby(self.x_lon_dim) )
                ret[member].name = 'full_field'
                dct = { self.x_lat_dim: self.xmask.coords[self.x_lat_dim],  self.x_lon_dim: self.xmask.coords[self.x_lon_dim]  }
                ret[member] = ret[member].reindex(**dct)
            ret = xr.concat(ret, x_feature_dim).assign_coords({x_feature_dim: X.coords[x_feature_dim].values})
        else:
            scores = X.transpose(x_sample_dim, 'mode').values
            features = self.eof.inverse_transform(scores)
            ret = xr.DataArray(name='full_field', data=features, dims=[ x_sample_dim, 'point'], coords={x_sample_dim: X.coords[x_sample_dim].values, 'point': self.xpoints[member]}, attrs={'generated_by': 'XCast CCA Regressor X EOF modes'}).unstack('point').sortby(self.x_lat_dim).sortby(self.x_lon_dim)
            ret.name = 'full_field'
            dct = { self.x_lat_dim: self.xmask.coords[self.x_lat_dim],  self.x_lon_dim: self.xmask.coords[self.x_lon_dim]  }
            ret = ret.reindex(**dct)
        return ret



class eof_:
    def __init__(self, modes=None, latitude_weights=None):
        self.modes = modes
        self.latitude_weights = latitude_weights

    def fit(self, x_data):
        self.mean = x_data.mean(axis=0)
        self.stddev = x_data.std(axis=0)

        assert np.min(self.stddev) > 0, "There is a Zero Standard-Deviation period somewhere within the data - could it be a long drought period? Use xc.drymask?"

        x_data = (x_data.copy() - self.mean) / self.stddev
        x_data = x_data * self.latitude_weights if self.latitude_weights is not None else x_data

        # U1 is NxP1, S = P1x1, V1t is P1xM1
        u, s, vt = svd(x_data, full_matrices=False)
        vt, xsigns = svd_flip_v(vt)
        u *= xsigns#.reshape(-1,1)
        self.percent_variance_explained = s[s!=0] / s.sum()

        s = s[:self.modes] if self.modes is not None else s
        s = s[s>0]
        self.modes = s[np.isfinite(1 / s)].shape[0]
        s = s[:self.modes]
        u = u[:, :self.modes ]
        vt = vt[:self.modes, :] * self.latitude_weights if self.latitude_weights is not None else vt[:self.modes, :]
        s = np.eye(self.modes) * s

        self.eof_scores = x_data.dot(vt.T)
        self.eof_loadings = vt
        self.eof_singular_values = s
        self.percent_variance_explained = self.percent_variance_explained[:self.modes]

    def transform(self, x_data):
        x_data = (x_data.copy() - self.mean) / self.stddev
        x_data = x_data * self.latitude_weights if self.latitude_weights is not None else x_data
        return x_data.dot(self.eof_loadings.T) #/ self.eof_singular_values[self.eof_singular_values!=0]

    def inverse_transform(self, xscores):
        one = xscores#.dot(self.eof_singular_vaules)
        two = one.dot(self.eof_loadings) if self.latitude_weights is None else one.dot(self.eof_loadings / self.latitude_weights)
        three = two / self.latitude_weights if self.latitude_weights is not None else two
        return three * self.stddev + self.mean

