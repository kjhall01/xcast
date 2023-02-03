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


def svd_flip_v(v):
    signs = []
    for i in range(v.shape[0]):
        if np.abs(np.max(v[i, :])) < np.abs(np.min(v[i, :])):
            v[i, :] *= -1
            signs.append(-1)
        else:
            signs.append(1)
    return v, np.asarray(signs)

class CrossValidatedLinearRegression:
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

class rXValMLR(BaseEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = CrossValidatedLinearRegression


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
            new_x = xr.concat([ xr.ones_like(self.xmask) for i in range(x_data.shape[0]) ], self.x_sample_dim).assign_coords({self.x_sample_dim: X.coords[x_sample_dim].values}).swap_dims(**{ self.x_lat_dim: x_lat_dim, self.x_lon_dim: x_lon_dim})
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

            new_x = xr.concat([ xr.ones_like(self.xmask) for i in range(x_data.shape[0]) ], self.x_sample_dim)
            new_x = new_x#.assign_coords({self.x_sample_dim: X.coords[x_sample_dim].values}).swap_dims(**{ self.y_lat_dim: x_lat_dim, self.y_lon_dim: x_lon_dim})
        return new_x * eof_scores

    def inverse_transform(self, X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None):
        assert 'mode' in X.dims, 'X needs to have a "mode" dimension of {} for inverse_transform to work'.format(self.eof_modes)
        X2 = X.isel(mode=0)
        x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim = guess_coords(X2, x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim)
        check_all(X2, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        if self.separate_members:
            ret = []
            for member, mcoord in enumerate(X.coords[x_feature_dim].values):
                scores = X.mean(x_lat_dim).mean(x_lon_dim).sel(**{x_feature_dim: mcoord}).transpose(x_sample_dim, 'mode').values
                features = self.eofs[member].inverse_transform(scores)
                ret.append(  xr.DataArray(name='full_field', data=features, dims=[ x_sample_dim, 'point'], coords={x_sample_dim: X.coords[x_sample_dim].values, 'point': self.xpoints[member]}, attrs={'generated_by': 'XCast CCA Regressor X EOF modes'}).unstack('point').sortby(x_lat_dim).sortby(x_lon_dim) )
                ret[member].name = 'full_field'
                dct = { self.x_lat_dim: self.xmask.coords[self.x_lat_dim],  self.x_lon_dim: self.xmask.coords[self.x_lon_dim]  }
                ret[member] = ret[member].reindex(**dct)
            ret = xr.concat(ret, x_feature_dim).assign_coords({x_feature_dim: X.coords[x_feature_dim].values})
        else:
            scores = X.mean(x_lat_dim).mean(x_lon_dim).transpose(x_sample_dim, 'mode').values
            features = self.eof.inverse_transform(scores)
            ret = xr.DataArray(name='full_field', data=features, dims=[ x_sample_dim, 'point'], coords={x_sample_dim: X.coords[x_sample_dim].values, 'point': self.xpoints[member]}, attrs={'generated_by': 'XCast CCA Regressor X EOF modes'}).unstack('point').sortby(x_lat_dim).sortby(x_lon_dim)
            ret.name = 'full_field'
            dct = { self.x_lat_dim: self.xmask.coords[self.x_lat_dim],  self.x_lon_dim: self.xmask.coords[self.x_lon_dim]  }
            ret = ret.reindex(**dct)
        return ret


class PCR:
    def __init__(self, eof_modes=None, latitude_weighting=False, separate_members=True, crossvalidation_splits=5, chunks=(5,5), **kwargs):
        self.eof_modes = eof_modes
        self.latitude_weighting = latitude_weighting
        self.mlr_kwargs = kwargs
        self.model_type = rXValMLR # this must be a class which has predict and predict_proba, and also probability of non-exceedance
        self.separate_members=separate_members
        self.chunks=chunks
        self.mlr_kwargs['crossvalsplits'] = crossvalidation_splits

    def fit(self, X, Y, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None, y_lat_dim=None, y_lon_dim=None, y_sample_dim=None, y_feature_dim=None):
        x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim = guess_coords(X, x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim)
        y_lat_dim, y_lon_dim, y_sample_dim,  y_feature_dim = guess_coords(Y, y_lat_dim, y_lon_dim, y_sample_dim,  y_feature_dim)
        check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        check_all(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)

        self.x_lat_dim, self.x_lon_dim, self.x_feature_dim = x_lat_dim, x_lon_dim, x_feature_dim
        self.x_sample_dim = x_sample_dim


        xmask =  X.mean(x_sample_dim, skipna=False).mean(x_feature_dim, skipna=False)
        self.xmask = xr.ones_like( xmask ).where(~np.isnan(xmask), other=np.nan)

        self.y_lat_dim, self.y_lon_dim, self.y_sample_dim = y_lat_dim, y_lon_dim, y_sample_dim
        self.y_feature_dim = y_feature_dim
        ymask =  Y.mean(y_sample_dim, skipna=False).mean(y_feature_dim, skipna=False)
        self.ymask = xr.ones_like( ymask ).where(~np.isnan(ymask), other=np.nan)

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
            self.eof_scores = xr.DataArray(name='eof_scores', data=self.eof.eof_scores, dims=[y_sample_dim, 'mode'], coords={'mode': [i+1 for i in range(self.eof.eof_loadings.shape[0])], y_sample_dim: Y.coords[y_sample_dim]}, attrs={'generated_by': 'XCast CCA Regressor X EOF Scores'})
            new_x = xr.ones_like(Y).mean(y_feature_dim).swap_dims(**{ y_lat_dim: x_lat_dim, y_lon_dim: x_lon_dim})#.assign_coords(**{ y_sample_dim: getattr(Y, y_sample_dim)})
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
                self.eofs.append(eof_(modes=self.eof_modes, latitude_weights=xweights_flat.sel(**{x_feature_dim: mcoord}).values  if xweights_flat is not None else None))
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
            self.eof_scores = xr.DataArray(name='eof_scores', data=self.eof_scores, dims=[x_feature_dim, y_sample_dim, 'mode'], coords={'mode': [i+1 for i in range(self.eofs[0].eof_scores.shape[1])], y_sample_dim: Y.coords[y_sample_dim],  x_feature_dim: getattr(X, x_feature_dim)}, attrs={'generated_by': 'XCast CCA Regressor X EOF Scores'})
            self.eof_variance_explained = xr.DataArray(name='percent_variance_explained', data=self.pct_variances, dims=[x_feature_dim, 'mode'], coords={'mode': [i+1 for i in range(self.eofs[0].eof_scores.shape[1])],   x_feature_dim: getattr(X, x_feature_dim)}, attrs={'generated_by': 'XCast CCA Regressor X EOF Scores'})
            new_x = xr.ones_like(Y).mean(y_feature_dim).swap_dims(**{ y_lat_dim: x_lat_dim, y_lon_dim: x_lon_dim}) #.assign_coords(**{ y_lat_dim: getattr(X, x_lat_dim), y_lon_dim: getattr(X, x_lon_dim)})

        new_x = new_x * self.eof_scores
        if x_feature_dim in new_x.dims and new_x.shape[list(new_x.dims).index(x_feature_dim)] > 1:
            new_x = new_x.stack(F=(x_feature_dim, 'mode'))
            new_x_feature = 'F'
        elif x_feature_dim in new_x.dims:
            new_x = new_x.mean(x_feature_dim)
            new_x_feature='mode'
        else:
            new_x_feature='mode'


        new_x, Y = align_chunks(new_x, Y, *self.chunks)
        self.mlr = self.model_type(**self.mlr_kwargs)
        self.mlr.fit(new_x, Y, x_lat_dim=y_lat_dim, x_lon_dim=y_lon_dim, x_sample_dim=y_sample_dim, x_feature_dim=new_x_feature, y_lat_dim=y_lat_dim, y_lon_dim=y_lon_dim, y_sample_dim=y_sample_dim, y_feature_dim=y_feature_dim, )

    def predict(self, X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None):
        x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim = guess_coords(X, x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim)
        check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        if not self.separate_members:
            flat_x = X.transpose(x_sample_dim, x_lat_dim, x_lon_dim,  x_feature_dim).stack(point=(x_lat_dim, x_lon_dim,  x_feature_dim)).sel(point=self.xpoint)
            x_data = flat_x.values
            scores = self.eof.transform(x_data)
            eof_scores = xr.DataArray(name='eof_scores', data=scores, dims=[self.y_sample_dim, 'mode'], coords={'mode': [i+1 for i in range(self.eof.eof_loadings.shape[0])], self.y_sample_dim: X.coords[x_sample_dim].values}, attrs={'generated_by': 'XCast CCA Regressor X EOF Scores'})
            new_x = xr.concat([ xr.ones_like(self.ymask) for i in range(x_data.shape[0]) ], self.y_sample_dim).assign_coords({self.y_sample_dim: X.coords[x_sample_dim].values}).swap_dims(**{ self.y_lat_dim: x_lat_dim, self.y_lon_dim: x_lon_dim})
        else:
            flat_x = X.transpose(x_sample_dim, x_lat_dim, x_lon_dim,  x_feature_dim).stack(point=(x_lat_dim, x_lon_dim,  x_feature_dim)).sel(point=self.xpoint)
            eof_scores = []
            for member, mcoord in enumerate(X.coords[x_feature_dim].values):
                flat_x2 = flat_x.sel(**{x_feature_dim: mcoord})
                x_data = flat_x2.values
                scores = self.eofs[member].transform(x_data)
                eof_scores.append(scores)

            eof_scores = np.stack(eof_scores, axis=0)
            eof_scores = xr.DataArray(name='eof_scores', data=eof_scores, dims=[x_feature_dim, self.y_sample_dim, 'mode'], coords={'mode': [i+1 for i in range(self.eofs[0].eof_scores.shape[1])], self.y_sample_dim: X.coords[x_sample_dim].values,  x_feature_dim: getattr(X, x_feature_dim)}, attrs={'generated_by': 'XCast CCA Regressor X EOF Scores'})

            new_x = xr.concat([ xr.ones_like(self.ymask) for i in range(x_data.shape[0]) ], self.y_sample_dim)
            new_x = new_x.assign_coords({self.y_sample_dim: X.coords[x_sample_dim].values}).swap_dims(**{ self.y_lat_dim: x_lat_dim, self.y_lon_dim: x_lon_dim})

        new_x = new_x * eof_scores
        if x_feature_dim in new_x.dims and new_x.shape[list(new_x.dims).index(x_feature_dim)] > 1:
            new_x = new_x.stack(F=(x_feature_dim, 'mode'))
            new_x_feature = 'F'
        elif x_feature_dim in new_x.dims:
            new_x = new_x.mean(x_feature_dim)
            new_x_feature='mode'
        else:
            new_x_feature='mode'
        preds = self.mlr.predict(new_x, x_lat_dim=self.y_lat_dim, x_lon_dim=self.y_lon_dim, x_sample_dim=self.y_sample_dim, x_feature_dim=new_x_feature)
        return preds.mean('ND').swap_dims({new_x_feature: x_feature_dim}).assign_coords({x_feature_dim: preds.coords[new_x_feature].values})

    def predict_proba(self, X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None, quantile=None):
        x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim = guess_coords(X, x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim)
        check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

        if not self.separate_members:
            flat_x = X.transpose(x_sample_dim, x_lat_dim, x_lon_dim,  x_feature_dim).stack(point=(x_lat_dim, x_lon_dim,  x_feature_dim)).sel(point=self.xpoint)
            x_data = flat_x.values
            scores = self.eof.transform(x_data)
            eof_scores = xr.DataArray(name='eof_scores', data=scores, dims=[self.y_sample_dim, 'mode'], coords={'mode': [i+1 for i in range(self.eof.eof_loadings.shape[0])], self.y_sample_dim: X.coords[x_sample_dim].values}, attrs={'generated_by': 'XCast CCA Regressor X EOF Scores'})
            new_x = xr.concat([ xr.ones_like(self.ymask) for i in range(x_data.shape[0]) ], self.y_sample_dim).assign_coords({self.y_sample_dim: X.coords[x_sample_dim].values}).swap_dims(**{ self.y_lat_dim: x_lat_dim, self.y_lon_dim: x_lon_dim})
        else:
            flat_x = X.transpose(x_sample_dim, x_lat_dim, x_lon_dim,  x_feature_dim).stack(point=(x_lat_dim, x_lon_dim,  x_feature_dim)).sel(point=self.xpoint)
            eof_scores = []
            for member, mcoord in enumerate(X.coords[x_feature_dim].values):
                flat_x2 = flat_x.sel(**{x_feature_dim: mcoord})
                x_data = flat_x2.values
                scores = self.eofs[member].transform(x_data)
                eof_scores.append(scores)

            eof_scores = np.stack(eof_scores, axis=0)
            eof_scores = xr.DataArray(name='eof_scores', data=eof_scores, dims=[x_feature_dim, self.y_sample_dim, 'mode'], coords={'mode': [i+1 for i in range(self.eofs[0].eof_scores.shape[1])], self.y_sample_dim: X.coords[x_sample_dim].values,  x_feature_dim: getattr(X, x_feature_dim)}, attrs={'generated_by': 'XCast CCA Regressor X EOF Scores'})
            new_x = xr.concat([ xr.ones_like(self.ymask) for i in range(x_data.shape[0]) ], self.y_sample_dim).assign_coords({self.y_sample_dim: X.coords[x_sample_dim].values}).swap_dims(**{ self.y_lat_dim: x_lat_dim, self.y_lon_dim: x_lon_dim})
        new_x = new_x * eof_scores
        if x_feature_dim in new_x.dims and new_x.shape[list(new_x.dims).index(x_feature_dim)] > 1:
            new_x = new_x.stack(F=(x_feature_dim, 'mode'))
            new_x_feature = 'F'
        elif x_feature_dim in new_x.dims:
            new_x = new_x.mean(x_feature_dim)
            new_x_feature='mode'
        else:
            new_x_feature='mode'
        preds = self.mlr.predict_proba(new_x, x_lat_dim=self.y_lat_dim, x_lon_dim=self.y_lon_dim, x_sample_dim=self.y_sample_dim, x_feature_dim=new_x_feature, quantile=quantile)
        return preds.mean('ND').swap_dims({new_x_feature: x_feature_dim}).assign_coords({x_feature_dim: preds.coords[new_x_feature].values})

    def scores(self, X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None, quantile=None):
        x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim = guess_coords(X, x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim)
        check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

        if not self.separate_members:
            flat_x = X.transpose(x_sample_dim, x_lat_dim, x_lon_dim,  x_feature_dim).stack(point=(x_lat_dim, x_lon_dim,  x_feature_dim)).sel(point=self.xpoint)
            x_data = flat_x.values
            scores = self.eof.transform(x_data)
            eof_scores = xr.DataArray(name='eof_scores', data=scores, dims=[self.y_sample_dim, 'mode'], coords={'mode': [i+1 for i in range(self.eof.eof_loadings.shape[0])], self.y_sample_dim: X.coords[x_sample_dim].values}, attrs={'generated_by': 'XCast CCA Regressor X EOF Scores'})
            new_x = xr.concat([ xr.ones_like(self.ymask) for i in range(x_data.shape[0]) ], self.y_sample_dim).assign_coords({self.y_sample_dim: getattr(X, x_sample_dim).values}).swap_dims(**{ self.y_lat_dim: x_lat_dim, self.y_lon_dim: x_lon_dim})
        else:
            flat_x = X.transpose(x_sample_dim, x_lat_dim, x_lon_dim,  x_feature_dim).stack(point=(x_lat_dim, x_lon_dim,  x_feature_dim)).sel(point=self.xpoint)
            eof_scores = []
            for member, mcoord in enumerate(X.coords[x_feature_dim].values):
                flat_x2 = flat_x.sel(**{x_feature_dim: mcoord})
                x_data = flat_x2.values
                scores = self.eofs[member].transform(x_data)
                eof_scores.append(scores)

            eof_scores = np.stack(eof_scores, axis=0)
            eof_scores = xr.DataArray(name='eof_scores', data=eof_scores, dims=[x_feature_dim, self.y_sample_dim, 'mode'], coords={'mode': [i+1 for i in range(self.eofs[0].eof_scores.shape[1])], self.y_sample_dim: X.coords[x_sample_dim].values,  x_feature_dim: getattr(X, x_feature_dim)}, attrs={'generated_by': 'XCast CCA Regressor X EOF Scores'})
            new_x = xr.concat([ xr.ones_like(self.ymask) for i in range(x_data.shape[0]) ], self.y_sample_dim).assign_coords({self.y_sample_dim: getattr(X, x_sample_dim).values}).swap_dims(**{ self.y_lat_dim: x_lat_dim, self.y_lon_dim: x_lon_dim})

        new_x = new_x * eof_scores
        if x_feature_dim in new_x.dims and new_x.shape[list(new_x.dims).index(x_feature_dim)] > 1:
            new_x = new_x.stack(F=(x_feature_dim, 'mode'))
            new_x_feature = 'F'
        elif x_feature_dim in new_x.dims:
            new_x = new_x.mean(x_feature_dim)
            new_x_feature='mode'
        else:
            new_x_feature='mode'
        return  eof_scores

    def show_report(self):
        assert self.separate_members, "PCA report is not available if separate_members=False "
        for feat in self.eof_loadings.coords[self.x_feature_dim].values:
            # save loadings maps
            if self.eof_modes > 1:
                el = self.eof_loadings.sel(**{self.x_feature_dim: feat}).plot(col='mode', col_wrap=self.eof_modes, subplot_kws={'projection': ccrs.PlateCarree()})
                plt.suptitle(str(feat).upper())
                for i, ax in enumerate(el.axs.flat):
                    ax.set_ylim(self.eof_loadings.coords[self.x_lat_dim].values.min(), self.eof_loadings.coords[self.x_lat_dim].values.max())
                    ax.set_xlim(self.eof_loadings.coords[self.x_lon_dim].values.min(), self.eof_loadings.coords[self.x_lon_dim].values.max())
                    ax.coastlines()
                    sd = {self.x_feature_dim: feat, 'mode': i+1}
                    ax.set_title('EOF {} ({}%)'.format(i+1, round(self.eof_variance_explained.sel(**sd ).values*100, 1)))
            else:
                ax = self.eof_loadings.sel(**{self.x_feature_dim: feat}).plot(subplot_kws={'projection': ccrs.PlateCarree()}).axes
                ax.set_ylim(self.eof_loadings.coords[self.x_lat_dim].values.min(), self.eof_loadings.coords[self.x_lat_dim].values.max())
                ax.set_xlim(self.eof_loadings.coords[self.x_lon_dim].values.min(), self.eof_loadings.coords[self.x_lon_dim].values.max())
                ax.coastlines()
                sd = {self.x_feature_dim: feat, 'mode': 1}
                ax.set_title('EOF {} ({}%)'.format(1, round(self.eof_variance_explained.sel(**sd ).values*100, 1)))
            plt.show()

            # save time series scores
            ts = self.eof_scores.sel(**{self.x_feature_dim: feat}).plot.line(hue='mode')
            plt.show()


    def report(self, filename):
        assert self.separate_members, "PCA report is not available if separate_members=False "
        with PdfPages(filename) as pdf:
            if self.separate_members:
                for feat in self.eof_loadings.coords[self.x_feature_dim].values:
                    # save loadings maps
                    if self.eof_modes > 1:
                        el = self.eof_loadings.sel(**{self.x_feature_dim: feat}).plot(col='mode', col_wrap=self.eof_modes, subplot_kws={'projection': ccrs.PlateCarree()})
                        plt.suptitle(str(feat).upper())
                        for i, ax in enumerate(el.axs.flat):
                            ax.set_ylim(self.eof_loadings.coords[self.x_lat_dim].values.min(), self.eof_loadings.coords[self.x_lat_dim].values.max())
                            ax.set_xlim(self.eof_loadings.coords[self.x_lon_dim].values.min(), self.eof_loadings.coords[self.x_lon_dim].values.max())
                            ax.coastlines()
                            sd = {self.x_feature_dim: feat, 'mode': i+1}
                            ax.set_title('EOF {} ({}%)'.format(i+1, round(self.eof_variance_explained.sel(**sd ).values*100, 1)))
                    else:
                        ax = self.eof_loadings.sel(**{self.x_feature_dim: feat}).plot(subplot_kws={'projection': ccrs.PlateCarree()}).axes
                        ax.set_ylim(self.eof_loadings.coords[self.x_lat_dim].values.min(), self.eof_loadings.coords[self.x_lat_dim].values.max())
                        ax.set_xlim(self.eof_loadings.coords[self.x_lon_dim].values.min(), self.eof_loadings.coords[self.x_lon_dim].values.max())
                        ax.coastlines()
                        sd = {self.x_feature_dim: feat, 'mode': 1}
                        ax.set_title('EOF {} ({}%)'.format(1, round(self.eof_variance_explained.sel(**sd ).values*100, 1)))
                    pdf.savefig()  # saves the current figure into a pdf page
                    plt.close()

                    # save time series scores
                    ts = self.eof_scores.sel(**{self.x_feature_dim: feat}).plot.line(hue='mode')
                    pdf.savefig()  # saves the current figure into a pdf page
                    plt.close()

            # We can also set the file's metadata via the PdfPages object:
            d = pdf.infodict()
            d['Title'] = 'XCast Principal Components Regression Report'
            d['Author'] = u'Kyle Hall'
