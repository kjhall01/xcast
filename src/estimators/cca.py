import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd
from scipy.stats import spearmanr, t
from sklearn.model_selection import KFold
from collections.abc import Iterable
import xcast as xc
from matplotlib.backends.backend_pdf import PdfPages
import datetime
import cartopy.crs as ccrs
from .pcr import std

def svd_flip_v(v):
    signs = []
    for i in range(v.shape[0]):
        if np.abs(np.max(v[i, :])) < np.abs(np.min(v[i, :])):
            v[i, :] *= -1
            signs.append(-1)
        else:
            signs.append(1)
    return v, np.asarray(signs)

class CCARegressor:
    def __init__(self, xmodes=(1, 5), ymodes=(1, 5), ccamodes=(1,5), crossvalidation_splits=5, probability_method='error_variance', latitude_weights_x=None, latitude_weights_y=None, search_override=(None, None, None)):
        self.search_override = search_override
        self.xmodes1_range = xmodes
        self.ymodes1_range = ymodes
        self.ccamodes1_range = ccamodes
        self.splits = crossvalidation_splits
        self.latitude_weights_x = latitude_weights_x
        self.latitude_weights_y = latitude_weights_y
        self.ccamodes=None
        self.probability_method = probability_method
        assert probability_method in ['error_variance', 'adjusted_error_variance'], 'invalid probability_method'


    def fit(self, x, y):
        self.y = y.copy()

        best_goodness = -999
        if self.search_override[0] is not None and self.search_override[1] is not None and self.search_override[2] is not None:
            best_combo = self.search_override
        else:
            for i in range(self.xmodes1_range[0], self.xmodes1_range[1] + 1):
                for j in range(self.ymodes1_range[0], self.ymodes1_range[1] + 1):
                    for k in range(1, min(i,j, self.ccamodes1_range[1]+1)+1):
                        self.xmodes1 = i
                        self.ymodes1 = j
                        self.ccamodes1 = k
                        combo, goodness = self._xval_fit(x, y)
                        print(combo, goodness)
                        if goodness > best_goodness:
                            best_goodness = goodness
                            best_combo = combo
        self.xmodes1 = best_combo[0]
        self.ymodes1 = best_combo[1]
        self.ccamodes1 = best_combo[2]
        bc, g = self._xval_fit(x, y) # to save the hcst_err_var
        self._fit(x, y, calc_loadings=True) # to set the loadings

    def _xval_fit(self, x, y):
        assert x.shape[0] == y.shape[0], "x and y must have the same number of samples "
        kf = KFold(n_splits=self.splits)
        indices  = np.arange(x.shape[0])

        predictions, obs = [], []
        for train_ndxs, test_ndxs in kf.split(indices):
            xtrain, ytrain = x[train_ndxs, :], y[train_ndxs, :]
            self._fit(xtrain, ytrain)
            preds = self.predict(x[test_ndxs,:], do_post=False)
            predictions.append(preds)
            obs.append(y[test_ndxs, :])
        predictions = np.vstack(predictions)
        obs = np.vstack(obs)
        self.hcst_err_var = ((predictions - obs)**2).sum(axis=0) / (self.y.shape[0] - self.ccamodes1 - 1)
        goodness, _ = spearmanr(predictions.flatten(), obs.flatten())
        return (self.xmodes, self.ymodes, self.ccamodes), goodness

    def _fit(self, x_data, y_data, calc_loadings=False):
        self.xmean = x_data.mean(axis=0)
        self.xstd = x_data.std(axis=0)

        self.ymean = y_data.mean(axis=0)
        self.ystd = y_data.std(axis=0)

        assert np.min(self.xstd) > 0, "There is a Zero Standard-Deviation period somewhere within the predictor data - could it be a drought period? Use xc.drymask?"
        assert np.min(self.ystd) > 0, "There is a Zero Standard-Deviation period somewhere within the predictand data - could it be a drought period? Use xc.drymask?"

        x_data = (x_data.copy() - self.xmean) / self.xstd
        y_data = (y_data.copy() - self.ymean) / self.ystd

        x_data = x_data * self.latitude_weights_x if self.latitude_weights_x is not None else x_data
        y_data = y_data * self.latitude_weights_y if self.latitude_weights_y is not None else y_data


        # U1 is NxP1, S = P1x1, V1t is P1xM1
        u, s, vt = svd(x_data, full_matrices=False)
        vt, xsigns = svd_flip_v(vt)
        u *= xsigns#.reshape(-1,1)
        self.x_pct_variances = s[s!=0] / s.sum()

        s = s[:self.xmodes1]
        s = s[s>0]
        self.xmodes = s[np.isfinite(1 / s)].shape[0]
        s = s[:self.xmodes]
        u = u[:, :self.xmodes ]
        vt = vt[:self.xmodes, :] * self.latitude_weights_x if self.latitude_weights_x is not None else vt[:self.xmodes, :]
        s = np.eye(self.xmodes) * s
        xscores = x_data.dot(vt.T) #u.dot(s)

        self.x_eof_scores = xscores
        self.x_eof_loadings = vt
        self.x_eof_singular_values = s
        self.x_pct_variances = self.x_pct_variances[:self.xmodes]


        uu, ss, vvt = svd(y_data, full_matrices=False)
        vvt, ysigns = svd_flip_v(vvt)
        uu *= ysigns#.reshape(-1,1)
        self.y_pct_variances = ss[ss!=0] / ss.sum()

        ss = ss[:self.ymodes1]
        ss = ss[ss>0]
        self.ymodes = ss[np.isfinite(1 / ss)].shape[0]
        ss = ss[:self.ymodes]
        uu = uu[:, :self.ymodes ]
        vvt = vvt[:self.ymodes, :] * self.latitude_weights_y if self.latitude_weights_y is not None else vvt[:self.ymodes, :]
        ss = np.eye(self.ymodes) * ss
        yscores = y_data.dot(vvt.T)#uu.dot(ss)

        self.y_eof_scores = yscores #uu.dot(ss)
        self.y_eof_loadings = vvt
        self.y_eof_singular_values = ss
        self.y_pct_variances = self.y_pct_variances[:self.ymodes]

        C1 = u.T.dot(uu)
        _, sss, _ = svd(C1, full_matrices=False)
        self.canonical_correlations =  sss

        C = xscores.T.dot(yscores)
        uuu, sss1, vvvt = svd(C, full_matrices=False)
        sss1 = sss1[:self.ccamodes1]
        sss1 = sss1[sss1>0]
        self.ccamodes = sss1[np.isfinite(1 / sss1)].shape[0]
        sss1 = sss1[:self.ccamodes]
        uuu = uuu[:, :self.ccamodes ]
        vvvt = vvvt[:self.ccamodes, :]
        sss1 = np.eye(self.ccamodes) * sss1

        self.cca_u = uuu
        self.cca_vt = vvvt
        self.canonical_correlations = self.canonical_correlations[:self.ccamodes]

        if calc_loadings:
            self.x_cca_scores = ( xscores / s[s!=0] ).dot(self.cca_u)#[:, :self.ccamodes1]
            self.y_cca_scores = ( yscores / ss[ss!=0]).dot(self.cca_vt.T)
            self.x_cca_loadings = self.x_eof_loadings.T.dot(s).dot(self.cca_u)
            self.y_cca_loadings = self.y_eof_loadings.T.dot(ss).dot(self.cca_vt.T)


    def predict(self, x_data, do_prep=True, do_post=True):
        x_data = (x_data.copy() - self.xmean) / self.xstd
        x_data = x_data * self.latitude_weights_x if self.latitude_weights_x is not None else x_data

        one = x_data.dot( self.x_eof_loadings.T )
        two = one / self.x_eof_singular_values[self.x_eof_singular_values != 0]
        three = two.dot(self.cca_u)
        four = three.dot(self.cca_vt)
        five = four * self.y_eof_singular_values[self.y_eof_singular_values != 0]
        six = five.dot(self.y_eof_loadings) #/ self.latitude_weights_y if self.latitude_weights_y is not None else five.dot(self.y_eof_loadings)
        six = six / self.latitude_weights_y**2 if self.latitude_weights_y is not None else six
        return six  * self.ystd + self.ymean


    def predict_proba(self, x_data, quantile=None, an=None, bn=None):
        x_data = (x_data.copy() - self.xmean) / self.xstd
        x_data = x_data * self.latitude_weights_x if self.latitude_weights_x is not None else x_data

        one = x_data.dot( self.x_eof_loadings.T )
        two = one / self.x_eof_singular_values[self.x_eof_singular_values != 0]
        three = two.dot(self.cca_u)
        four = three.dot(self.cca_vt)
        five = four * self.y_eof_singular_values[self.y_eof_singular_values != 0]
        six = five.dot(self.y_eof_loadings)# / self.latitude_weights_y if self.latitude_weights_y is not None else five.dot(self.y_eof_loadings)
        six = six / self.latitude_weights_y**2 if self.latitude_weights_y is not None else six
        mu = six  * self.ystd + self.ymean


        xvp = ((1 / (self.y.shape[0] - self.ccamodes1 - 1)) + (x_data.dot( self.x_eof_loadings.T ) / self.x_eof_singular_values[self.x_eof_singular_values != 0] ).dot(self.cca_u))  #*  self.canonical_correlations[self.canonical_correlations != 0]
        xvp = xvp.sum(axis=-1) **2
        pred_err_var = (1+xvp).reshape(-1, 1).dot( self.hcst_err_var.reshape(1, -1) )
        if self.probability_method == 'error_variance':
            pred_err_var = np.ones_like(pred_err_var) * self.hcst_err_var.reshape(1, -1)


        if quantile is not None:
            if not isinstance(quantile, Iterable):
                quantile = [quantile]
            ret = []
            for q in quantile:
                assert q > 0 and q < 1, 'quantile must be float between 0 and 1'
                threshold = np.nanquantile(self.y, q, axis=0)
                nn = (self.y.shape[0] - self.ccamodes1 - 1) / self.y.shape[0]
                ret.append( 1 - t.sf(threshold, self.y.shape[0] - self.ccamodes1 - 1 ,  loc=mu, scale=np.sqrt( pred_err_var)))
            return np.stack(ret, axis=0)
        else:
            if bn is None:
                bn = np.nanquantile(self.y, (1.0/3.0), axis=0)
            if an is None:
                an = np.nanquantile(self.y, (2.0/3.0), axis=0, )
            bn_prob = 1 - t.sf(bn, self.y.shape[0] - self.ccamodes1 - 1,  loc=mu, scale=np.sqrt(pred_err_var))
            an_prob = t.sf(an, self.y.shape[0]- self.ccamodes1 - 1 ,  loc=mu, scale=np.sqrt(pred_err_var))
            nn_prob = 1 - bn_prob - an_prob
            return np.stack([bn_prob, nn_prob, an_prob], axis=0)

    def prediction_error_variance(self, x_data):
        x_data = (x_data.copy() - self.xmean) / self.xstd
        x_data = x_data * self.latitude_weights_x if self.latitude_weights_x is not None else x_data

        one = x_data.dot( self.x_eof_loadings.T )
        two = one / self.x_eof_singular_values[self.x_eof_singular_values != 0]
        three = two.dot(self.cca_u)
        four = three.dot(self.cca_vt)
        five = four * self.y_eof_singular_values[self.y_eof_singular_values != 0]
        six = five.dot(self.y_eof_loadings) #/ self.latitude_weights_y if self.latitude_weights_y is not None else five.dot(self.y_eof_loadings)
        six = six / self.latitude_weights_y**2 if self.latitude_weights_y is not None else six
        mu = six  * self.ystd + self.ymean

        xvp = ((1 / (self.y.shape[0] - self.ccamodes1 - 1)) + (x_data.dot( self.x_eof_loadings.T ) / self.x_eof_singular_values[self.x_eof_singular_values != 0] ).dot(self.cca_u))  #*  self.canonical_correlations[self.canonical_correlations != 0]
        xvp = xvp.sum(axis=-1) **2
        pred_err_var = (1+xvp).reshape(-1, 1).dot( self.hcst_err_var.reshape(1, -1) )
        return pred_err_var

    def patterns(self, x_data):
        x_data = (x_data.copy() - self.xmean) / self.xstd
        x_data = x_data * self.latitude_weights_x if self.latitude_weights_x is not None else x_data
        x_eof_scores = x_data.dot( self.x_eof_loadings.T )
        x_cca_scores = (x_eof_scores / self.x_eof_singular_values[self.x_eof_singular_values != 0]).dot(self.cca_u)
        y_cca_scores = x_cca_scores .dot(self.cca_vt)
        y_eof_scores = (y_cca_scores * self.y_eof_singular_values[self.y_eof_singular_values != 0])
        return x_eof_scores, x_cca_scores, y_eof_scores


class CCA:
    def __init__(self, xmodes=(1, 5), ymodes=(1, 5), ccamodes=(1, 3), crossvalidation_splits='auto', probability_method='error_variance', latitude_weighting=False, search_override=(None, None, None)):
        self.xmodes = xmodes
        self.ymodes = ymodes
        self.ccamodes = ccamodes
        self.splits = crossvalidation_splits
        self.latitude_weighting = latitude_weighting
        self.probability_method = probability_method
        self.search_override = search_override

    def fit(self, X, Y, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None, y_lat_dim=None, y_lon_dim=None, y_sample_dim=None, y_feature_dim=None):
        x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim = xc.guess_coords(X, x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim)
        y_lat_dim, y_lon_dim, y_sample_dim,  y_feature_dim = xc.guess_coords(Y, y_lat_dim, y_lon_dim, y_sample_dim,  y_feature_dim)
        xc.check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        xc.check_all(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)

        if self.splits == 'auto':
            self.splits = X.shape[list(X.dims).index(x_sample_dim)] // 5

        self.x_feature_dim = x_feature_dim
        self.x_lat_dim, self.x_lon_dim = x_lat_dim, x_lon_dim


        self.y_feature_dim = y_feature_dim
        self.y_lat_dim, self.y_lon_dim = y_lat_dim, y_lon_dim
        ymask =  Y.mean(y_sample_dim, skipna=False).mean(y_feature_dim, skipna=False)
        self.ymask = xr.ones_like( ymask ).where(~np.isnan(ymask), other=np.nan)

        if self.latitude_weighting:
            self.xweights = np.cos(np.deg2rad(getattr(X, x_lat_dim)))
            self.xweights.name = "weights"
            self.xweights = np.sqrt(self.xweights)

            #X = X * self.xweights
            xweights_flat = (X.where(np.isnan(X), other=1).transpose(x_sample_dim, x_lat_dim, x_lon_dim,  x_feature_dim) * self.xweights).mean(x_sample_dim, skipna=False).stack(point=(x_lat_dim, x_lon_dim,  x_feature_dim))

            self.yweights = np.cos(np.deg2rad(getattr(Y, y_lat_dim)))
            self.yweights.name = "weights"
            self.yweights = np.sqrt(self.yweights)
            #Y = Y * self.yweights
            yweights_flat = (Y.where(np.isnan(Y), other=1).transpose(y_sample_dim, y_lat_dim, y_lon_dim,  y_feature_dim) * self.yweights).mean(y_sample_dim, skipna=False).stack(point=(y_lat_dim, y_lon_dim,  y_feature_dim))

        flat_x = X.transpose(x_sample_dim, x_lat_dim, x_lon_dim,  x_feature_dim).stack(point=(x_lat_dim, x_lon_dim,  x_feature_dim)).dropna('point', how='any')
        flat_y = Y.transpose(y_sample_dim, y_lat_dim, y_lon_dim,  y_feature_dim).stack(point=(y_lat_dim, y_lon_dim,  y_feature_dim)).dropna('point', how='any')
        self.xpoint = flat_x.point
        self.ypoint = flat_y.point

        if self.latitude_weighting:
            xweights_flat = xweights_flat.sel(point=self.xpoint).values
            yweights_flat = yweights_flat.sel(point=self.ypoint).values
        else:
            xweights_flat = None
            yweights_flat = None



        x_data, y_data = flat_x.values, flat_y.values
        self.ccareg = CCARegressor(xmodes=self.xmodes, ymodes=self.ymodes, ccamodes=self.ccamodes, latitude_weights_x=xweights_flat, latitude_weights_y=yweights_flat, crossvalidation_splits=self.splits, search_override=self.search_override, probability_method=self.probability_method)
        self.ccareg.fit(x_data, y_data)

        # save spatial loadings (which are maybe eof spatial loadings dot cca loadings)  and time series scores
        self.x_cca_loadings = xr.DataArray(data=self.ccareg.x_cca_loadings, dims=['point', 'mode'], coords={'point': self.xpoint, 'mode':  [i+1 for i in range(self.ccareg.x_cca_loadings.shape[1])]}).unstack('point').sortby(x_lat_dim).sortby(x_lon_dim).sortby(x_feature_dim)
        self.y_cca_loadings = xr.DataArray(data=self.ccareg.y_cca_loadings, dims=['point', 'mode'], coords={'point': self.ypoint, 'mode':  [i+1 for i in range(self.ccareg.y_cca_loadings.shape[1])]}).unstack('point').sortby(y_lat_dim).sortby(y_lon_dim).sortby(y_feature_dim)
        self.x_cca_loadings.name = 'ccax_loadings'
        self.y_cca_loadings.name = 'ccay_loadings'

        self.x_cca_scores = xr.DataArray(name='ccax_scores', data=self.ccareg.x_cca_scores, dims=[x_sample_dim, 'mode'], coords={'mode': [i+1 for i in range(self.ccareg.ccamodes)], x_sample_dim: X.coords[x_sample_dim]}, attrs={'generated_by': 'XCast CCA Regressor X CCA Scores'})
        self.y_cca_scores = xr.DataArray(name='ccay_scores', data=self.ccareg.y_cca_scores, dims=[y_sample_dim, 'mode'], coords={'mode': [i+1 for i in range(self.ccareg.ccamodes)], y_sample_dim: Y.coords[y_sample_dim]}, attrs={'generated_by': 'XCast CCA Regressor Y CCA Scores'})

        self.x_eof_loadings = xr.DataArray(name='eofx_loadings', data=self.ccareg.x_eof_loadings, dims=['mode', 'point'], coords={'mode': [i+1 for i in range(self.ccareg.x_eof_loadings.shape[0])], 'point': self.xpoint}, attrs={'generated_by': 'XCast CCA Regressor X EOF modes'}).unstack('point').sortby(x_lat_dim).sortby(x_lon_dim).sortby(x_feature_dim)
        self.x_eof_loadings.name = 'eofx_loadings'
        self.x_eof_scores = xr.DataArray(name='eofx_scores', data=self.ccareg.x_eof_scores, dims=[x_sample_dim, 'mode'], coords={'mode': [i+1 for i in range(self.ccareg.x_eof_loadings.shape[0])], x_sample_dim: X.coords[x_sample_dim]}, attrs={'generated_by': 'XCast CCA Regressor X EOF Scores'})

        self.y_eof_loadings = xr.DataArray(name='eofy_loadings', data=self.ccareg.y_eof_loadings, dims=['mode', 'point'], coords={'mode': [i+1 for i in range(self.ccareg.y_eof_loadings.shape[0])], 'point': self.ypoint}, attrs={'generated_by': 'XCast  CCA Regressor Y EOF modes'}).unstack('point').sortby(y_lat_dim).sortby(y_lon_dim).sortby(y_feature_dim)
        self.y_eof_loadings.name = 'eofy_loadings'
        self.y_eof_scores = xr.DataArray(name='eofy_scores', data=self.ccareg.y_eof_scores, dims=[y_sample_dim, 'mode'], coords={'mode': [i+1 for i in range(self.ccareg.y_eof_loadings.shape[0])], y_sample_dim: Y.coords[y_sample_dim]}, attrs={'generated_by': 'XCast CCA Regressor Y EOF Scores'})

        self.canonical_correlations = xr.DataArray(name='canonical_correlations', data=self.ccareg.canonical_correlations, dims=['mode'], coords={'mode': [i+1 for i in range(self.ccareg.ccamodes1)]}, attrs={'generated_by': 'XCast CCA Regressor CCA Correlations'})
        self.y_variance_explained = xr.DataArray(name='percent_variance', data=self.ccareg.y_pct_variances, dims=['mode'], coords={'mode': [i+1 for i in range(self.ccareg.y_pct_variances.shape[0])]}, attrs={'generated_by': 'XCast CCA Regressor PCA percent variance explained'})
        self.x_variance_explained = xr.DataArray(name='percent_variance', data=self.ccareg.x_pct_variances, dims=['mode'], coords={'mode': [i+1 for i in range(self.ccareg.x_pct_variances.shape[0])]}, attrs={'generated_by': 'XCast CCA Regressor PCA percent variance explained'})


    def predict(self, X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None):
        x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim = xc.guess_coords(
            X, x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim)
        xc.check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

        flat_x = X.transpose(x_sample_dim, x_lat_dim, x_lon_dim,  x_feature_dim).stack(point=(x_lat_dim, x_lon_dim,  x_feature_dim))
        flat_x = flat_x.sel(point=self.xpoint)
        x_data = flat_x.values

        preds = self.ccareg.predict(x_data)
        ret = xr.DataArray(name='predicted_values', data=preds, dims=[x_sample_dim, 'point'], coords={'point': self.ypoint, x_sample_dim: X.coords[x_sample_dim]}, attrs={'generated_by': 'XCAST CCA Regressor Predicted'}).unstack('point').sortby(self.y_lat_dim).sortby(self.y_lon_dim)
        ret.attrs.update(X.attrs)

        dct = { self.y_lat_dim: self.ymask.coords[self.y_lat_dim],  self.y_lon_dim: self.ymask.coords[self.y_lon_dim]  }
        ret = ret.reindex(**dct)
        return ret

    def scores(self, X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None):
        x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim = xc.guess_coords(
            X, x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim)
        xc.check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

        flat_x = X.transpose(x_sample_dim, x_lat_dim, x_lon_dim,  x_feature_dim).stack(point=(x_lat_dim, x_lon_dim,  x_feature_dim))
        flat_x = flat_x.sel(point=self.xpoint)
        x_data = flat_x.values
        x_eof_scores, x_cca_scores, y_eof_scores = self.ccareg.patterns(x_data)
        xeof = xr.DataArray(name='eofx_scores', data=x_eof_scores, dims=[x_sample_dim, 'mode'], coords={'mode': [i+1 for i in range(self.ccareg.x_eof_loadings.shape[0])], x_sample_dim: X.coords[x_sample_dim]}, attrs={'generated_by': 'XCast CCA Regressor X EOF Scores'})
        yeof = xr.DataArray(name='eofy_scores', data=y_eof_scores, dims=[x_sample_dim, 'mode'], coords={'mode': [i+1 for i in range(self.ccareg.y_eof_loadings.shape[0])], x_sample_dim: X.coords[x_sample_dim]}, attrs={'generated_by': 'XCast CCA Regressor Y EOF Scores'})
        xcca = xr.DataArray(name='cca_scores', data=x_cca_scores, dims=[x_sample_dim, 'mode'], coords={'mode': [i+1 for i in range(self.ccareg.ccamodes)], x_sample_dim: X.coords[x_sample_dim]}, attrs={'generated_by': 'XCast CCA Regressor X CCA Scores'})
        #ycca = xr.DataArray(name='ccay_scores', data=y_cca_scores, dims=[x_sample_dim, 'mode'], coords={'mode': [i+1 for i in range(self.ccareg.ccamodes)], x_sample_dim: X.coords[x_sample_dim]}, attrs={'generated_by': 'XCast CCA Regressor Y CCA Scores'})
        return xr.merge([xeof, xcca, yeof])

    def prediction_error_variance(self, X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None):
        x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim = xc.guess_coords(
            X, x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim)
        xc.check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

        flat_x = X.transpose(x_sample_dim, x_lat_dim, x_lon_dim,  x_feature_dim).stack(point=(x_lat_dim, x_lon_dim,  x_feature_dim))
        flat_x = flat_x.sel(point=self.xpoint)
        x_data = flat_x.values

        preds = self.ccareg.prediction_error_variance(x_data)
        ret = xr.DataArray(name='prediction_error_variance', data=preds, dims=[x_sample_dim, 'point'], coords={'point': self.ypoint, x_sample_dim: X.coords[x_sample_dim]}, attrs={'generated_by': 'XCAST CCA Regressor Predicted'}).unstack('point').sortby(self.y_lat_dim).sortby(self.y_lon_dim)
        ret.attrs.update(X.attrs)

        dct = { self.y_lat_dim: self.ymask.coords[self.y_lat_dim],  self.y_lon_dim: self.ymask.coords[self.y_lon_dim]  }
        ret = ret.reindex(**dct)
        return ret

    def predict_proba(self, X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None, quantile=None, bn=None, an=None):
        x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim = xc.guess_coords(
            X, x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim)
        xc.check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

        flat_x = X.transpose(x_sample_dim, x_lat_dim, x_lon_dim,  x_feature_dim).stack(point=(x_lat_dim, x_lon_dim,  x_feature_dim))
        flat_x = flat_x.sel(point=self.xpoint)
        x_data = flat_x.values


        preds = self.ccareg.predict_proba(x_data, quantile=quantile, bn=bn, an=an)
        if quantile is None:
            ret = xr.DataArray(name='predicted_values', data=preds, dims=[x_feature_dim+'_temp', x_sample_dim, 'point'], coords={'point': self.ypoint, x_sample_dim: X.coords[x_sample_dim], x_feature_dim+'_temp':['BN', 'NN', "AN"]}, attrs={'generated_by': 'XCAST CCA Regressor Predict-Proba'}).unstack('point').sortby(self.y_lat_dim).sortby(self.y_lon_dim)
        else:
            if not isinstance(quantile, Iterable):
                quantile = [quantile]
            ret = xr.DataArray(name='predicted_values', data=preds, dims=[x_feature_dim+'_temp', x_sample_dim, 'point'], coords={'point': self.ypoint, x_sample_dim: X.coords[x_sample_dim], x_feature_dim+'_temp': quantile}, attrs={'generated_by': 'XCAST CCA Regressor Non-Exceedance'}).unstack('point').sortby(self.y_lat_dim).sortby(self.y_lon_dim)
        ret.attrs.update(X.attrs)
        ret = ret.mean(x_feature_dim).rename({x_feature_dim+'_temp': x_feature_dim})

        dct = { self.y_lat_dim: self.ymask.coords[self.y_lat_dim],  self.y_lon_dim: self.ymask.coords[self.y_lon_dim]  }
        return ret.reindex(**dct)

    def report(self, filename):
        with PdfPages(filename) as pdf:
            if True: #self.separate_members:
                for feat in self.x_eof_loadings.coords[self.x_feature_dim].values:
                    # save loadings maps
                    if self.ccareg.xmodes1 > 1:
                        el = self.x_eof_loadings.sel(**{self.x_feature_dim: feat}).plot(col='mode', col_wrap=self.ccareg.xmodes1, subplot_kws={'projection': ccrs.PlateCarree()})
                        plt.suptitle(str(feat).upper())
                        for i, ax in enumerate(el.axs.flat):
                            ax.set_ylim(self.x_eof_loadings.coords[self.x_lat_dim].values.min(), self.x_eof_loadings.coords[self.x_lat_dim].values.max())
                            ax.set_xlim(self.x_eof_loadings.coords[self.x_lon_dim].values.min(), self.x_eof_loadings.coords[self.x_lon_dim].values.max())
                            ax.coastlines()
                            sd = {'mode': i+1}
                            ax.set_title('EOF {} ({}%)'.format(i+1, round(self.x_variance_explained.sel(**sd ).values*100, 1)))
                    else:
                        ax = self.x_eof_loadings.sel(**{self.x_feature_dim: feat}).plot(subplot_kws={'projection': ccrs.PlateCarree()}).axes
                        ax.set_ylim(self.x_eof_loadings.coords[self.x_lat_dim].values.min(), self.x_eof_loadings.coords[self.x_lat_dim].values.max())
                        ax.set_xlim(self.x_eof_loadings.coords[self.x_lon_dim].values.min(), self.x_eof_loadings.coords[self.x_lon_dim].values.max())
                        ax.coastlines()
                        sd = {'mode': 1}
                        ax.set_title('EOF {} ({}%)'.format(1, round(self.x_variance_explained.sel(**sd ).values*100, 1)))
                    pdf.savefig()  # saves the current figure into a pdf page
                    plt.close()

                # save time series scores
                ts = self.x_eof_scores.plot.line(hue='mode')
                pdf.savefig()  # saves the current figure into a pdf page
                plt.close()

                for feat in self.y_eof_loadings.coords[self.y_feature_dim].values:
                    # save loadings maps
                    if self.ccareg.ymodes1 > 1:
                        el = self.y_eof_loadings.sel(**{self.y_feature_dim: feat}).plot(col='mode', col_wrap=self.ccareg.ymodes1, subplot_kws={'projection': ccrs.PlateCarree()})
                        plt.suptitle(str(feat).upper())
                        for i, ax in enumerate(el.axs.flat):
                            ax.set_ylim(self.y_eof_loadings.coords[self.y_lat_dim].values.min(), self.y_eof_loadings.coords[self.y_lat_dim].values.max())
                            ax.set_xlim(self.y_eof_loadings.coords[self.y_lon_dim].values.min(), self.y_eof_loadings.coords[self.y_lon_dim].values.max())
                            ax.coastlines()
                            sd = {'mode': i+1}
                            ax.set_title('EOF {} ({}%)'.format(i+1, round(self.y_variance_explained.sel(**sd ).values*100, 1)))
                    else:
                        ax = self.y_eof_loadings.sel(**{self.y_feature_dim: feat}).plot(subplot_kws={'projection': ccrs.PlateCarree()}).axes
                        ax.set_ylim(self.y_eof_loadings.coords[self.y_lat_dim].values.min(), self.y_eof_loadings.coords[self.y_lat_dim].values.max())
                        ax.set_xlim(self.y_eof_loadings.coords[self.y_lon_dim].values.min(), self.y_eof_loadings.coords[self.y_lon_dim].values.max())
                        ax.coastlines()
                        sd = {'mode': 1}
                        ax.set_title('EOF {} ({}%)'.format(1, round(self.y_variance_explained.sel(**sd ).values*100, 1)))
                    pdf.savefig()  # saves the current figure into a pdf page
                    plt.close()

                # save time series scores
                ts = self.y_eof_scores.plot.line(hue='mode')
                pdf.savefig()  # saves the current figure into a pdf page
                plt.close()


                for feat in self.x_cca_loadings.coords[self.x_feature_dim].values:
                    # save loadings maps
                    if self.ccareg.ccamodes1 > 1:
                        el = self.x_cca_loadings.sel(**{self.x_feature_dim: feat}).plot(col='mode', col_wrap=self.ccareg.ccamodes1, subplot_kws={'projection': ccrs.PlateCarree()})
                        plt.suptitle(str(feat).upper())
                        for i, ax in enumerate(el.axs.flat):
                            ax.set_ylim(self.x_cca_loadings.coords[self.x_lat_dim].values.min(), self.x_cca_loadings.coords[self.x_lat_dim].values.max())
                            ax.set_xlim(self.x_cca_loadings.coords[self.x_lon_dim].values.min(), self.x_cca_loadings.coords[self.x_lon_dim].values.max())
                            ax.coastlines()
                            sd = {'mode': i+1}
                            ax.set_title('CCA MODE {} ({} R2)'.format(i+1, round(float(self.canonical_correlations.sel(**sd ).values), 3)))
                    else:
                        ax = self.x_cca_loadings.sel(**{self.x_feature_dim: feat}).plot( subplot_kws={'projection': ccrs.PlateCarree()}).axes
                        ax.set_ylim(self.x_cca_loadings.coords[self.x_lat_dim].values.min(), self.x_cca_loadings.coords[self.x_lat_dim].values.max())
                        ax.set_xlim(self.x_cca_loadings.coords[self.x_lon_dim].values.min(), self.x_cca_loadings.coords[self.x_lon_dim].values.max())
                        ax.coastlines()
                        sd = {'mode': 1}
                        ax.set_title('CCA MODE {} ({} R2)'.format(1, round(float(self.canonical_correlations.sel(**sd ).values), 3)))

                    pdf.savefig()  # saves the current figure into a pdf page
                    plt.close()

                # save time series scores
                ts = self.x_cca_scores.plot.line(hue='mode')
                pdf.savefig()  # saves the current figure into a pdf page
                plt.close()

                for feat in self.y_cca_loadings.coords[self.y_feature_dim].values:
                    # save loadings maps
                    if self.ccareg.ccamodes1 > 1:
                        el = self.y_cca_loadings.sel(**{self.y_feature_dim: feat}).plot(col='mode', col_wrap=self.ccareg.ccamodes1, subplot_kws={'projection': ccrs.PlateCarree()})
                        plt.suptitle(str(feat).upper())
                        for i, ax in enumerate(el.axs.flat):
                            ax.set_ylim(self.y_cca_loadings.coords[self.y_lat_dim].values.min(), self.y_cca_loadings.coords[self.y_lat_dim].values.max())
                            ax.set_xlim(self.y_cca_loadings.coords[self.y_lon_dim].values.min(), self.y_cca_loadings.coords[self.y_lon_dim].values.max())
                            ax.coastlines()
                            sd = {'mode': i+1}
                            ax.set_title('CCA MODE {} ({} R2)'.format(i+1, round(float(self.canonical_correlations.sel(**sd ).values), 3)))
                    else:
                        ax = self.y_cca_loadings.sel(**{self.y_feature_dim: feat}).plot(subplot_kws={'projection': ccrs.PlateCarree()}).axes
                        ax.set_ylim(self.y_cca_loadings.coords[self.y_lat_dim].values.min(), self.y_cca_loadings.coords[self.y_lat_dim].values.max())
                        ax.set_xlim(self.y_cca_loadings.coords[self.y_lon_dim].values.min(), self.y_cca_loadings.coords[self.y_lon_dim].values.max())
                        ax.coastlines()
                        sd = {'mode': 1}
                        ax.set_title('CCA MODE {} ({} R2)'.format(1, round(float(self.canonical_correlations.sel(**sd ).values), 3)))

                    pdf.savefig()  # saves the current figure into a pdf page
                    plt.close()

                # save time series scores
                ts = self.y_cca_scores.plot.line(hue='mode')
                pdf.savefig()  # saves the current figure into a pdf page
                plt.close()

            # We can also set the file's metadata via the PdfPages object:
            d = pdf.infodict()
            d['Title'] = 'XCast Principal Components Regression Report'
            d['Author'] = u'Kyle Hall'

    def show_report(self):
        for feat in self.x_eof_loadings.coords[self.x_feature_dim].values:
            # save loadings maps
            if self.ccareg.xmodes1 > 1:
                el = self.x_eof_loadings.sel(**{self.x_feature_dim: feat}).plot(col='mode', col_wrap=self.ccareg.xmodes1, subplot_kws={'projection': ccrs.PlateCarree()})
                plt.suptitle(str(feat).upper())
                for i, ax in enumerate(el.axs.flat):
                    ax.set_ylim(self.x_eof_loadings.coords[self.x_lat_dim].values.min(), self.x_eof_loadings.coords[self.x_lat_dim].values.max())
                    ax.set_xlim(self.x_eof_loadings.coords[self.x_lon_dim].values.min(), self.x_eof_loadings.coords[self.x_lon_dim].values.max())
                    ax.coastlines()
                    sd = {'mode': i+1}
                    ax.set_title('EOF {} ({}%)'.format(i+1, round(self.x_variance_explained.sel(**sd ).values*100, 1)))
            else:
                ax = self.x_eof_loadings.sel(**{self.x_feature_dim: feat}).plot(subplot_kws={'projection': ccrs.PlateCarree()}).axes
                ax.set_ylim(self.x_eof_loadings.coords[self.x_lat_dim].values.min(), self.x_eof_loadings.coords[self.x_lat_dim].values.max())
                ax.set_xlim(self.x_eof_loadings.coords[self.x_lon_dim].values.min(), self.x_eof_loadings.coords[self.x_lon_dim].values.max())
                ax.coastlines()
                sd = {'mode': 1}
                ax.set_title('EOF {} ({}%)'.format(1, round(self.x_variance_explained.sel(**sd ).values*100, 1)))
            plt.show()

        # save time series scores
        ts = self.x_eof_scores.plot.line(hue='mode')
        plt.show()

        for feat in self.y_eof_loadings.coords[self.y_feature_dim].values:
            # save loadings maps
            if self.ccareg.ymodes1 > 1:
                el = self.y_eof_loadings.sel(**{self.y_feature_dim: feat}).plot(col='mode', col_wrap=self.ccareg.ymodes1, subplot_kws={'projection': ccrs.PlateCarree()})
                plt.suptitle(str(feat).upper())
                for i, ax in enumerate(el.axs.flat):
                    ax.set_ylim(self.y_eof_loadings.coords[self.y_lat_dim].values.min(), self.y_eof_loadings.coords[self.y_lat_dim].values.max())
                    ax.set_xlim(self.y_eof_loadings.coords[self.y_lon_dim].values.min(), self.y_eof_loadings.coords[self.y_lon_dim].values.max())
                    ax.coastlines()
                    sd = {'mode': i+1}
                    ax.set_title('EOF {} ({}%)'.format(i+1, round(self.y_variance_explained.sel(**sd ).values*100, 1)))
            else:
                ax = self.y_eof_loadings.sel(**{self.y_feature_dim: feat}).plot(subplot_kws={'projection': ccrs.PlateCarree()}).axes
                ax.set_ylim(self.y_eof_loadings.coords[self.y_lat_dim].values.min(), self.y_eof_loadings.coords[self.y_lat_dim].values.max())
                ax.set_xlim(self.y_eof_loadings.coords[self.y_lon_dim].values.min(), self.y_eof_loadings.coords[self.y_lon_dim].values.max())
                ax.coastlines()
                sd = {'mode': 1}
                ax.set_title('EOF {} ({}%)'.format(1, round(self.y_variance_explained.sel(**sd ).values*100, 1)))
            plt.show()

        # save time series scores
        ts = self.y_eof_scores.plot.line(hue='mode')
        plt.show()


        for feat in self.x_cca_loadings.coords[self.x_feature_dim].values:
            # save loadings maps
            if self.ccareg.ccamodes1 > 1:
                el = self.x_cca_loadings.sel(**{self.x_feature_dim: feat}).plot(col='mode', col_wrap=self.ccareg.ccamodes1, subplot_kws={'projection': ccrs.PlateCarree()})
                plt.suptitle(str(feat).upper())
                for i, ax in enumerate(el.axs.flat):
                    ax.set_ylim(self.x_cca_loadings.coords[self.x_lat_dim].values.min(), self.x_cca_loadings.coords[self.x_lat_dim].values.max())
                    ax.set_xlim(self.x_cca_loadings.coords[self.x_lon_dim].values.min(), self.x_cca_loadings.coords[self.x_lon_dim].values.max())
                    ax.coastlines()
                    sd = {'mode': i+1}
                    ax.set_title('CCA MODE {} ({} R2)'.format(i+1, round(float(self.canonical_correlations.sel(**sd ).values), 3)))
            else:
                ax = self.x_cca_loadings.sel(**{self.x_feature_dim: feat}).plot( subplot_kws={'projection': ccrs.PlateCarree()}).axes
                ax.set_ylim(self.x_cca_loadings.coords[self.x_lat_dim].values.min(), self.x_cca_loadings.coords[self.x_lat_dim].values.max())
                ax.set_xlim(self.x_cca_loadings.coords[self.x_lon_dim].values.min(), self.x_cca_loadings.coords[self.x_lon_dim].values.max())
                ax.coastlines()
                sd = {'mode': 1}
                ax.set_title('CCA MODE {} ({} R2)'.format(1, round(float(self.canonical_correlations.sel(**sd ).values), 3)))
        plt.show()

        # save time series scores
        ts = self.x_cca_scores.plot.line(hue='mode')
        plt.show()

        for feat in self.y_cca_loadings.coords[self.y_feature_dim].values:
            # save loadings maps
            if self.ccareg.ccamodes1 > 1:
                el = self.y_cca_loadings.sel(**{self.y_feature_dim: feat}).plot(col='mode', col_wrap=self.ccareg.ccamodes1, subplot_kws={'projection': ccrs.PlateCarree()})
                plt.suptitle(str(feat).upper())
                for i, ax in enumerate(el.axs.flat):
                    ax.set_ylim(self.y_cca_loadings.coords[self.y_lat_dim].values.min(), self.y_cca_loadings.coords[self.y_lat_dim].values.max())
                    ax.set_xlim(self.y_cca_loadings.coords[self.y_lon_dim].values.min(), self.y_cca_loadings.coords[self.y_lon_dim].values.max())
                    ax.coastlines()
                    sd = {'mode': i+1}
                    ax.set_title('CCA MODE {} ({} R2)'.format(i+1, round(float(self.canonical_correlations.sel(**sd ).values), 3)))
            else:
                ax = self.y_cca_loadings.sel(**{self.y_feature_dim: feat}).plot(subplot_kws={'projection': ccrs.PlateCarree()}).axes
                ax.set_ylim(self.y_cca_loadings.coords[self.y_lat_dim].values.min(), self.y_cca_loadings.coords[self.y_lat_dim].values.max())
                ax.set_xlim(self.y_cca_loadings.coords[self.y_lon_dim].values.min(), self.y_cca_loadings.coords[self.y_lon_dim].values.max())
                ax.coastlines()
                sd = {'mode': 1}
                ax.set_title('CCA MODE {} ({} R2)'.format(1, round(float(self.canonical_correlations.sel(**sd ).values), 3)))

            plt.show()

        # save time series scores
        ts = self.y_cca_scores.plot.line(hue='mode')
        plt.show()


secret = """
    pl = cca.x_eof_loadings.plot(subplot_kws={'projection': ccrs.PlateCarree()}, col='mode', col_wrap=cca.x_eof_loadings.shape[list(cca.x_eof_loadings.dims).index('mode')])
    for ax in pl.axs.flat:
        coasts = ax.coastlines()
    plt.suptitle('Model EOFs - {}'.format(gcms[i].upper()))

    pl = cca.y_eof_loadings.plot(subplot_kws={'projection': ccrs.PlateCarree()}, col='mode', col_wrap=cca.y_eof_loadings.shape[list(cca.y_eof_loadings.dims).index('mode')])
    for ax in pl.axs.flat:
        coasts = ax.coastlines()
    plt.suptitle('Observed EOFs - {}'.format(observations))

    scores = xr.concat([cca.x_eof_scores, cca.y_eof_scores], 'FIELD').assign_coords({'FIELD': ['X', 'Y']})
    for j in range(max(cca.xmodes1, cca.ymodes1)):
        scores.isel(mode=j).plot.line(hue='FIELD')

    pl = cca.x_cca_loadings.plot(subplot_kws={'projection': ccrs.PlateCarree()}, col='mode', col_wrap=cca.x_cca_loadings.shape[list(cca.x_cca_loadings.dims).index('mode')])
    for ax in pl.axs.flat:
        coasts = ax.coastlines()
    plt.suptitle('Model CCA Loadings - {}'.format(gcms[i].upper()))

    pl = cca.y_cca_loadings.plot(subplot_kws={'projection': ccrs.PlateCarree()}, col='mode', col_wrap=cca.y_cca_loadings.shape[list(cca.y_cca_loadings.dims).index('mode')])
    for ax in pl.axs.flat:
        coasts = ax.coastlines()
    plt.suptitle('Observation CCA Loadings - {}'.format(observations)) """


if __name__ == "__main__":
    import cptcore as cc
    X, Y = cc.load_southasia_nmme()
    #Y = xr.open_dataset('enacts-bd-jjas.nc').prec.expand_dims({'M':[0]})
    X = X.expand_dims({'M':[0]})
    Y = Y.expand_dims({'M':[0]})

    #X = X.sel(X=slice(80, 100), Y=slice(20, 28)
    cca = CCA()
    cca.fit(X, Y)
    p = cca.predict(X)
    probs = cca.predict_proba(X)
