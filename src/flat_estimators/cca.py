from scipy.linalg import svd
import numpy as np 
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, t
from sklearn.model_selection import KFold
from collections.abc import Iterable
import datetime



def svd_flip_v(v):
    signs = []
    for i in range(v.shape[0]):
        if np.abs(np.max(v[i, :])) < np.abs(np.min(v[i, :])):
            v[i, :] *= -1
            signs.append(-1)
        else:
            signs.append(1)
    return v, np.asarray(signs)
    
class canonical_correlation_analysis:
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