from sklearn.cross_decomposition import PLSCanonical, CCA, PLSSVD
from sklearn.decomposition import PCA
import xarray as xr
import numpy as np
from scipy.linalg import svd
from ..core.utilities import guess_coords, check_all
from .prep import svd_flip_v


class sCCA:
    def __init__(self, xmodes=5, ymodes=5, ccamodes=5):
        self.xmodes, self.ymodes, self.ccamodes = xmodes, ymodes, ccamodes

    def predict(self, X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None):
        x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim = guess_coords(
            X, x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim)
        check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

        X = (X - self.xmean) / self.xstd

        flat_x = X.transpose(x_sample_dim, x_lat_dim, x_lon_dim,  x_feature_dim).stack(
            point=(x_lat_dim, x_lon_dim,  x_feature_dim)).dropna('point')
        x_data = flat_x.values
        xscores = self.xeof_.transform(x_data) / self.xeof_.singular_values_
        prj = np.dot(xscores, self.U.T) * self.canonical_correlations
        pred_yscores = np.dot(prj, self.V.T) * self.yeof_.singular_values_
        preds = self.yeof_.inverse_transform(pred_yscores)
        ret = xr.DataArray(name='predicted_values', data=preds, dims=[x_sample_dim, 'point'], coords={'point': self.ypoint_flat, x_sample_dim: X.coords[x_sample_dim]}, attrs={
                           'generated_by': 'Pure-Python CPT CCA Regressor Predicted Valuse'}).unstack('point').sortby(x_lat_dim).sortby(x_lon_dim)
        ret.attrs.update(X.attrs)
        return (ret * self.ystd + self.ymean)

    def fit(self, X, Y, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None, y_lat_dim=None, y_lon_dim=None, y_sample_dim=None, y_feature_dim=None):
        x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim = guess_coords(
            X, x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim)
        y_lat_dim, y_lon_dim, y_sample_dim,  y_feature_dim = guess_coords(
            Y, y_lat_dim, y_lon_dim, y_sample_dim,  y_feature_dim)
        check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        check_all(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)

        self.xmean, self.xstd = X.mean(x_sample_dim), X.std(x_sample_dim)
        self.ymean, self.ystd = Y.mean(y_sample_dim), Y.std(y_sample_dim)
        X = (X - self.xmean) / self.xstd
        Y = (Y - self.ymean) / self.ystd

        # integrated cosine weighting for X (predictors)
        average_latitude_width = np.sqrt(X.Y.diff(x_lat_dim).mean() ** 2) * 0.5
        R1 = X.Y - average_latitude_width
        R2 = X.Y + average_latitude_width
        xweights = (np.sin(R1/180.0*np.pi) -
                    np.sin(R2/180.0*np.pi)) / (R1 - R2)
        xweights = np.abs(xweights)
        xweights = np.sqrt(xweights)
        X = X * xweights
        xweights_flat = ((X/X).transpose(x_sample_dim, x_lat_dim, x_lon_dim,  x_feature_dim) * xweights).mean(
            x_sample_dim).stack(point=(x_lat_dim, x_lon_dim,  x_feature_dim)).dropna('point')

        # integrated cosine weighting for Y (predictands)
        average_latitude_width = np.sqrt(Y.Y.diff(y_lat_dim).mean() ** 2) * 0.5
        R1 = Y.Y - average_latitude_width
        R2 = Y.Y + average_latitude_width
        yweights = (np.sin(R1/180.0*np.pi) -
                    np.sin(R2/180.0*np.pi)) / (R1 - R2)
        yweights = np.abs(yweights)
        yweights = np.sqrt(yweights)

        Y = Y * yweights
        yweights_flat = ((Y/Y).transpose(y_sample_dim, y_lat_dim, y_lon_dim,  y_feature_dim) * yweights).mean(
            y_sample_dim).stack(point=(y_lat_dim, y_lon_dim,  y_feature_dim)).dropna('point')
        self.ylatweights = yweights

        # extract 2D [ T x Stacked(X, Y, M)] datasets from 4D [T X Y M] DataArrays, removing points in space with missing values
        flat_x = X.transpose(x_sample_dim, x_lat_dim, x_lon_dim,  x_feature_dim).stack(
            point=(x_lat_dim, x_lon_dim,  x_feature_dim)).dropna('point')
        xweights_flat = xweights_flat.sel(point=flat_x.point)

        flat_y = Y.transpose(y_sample_dim, y_lat_dim, y_lon_dim,  y_feature_dim).stack(
            point=(y_lat_dim, y_lon_dim,  y_feature_dim)).dropna('point')
        yweights_flat = yweights_flat.sel(point=flat_y.point)

        x_data, y_data = flat_x.values, flat_y.values
        self.ypoint_flat = flat_y.point

        # fit X PCA decomposition, save spatial loadings and time series scores
        self.xeof_ = PCA(n_components=self.xmodes, svd_solver='full')
        self.xeof_.fit(x_data)  # + x_data.mean(axis=0))
        Vt = self.xeof_.components_ * xweights_flat.values
        Vt, xsigns = svd_flip_v(Vt)
        self.xeof_.singular_values_
        self.xeof_.components_ *= xsigns.reshape(-1, 1)

        xscores = self.xeof_.transform(x_data) / self.xeof_.singular_values_

        # fit Y PCA decomposition, save spatial loadings and time series scores
        self.yeof_ = PCA(n_components=self.ymodes, svd_solver='full')
        self.yeof_.fit(y_data)  # + y_data.mean(axis=0))
        Vt1 = self.yeof_.components_ * yweights_flat.values
        Vt1, ysigns = svd_flip_v(Vt1)
        self.yeof_.singular_values_  # *= ysigns
        self.yeof_.components_ *= ysigns.reshape(-1, 1)
        yscores = self.yeof_.transform(y_data) / self.yeof_.singular_values_

        # Compute SVD of cross-covariance matrix
        C = np.dot(xscores.T, yscores)
        U2, s, Vt2 = svd(C, full_matrices=False)
        Vt2_, ysigns2 = svd_flip_v(Vt2.copy())
        U2_, xsigns2 = svd_flip_v(U2.T.copy())
        self.U = U2.T
        self.U = self.U[:self.ccamodes]

        self.V = Vt2[:self.ccamodes]
        self.canonical_correlations = s
        xccascores = np.dot(self.U, xscores.T) * ysigns2.reshape(-1, 1)
        yccascores = np.dot(self.V, yscores.T) * ysigns2.reshape(-1, 1)

        rwkx = self.U * self.xeof_.singular_values_
        rwky = self.V * self.yeof_.singular_values_
        x_cca_loadings = np.dot(self.xeof_.components_.T, rwkx.T) * ysigns2
        y_cca_loadings = np.dot(self.yeof_.components_.T, rwky.T) * ysigns2

        # save spatial loadings (which are maybe eof spatial loadings dot cca loadings)  and time series scores
        self.ccax_loadings = xr.DataArray(data=x_cca_loadings, dims=['point', 'ccamode'], coords={
                                          'point': flat_x.point, 'ccamode':  [i+1 for i in range(x_cca_loadings.shape[1])]}).unstack('point')
        self.ccay_loadings = xr.DataArray(data=y_cca_loadings, dims=['point', 'ccamode'], coords={
                                          'point': flat_y.point, 'ccamode':  [i+1 for i in range(y_cca_loadings.shape[1])]}).unstack('point')

        self.ccax_loadings.name = 'ccax_loadings'
        self.ccay_loadings.name = 'ccay_loadings'
        self.ccax_scores = xr.DataArray(name='ccax_scores', data=xccascores.T, dims=[x_sample_dim, 'mode'], coords={'mode': [
                                        i+1 for i in range(self.ccamodes)], x_sample_dim: X.coords[x_sample_dim]}, attrs={'generated_by': 'Pure-Python CPT CCA Regressor X CCA Scores'})
        self.ccay_scores = xr.DataArray(name='ccay_scores', data=yccascores.T, dims=[y_sample_dim, 'mode'], coords={'mode': [
                                        i+1 for i in range(self.ccamodes)], y_sample_dim: Y.coords[y_sample_dim]}, attrs={'generated_by': 'Pure-Python CPT CCA Regressor Y CCA Scores'})

        self.eofx_loadings = xr.DataArray(name='eofx_loadings', data=Vt[:self.xmodes], dims=['mode', 'point'], coords={'mode': [i+1 for i in range(
            self.xeof_.components_.shape[0])], 'point': flat_x.point}, attrs={'generated_by': 'Pure-Python CPT CCA Regressor X EOF modes'}).unstack('point')
        self.eofx_loadings.name = 'eofx_loadings'
        self.eofx_scores = xr.DataArray(name='eofx_scores', data=xscores, dims=[x_sample_dim, 'mode'], coords={'mode': [i+1 for i in range(
            self.xeof_.components_.shape[0])], x_sample_dim: X.coords[x_sample_dim]}, attrs={'generated_by': 'Pure-Python CPT CCA Regressor X EOF Scores'})

        self.eofy_loadings = xr.DataArray(name='eofy_loadings', data=Vt1[:self.ymodes], dims=['mode', 'point'], coords={'mode': [i+1 for i in range(
            self.yeof_.components_.shape[0])], 'point': flat_y.point}, attrs={'generated_by': 'Pure-Python CPT CCA Regressor Y EOF modes'}).unstack('point')
        self.eofy_loadings.name = 'eofy_loadings'
        self.eofy_scores = xr.DataArray(name='eofy_scores', data=yscores, dims=[y_sample_dim, 'mode'], coords={'mode': [i+1 for i in range(
            self.yeof_.components_.shape[0])], y_sample_dim: Y.coords[y_sample_dim]}, attrs={'generated_by': 'Pure-Python CPT CCA Regressor Y EOF Scores'})
