from ...estimators.pcr import eof_
from .ests import  rEinsteinLearningMachine, rPOELM, rEPOELM
import xarray as xr
import numpy as np
from ...core.chunking import align_chunks

from collections.abc import Iterable
import scipy.linalg.lapack as la
from matplotlib.backends.backend_pdf import PdfPages
import datetime
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import dask.array as da

from ...core.utilities import guess_coords, check_all


class PCPOELM:
    def __init__(self, eof_modes=None, latitude_weighting=False, separate_members=True, chunks=(5,5), **kwargs):
        self.eof_modes = eof_modes
        self.latitude_weighting = latitude_weighting
        self.mlr_kwargs = kwargs
        self.model_type = rPOELM # this must be a class which has predict and predict_proba, and also probability of non-exceedance
        self.separate_members=separate_members
        self.chunks=chunks

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
        preds = self.mlr.predict_proba(new_x, x_lat_dim=self.y_lat_dim, x_lon_dim=self.y_lon_dim, x_sample_dim=self.y_sample_dim, x_feature_dim=new_x_feature) # cannot do flex forecast yet
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
