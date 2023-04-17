import xarray as xr
import numpy as np
import dask.array as da
import uuid
import sys
import os
from ..core.utilities import check_all, check_xyt_compatibility, guess_coords, shape
from ..core.chunking import align_chunks
import dask.diagnostics as dd
from ..flat_estimators.wrappers import nan_classifier, rf_classifier, nan_regression
from ..flat_estimators.einstein_elm import extreme_learning_machine
from ..flat_estimators.einstein_epoelm import epoelm
from sklearn.decomposition import PCA
from collections.abc import Iterable
from .tuning import DFS, get_score


def apply_tune_to_block(x_data, y_data, mme=epoelm, scorer=get_score, ND=1, queue_len=5, generation_size=5, n_mutations=2, lag=10, tol=0.001, kwargs={}):
    x_data2 = x_data.reshape(x_data.shape[0]*x_data.shape[1]*x_data.shape[2], x_data.shape[3] )
    y_data2 = y_data.reshape(y_data.shape[0]*y_data.shape[1]*y_data.shape[2], y_data.shape[3] )
    x_data2 = x_data2[~np.isnan(np.sum(y_data2, axis=1)) ]
    y_data2 = y_data2[~np.isnan(np.sum(y_data2, axis=1)) ]
    if y_data2.shape[0] > 0:
        params, score, _ = DFS(x_data2, y_data2, queue_len=queue_len, generation_size=generation_size, n_mutations=n_mutations, lag=lag, tol=tol,  estimator=mme, scorer=scorer, gene_set=kwargs)
    else:
        params, score = {}, np.nan
    #print(params, score)
    models = np.empty(
        (x_data.shape[0], x_data.shape[1], ND), dtype=np.dtype('O'))
    scores = np.empty(
        (x_data.shape[0], x_data.shape[1], 1), dtype=np.dtype('O'))
    params2 = np.empty(
        (x_data.shape[0], x_data.shape[1], 1), dtype=np.dtype('O'))
    for i in range(x_data.shape[0]):
        for j in range(x_data.shape[1]):
            x_train = x_data[i, j, :, :]
            y_train = y_data[i, j, :, :]
            if np.isnan(np.min(x_train)) or np.isnan(np.min(y_train)):
                temp_mme = nan_classifier
            else:
                temp_mme = mme
            if len(x_train.shape) < 2:
                x_train = x_train.reshape(-1, 1)
            if len(y_train.shape) < 2:
                y_train = y_train.reshape(-1, 1)
            for k in range(ND):
                models[i][j][k] = temp_mme(**params)
                models[i][j][k].fit(x_train, y_train)
            scores[i][j][0] = score
            params2[i][j][0] = params
    return np.concatenate([scores, params2], axis=2)


def apply_fit_to_block(x_data, y_data, mme=epoelm, ND=1, kwargs={}):
    models = np.empty(
        (x_data.shape[0], x_data.shape[1], ND), dtype=np.dtype('O'))
    for i in range(x_data.shape[0]):
        for j in range(x_data.shape[1]):
            x_train = x_data[i, j, :, :]
            y_train = y_data[i, j, :, :]
            if np.isnan(np.min(x_train)) or np.isnan(np.min(y_train)):
                temp_mme = nan_classifier
            else:
                temp_mme = mme
            if len(x_train.shape) < 2:
                x_train = x_train.reshape(-1, 1)
            if len(y_train.shape) < 2:
                y_train = y_train.reshape(-1, 1)
            for k in range(ND):
                models[i][j][k] = temp_mme(**kwargs)
                models[i][j][k].fit(x_train, y_train)
    return models


def apply_hyperfit_to_block(x_data, y_data, models, mme=epoelm, ND=1, kwargs={}):
    for i in range(x_data.shape[0]):
        for j in range(x_data.shape[1]):
            x_train = x_data[i, j, :, :]
            y_train = y_data[i, j, :, :]
            if len(x_train.shape) < 2:
                x_train = x_train.reshape(-1, 1)
            if len(y_train.shape) < 2:
                y_train = y_train.reshape(-1, 1)
            if np.isnan(np.min(x_train)) or np.isnan(np.min(y_train)):
                models[i][j] = nan_classifier()
            else:
                models[i][j] = mme(**models[i][j])
            models[i][j].fit(x_train, y_train)
    return models


def apply_fit_x_to_block(x_data, mme=PCA, ND=1, kwargs={}):
    models = np.empty(
        (x_data.shape[0], x_data.shape[1], ND), dtype=np.dtype('O'))
    for i in range(x_data.shape[0]):
        for j in range(x_data.shape[1]):
            x_train = x_data[i, j, :, :]
            if np.isnan(np.min(x_train)):
                temp_mme = nan_classifier
            else:
                temp_mme = mme
            if len(x_train.shape) < 2:
                x_train = x_train.reshape(-1, 1)
            for k in range(ND):
                models[i][j][k] = temp_mme(**kwargs)
                models[i][j][k].fit(x_train)
    return models


def apply_predict_proba_to_block(x_data, models, kwargs={}):
    if 'n_out' in kwargs.keys():
        n_out = kwargs['n_out']
        kwargs = {k: v for k, v in kwargs.items() if k != 'n_out'}
    else:
        n_out = 3

    ret = np.empty((x_data.shape[0], x_data.shape[1],
                   models.shape[2], x_data.shape[2], n_out), dtype=float)
    for i in range(x_data.shape[0]):
        for j in range(x_data.shape[1]):
            x_train = x_data[i, j, :, :]
            if len(x_train.shape) < 2:
                x_train = x_train.reshape(-1, 1)
            for k in range(models.shape[2]):
                if isinstance(models[i][j][k],  nan_classifier):
                    ret[i, j, k, :, :] = models[i][j][k].predict_proba(
                        x_train, n_out=n_out, **kwargs)
                else:
                    ret[i, j, k, :, :] = models[i][j][k].predict_proba(
                        x_train, **kwargs)
    return np.asarray(ret)


def apply_transform_to_block(x_data, models, kwargs={}):
    ret = []
    for i in range(x_data.shape[0]):
        ret.append([])
        for j in range(x_data.shape[1]):
            ret[i].append([])
            x_train = x_data[i, j, :, :]
            if len(x_train.shape) < 2:
                x_train = x_train.reshape(-1, 1)
            for k in range(models.shape[2]):
                ret1 = models[i][j][k].transform(x_train, **kwargs)
                ret[i][j].append(ret1)
    return np.asarray(ret)


def apply_predict_to_block(x_data, models, kwargs={}):
    if 'n_out' in kwargs.keys():
        n_out = kwargs['n_out']
        kwargs = {k: v for k, v in kwargs.items() if k != 'n_out'}
    else:
        n_out = 1
    ret = np.empty((x_data.shape[0], x_data.shape[1],
                   models.shape[2], x_data.shape[2], n_out), dtype=float)
    for i in range(x_data.shape[0]):
        for j in range(x_data.shape[1]):
            x_train = x_data[i, j, :, :]
            if len(x_train.shape) < 2:
                x_train = x_train.reshape(-1, 1)
            for k in range(models.shape[2]):
                if isinstance(models[i][j][k],  nan_classifier):
                    ret1 = models[i][j][k].predict(
                        x_train, n_out=n_out, **kwargs)
                else:
                    ret1 = models[i][j][k].predict(x_train, **kwargs)
                if len(ret1.shape) < 2:
                    ret1 = np.expand_dims(ret1, axis=1)
                ret[i, j, k, :, :] = ret1
    return np.asarray(ret)


class BaseEstimator:
    """ BaseEstimator class
    implements .fit(X, Y) and, .predict_proba(X), .predict(X)
    can be sub-classed to extend to new statistical methods
    new methods must implement .fit(x, y) and .predict(x)
    and then sub-class's .model_type must be set to the constructor of the new method """

    def __init__(self, client=None, lat_chunks=1, lon_chunks=1, verbose=False, params=None, **kwargs):
        self.model_type = epoelm
        self.models_, self.ND = None, 1
        self.client, self.kwargs = client, kwargs
        self.verbose = verbose
        self.lat_chunks, self.lon_chunks = lat_chunks, lon_chunks
        self.latitude, self.longitude, self.features = None, None, None
        self.params = params.copy() if params is not None else None

    def fit(self, X, *args,  x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None, y_lat_dim=None, y_lon_dim=None, y_sample_dim=None, y_feature_dim=None, rechunk=True):
        if len(args) > 0:
            assert len(args) < 2, 'too many args'
            Y = args[0]

            x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(
                X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
            y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim = guess_coords(
                Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
            check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
            check_all(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
            check_xyt_compatibility(X, Y, x_lat_dim, x_lon_dim, x_sample_dim,
                                    x_feature_dim, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
            self.latitude, self.longitude, _, self.features = shape(
                X, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim)

            if X.dims[0] != x_lat_dim or X.dims[1] != x_lon_dim or X.dims[2] != x_sample_dim or X.dims[3] != x_feature_dim:
                X1 = X.transpose(x_lat_dim, x_lon_dim,
                                 x_sample_dim, x_feature_dim)
            else:
                X1 = X

            if Y.dims[0] != y_lat_dim or Y.dims[1] != y_lon_dim or Y.dims[2] != y_sample_dim or Y.dims[3] != y_feature_dim:
                Y1 = Y.transpose(y_lat_dim, y_lon_dim,
                                 y_sample_dim, y_feature_dim)
            else:
                Y1 = Y

            if rechunk:
                X1, Y1 = align_chunks(X1, Y1,  self.lat_chunks, self.lon_chunks, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim,
                                      x_feature_dim=x_feature_dim, y_lat_dim=y_lat_dim, y_lon_dim=y_lon_dim, y_sample_dim=y_sample_dim, y_feature_dim=y_feature_dim)
            if self.params is not None:
                if self.params.dims[0] != x_lat_dim or self.params.dims[1] != x_lon_dim:
                    self.params= self.params.transpose(x_lat_dim, x_lon_dim)
                else:
                    self.params = self.params
                models_data = self.params.data

                x_data = X1.data
                y_data = Y1.data
                if not isinstance(x_data, da.core.Array):
                    x_data = da.from_array(x_data)
                if not isinstance(y_data, da.core.Array):
                    y_data = da.from_array(y_data, chunks=x_data.chunksize)
                if not isinstance(models_data, da.core.Array):
                    models_data = da.from_array(models_data, chunks=y_data.chunksize[:2])

                self.models_ = da.blockwise(apply_hyperfit_to_block, 'ij', x_data, 'ijkl', y_data, 'ijkm', models_data, 'ij', concatenate=True, kwargs=self.kwargs, meta=np.array((), dtype=np.dtype('O'))).persist()
                if type(self.models_) == np.ndarray:
                    self.models_ = da.from_array(self.models_, chunks=(max(
                        self.latitude // self.lat_chunks, 1), max(self.longitude // self.lon_chunks, 1), self.ND))
                self.models_ = da.stack( [self.models_ for ijk in range(self.ND)], axis=-1)
                #self.models = xr.DataArray(name='models', data=self.models_, dims=[x_lat_dim, x_lon_dim], coords={x_lat_dim: X1.coords[x_lat_dim].values, x_lon_dim: X1.coords[x_lon_dim].values })

            else: 
                x_data = X1.data
                y_data = Y1.data
                if not isinstance(x_data, da.core.Array):
                    x_data = da.from_array(x_data)
                if not isinstance(y_data, da.core.Array):
                    y_data = da.from_array(y_data)

                if self.verbose:
                    with dd.ProgressBar():
                        self.models_ = da.blockwise(apply_fit_to_block, 'ijn', x_data, 'ijkl', y_data, 'ijkm', new_axes={
                            'n': self.ND}, mme=self.model_type, ND=self.ND, concatenate=True, kwargs=self.kwargs, meta=np.array((), dtype=np.dtype('O'))).persist()
                else:
                    self.models_ = da.blockwise(apply_fit_to_block, 'ijn', x_data, 'ijkl', y_data, 'ijkm', new_axes={
                        'n': self.ND}, mme=self.model_type, ND=self.ND, concatenate=True, kwargs=self.kwargs, meta=np.array((), dtype=np.dtype('O'))).persist()
                if type(self.models_) == np.ndarray:
                    self.models_ = da.from_array(self.models_, chunks=(max(
                        self.latitude // self.lat_chunks, 1), max(self.longitude // self.lon_chunks, 1), self.ND))
                #self.models = xr.DataArray(name='models', data=self.models_, dims=[x_lat_dim, x_lon_dim, 'ND'], coords={x_lat_dim: X1.coords[x_lat_dim].values, x_lon_dim: X1.coords[x_lon_dim].values, 'ND': [iii+1 for iii in range(self.ND)]})
        else:
            x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(
                X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
            check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
            self.latitude, self.longitude, _, self.features = shape(
                X, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim)

            X1 = X.transpose(x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

            x_data = X1.data
            if not isinstance(x_data, da.core.Array):
                x_data = da.from_array(x_data)

            if self.verbose:
                with dd.ProgressBar():
                    self.models_ = da.blockwise(apply_fit_x_to_block, 'ijn', x_data, 'ijkl', new_axes={
                        'n': self.ND}, mme=self.model_type, concatenate=True, ND=self.ND, kwargs=self.kwargs, meta=np.array((), dtype=np.dtype('O'))).persist()
            else:
                self.models_ = da.blockwise(apply_fit_x_to_block, 'ijn', x_data, 'ijkl',  new_axes={
                    'n': self.ND}, mme=self.model_type, concatenate=True, ND=self.ND, kwargs=self.kwargs, meta=np.array((), dtype=np.dtype('O'))).persist()

            if type(self.models_) == np.ndarray:
                self.models_ = da.from_array(self.models_, chunks=(max(
                    self.latitude // self.lat_chunks, 1), max(self.longitude // self.lon_chunks, 1), self.ND))
        self.models = xr.DataArray(name='models', data=self.models_, dims=[x_lat_dim, x_lon_dim, 'ND'], coords={x_lat_dim: X1.coords[x_lat_dim].values, x_lon_dim: X1.coords[x_lon_dim].values, 'ND': [iii+1 for iii in range(self.ND)]})


    def tune(self, X, Y,  x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None, y_lat_dim=None, y_lon_dim=None, y_sample_dim=None, y_feature_dim=None, rechunk="1X1", queue_len=5, gene_set=None, generation_size=5, n_mutations=2, lag=10, tol=0.001, scorer=get_score, **kwargs):
        x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(
            X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim = guess_coords(
            Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
        check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        check_all(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
        check_xyt_compatibility(X, Y, x_lat_dim, x_lon_dim, x_sample_dim,
                                x_feature_dim, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
        self.latitude, self.longitude, _, self.features = shape(
            X, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim)

        if X.dims[0] != x_lat_dim or X.dims[1] != x_lon_dim or X.dims[2] != x_sample_dim or X.dims[3] != x_feature_dim:
            X1 = X.transpose(x_lat_dim, x_lon_dim,
                                x_sample_dim, x_feature_dim)
        else:
            X1 = X

        if Y.dims[0] != y_lat_dim or Y.dims[1] != y_lon_dim or Y.dims[2] != y_sample_dim or Y.dims[3] != y_feature_dim:
            Y1 = Y.transpose(y_lat_dim, y_lon_dim,
                                y_sample_dim, y_feature_dim)
        else:
            Y1 = Y

        if rechunk.upper() == "1X1":
            X1, Y1 = align_chunks(X1, Y1,  self.latitude, self.longitude, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim,
                                    x_feature_dim=x_feature_dim, y_lat_dim=y_lat_dim, y_lon_dim=y_lon_dim, y_sample_dim=y_sample_dim, y_feature_dim=y_feature_dim)
        elif rechunk:
            X1, Y1 = align_chunks(X1, Y1,  self.lat_chunks, self.lon_chunks, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim,
                                    x_feature_dim=x_feature_dim, y_lat_dim=y_lat_dim, y_lon_dim=y_lon_dim, y_sample_dim=y_sample_dim, y_feature_dim=y_feature_dim)
        else:
            pass

        x_data = X1.data
        y_data = Y1.data
        if not isinstance(x_data, da.core.Array):
            x_data = da.from_array(x_data)
        if not isinstance(y_data, da.core.Array):
            y_data = da.from_array(y_data)

        ret_= da.blockwise(apply_tune_to_block, 'ijn', x_data, 'ijkl', y_data, 'ijkm', new_axes={'n': 2}, mme=self.model_type, scorer=scorer, ND=self.ND, queue_len=queue_len, generation_size=generation_size, n_mutations=n_mutations, lag=lag, tol=tol, concatenate=True, kwargs=kwargs, meta=np.array((), dtype=np.dtype('O'))).persist()
        
        if type(self.models_) == np.ndarray:
            ret_ = da.from_array(ret_, chunks=(max(self.latitude // self.lat_chunks, 1), max(self.longitude // self.lon_chunks, 1), self.ND))
        scores = ret_[:,:,0].astype(float)
        params = ret_[:,:,1]
        scores = xr.DataArray(name='goodness', data=scores, dims=['latitude', 'longitude'], coords={'latitude': getattr(X1, x_lat_dim), 'longitude': getattr(X1, x_lon_dim)})
        params = xr.DataArray(name='params', data=params, dims=['latitude', 'longitude'], coords={'latitude': getattr(X1, x_lat_dim), 'longitude': getattr(X1, x_lon_dim)})
        return xr.merge([scores, params])


    def predict_proba(self, X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None, rechunk=True, **kwargs):
        x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(
            X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        xlat, xlon, xsamp, xfeat = shape(
            X, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim)

        assert xlat == self.latitude, 'XCast Estimators require new predictors to have the same dimensions as the training data- lat mismatch'
        assert xlon == self.longitude, 'XCast Estimators require new predictors to have the same dimensions as the training data- lon mismatch'
        assert xfeat == self.features, 'XCast Estimators require new predictors to have the same dimensions as the training data- feat mismatch'

        if 'n_out' not in kwargs.keys():
            if 'quantile' in kwargs.keys() and kwargs['quantile'] is not None:
                if not isinstance(kwargs['quantile'], Iterable):
                    kwargs['quantile'] = [kwargs['quantile']]
                kwargs['n_out'] = len(kwargs['quantile'])
            elif 'threshold' in kwargs.keys() and kwargs['threshold'] is not None:
                if not isinstance(kwargs['threshold'], Iterable):
                    kwargs['threshold'] = [kwargs['threshold']]
                kwargs['n_out'] = len(kwargs['threshold'])
            else:
                kwargs['n_out'] = 3

        if rechunk:
            X1 = X.chunk({x_lat_dim: max(xlat // self.lat_chunks, 1), x_lon_dim: max(xlon //
                         self.lon_chunks, 1)}).transpose(x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        else:
            if X.dims[0] != x_lat_dim or X.dims[1] != x_lon_dim or X.dims[2] != x_sample_dim or X.dims[3] != x_feature_dim:
                X1 = X.transpose(x_lat_dim, x_lon_dim,
                                 x_sample_dim, x_feature_dim)
            else:
                X1 = X

        x_data = X1.data
        if self.verbose:
            with dd.ProgressBar():
                results = da.blockwise(apply_predict_proba_to_block, 'ijnkm', x_data, 'ijkl', self.models_, 'ijn', new_axes={
                    'm': kwargs['n_out']},  dtype=float, concatenate=True, kwargs=kwargs).persist()
        else:
            results = da.blockwise(apply_predict_proba_to_block, 'ijnkm', x_data, 'ijkl', self.models_, 'ijn', new_axes={
                    'm': kwargs['n_out']},  dtype=float, concatenate=True, kwargs=kwargs).persist()


        feature_coords = [i for i in range(kwargs['n_out'])]
        if 'quantile' in kwargs.keys() and kwargs['quantile'] is not None:
            feature_coords = kwargs['quantile']
        if 'threshold' in kwargs.keys() and kwargs['quantile'] is not None:
            feature_coords = kwargs['threshold']
        coords = {
            x_lat_dim: X1.coords[x_lat_dim].values,
            x_lon_dim: X1.coords[x_lon_dim].values,
            x_sample_dim: X1.coords[x_sample_dim].values,
            x_feature_dim:  feature_coords,
            'ND': [i for i in range(self.ND)]
        }

        dims = [x_lat_dim, x_lon_dim, 'ND', x_sample_dim, x_feature_dim]
        attrs = X1.attrs
        attrs.update(
            {'generated_by': 'XCast Classifier - {}'.format(self.model_type)})
        return xr.DataArray(name='predicted_probability', data=results, coords=coords, dims=dims, attrs=attrs).mean('ND')

    def transform(self, X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None, rechunk=True, **kwargs):
        if 'n_out' not in kwargs.keys():
            assert 'n_components' in kwargs.keys(), 'if you dont pass n_components, you must pass n_out'
            kwargs['n_out'] = kwargs['n_components']
        x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(
            X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

        xlat, xlon, xsamp, xfeat = shape(
            X, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim)

        assert xlat == self.latitude, 'XCast Estimators require new predictors to have the same dimensions as the training data- lat mismatch'
        assert xlon == self.longitude, 'XCast Estimators require new predictors to have the same dimensions as the training data- lon mismatch'
        assert xfeat == self.features, 'XCast Estimators require new predictors to have the same dimensions as the training data- feat mismatch'

        if rechunk:
            X1 = X.chunk({x_lat_dim: max(xlat // self.lat_chunks, 1), x_lon_dim: max(xlon //
                         self.lon_chunks, 1)}).transpose(x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        else:
            if X.dims[0] != x_lat_dim or X.dims[1] != x_lon_dim or X.dims[2] != x_sample_dim or X.dims[3] != x_feature_dim:
                X1 = X.transpose(x_lat_dim, x_lon_dim,
                                 x_sample_dim, x_feature_dim)
            else:
                X1 = X

        x_data = X1.data
        if self.verbose:
            with dd.ProgressBar():
                results = da.blockwise(apply_transform_to_block, 'ijnkm', x_data, 'ijkl', self.models_, 'ijn', new_axes={
                                       'm': kwargs['n_out']}, dtype=float, concatenate=True, kwargs=kwargs).persist()
        else:
            results = da.blockwise(apply_transform_to_block, 'ijnkm', x_data, 'ijkl', self.models_, 'ijn', new_axes={
                                   'm': kwargs['n_out']}, dtype=float, concatenate=True, kwargs=kwargs).persist()

        coords = {
            x_lat_dim: X1.coords[x_lat_dim].values,
            x_lon_dim: X1.coords[x_lon_dim].values,
            x_sample_dim: X1.coords[x_sample_dim].values,
            x_feature_dim: [i for i in range(kwargs['n_out'])],
            'ND': [i for i in range(self.ND)]
        }

        dims = [x_lat_dim, x_lon_dim, 'ND', x_sample_dim, x_feature_dim]
        attrs = X1.attrs
        attrs.update(
            {'generated_by': 'XCast Classifier - {}'.format(self.model_type)})
        return xr.DataArray(name='transformed', data=results, coords=coords, dims=dims, attrs=attrs).mean('ND')

    def predict(self, X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None, rechunk=True, **kwargs):
        x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(
            X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        xlat, xlon, xsamp, xfeat = shape(
            X, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim)
        assert xlat == self.latitude, 'XCast Estimators require new predictors to have the same dimensions as the training data- lat mismatch'
        assert xlon == self.longitude, 'XCast Estimators require new predictors to have the same dimensions as the training data- lon mismatch'
        assert xfeat == self.features, 'XCast Estimators require new predictors to have the same dimensions as the training data- feat mismatch'
        if 'n_out' not in kwargs.keys():
            if 'quantile' in kwargs.keys() and kwargs['quantile'] is not None:
                if not isinstance(kwargs['quantile'], Iterable):
                    kwargs['quantile'] = [kwargs['quantile']]
                kwargs['n_out'] = len(kwargs['quantile'])
            elif 'threshold' in kwargs.keys() and kwargs['threshold'] is not None:
                if not isinstance(kwargs['threshold'], Iterable):
                    kwargs['threshold'] = [kwargs['threshold']]
                kwargs['n_out'] = len(kwargs['threshold'])
            else:
                kwargs['n_out'] = 1
        if rechunk:
            X1 = X.chunk({x_lat_dim: max(xlat // self.lat_chunks, 1), x_lon_dim: max(xlon //
                         self.lon_chunks, 1)}).transpose(x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        else:
            if X.dims[0] != x_lat_dim or X.dims[1] != x_lon_dim or X.dims[2] != x_sample_dim or X.dims[3] != x_feature_dim:
                X1 = X.transpose(x_lat_dim, x_lon_dim,
                                 x_sample_dim, x_feature_dim)
            else:
                X1 = X

        x_data = X1.data
        if self.verbose:
            with dd.ProgressBar():
                results = da.blockwise(apply_predict_to_block, 'ijnkm', x_data, 'ijkl', self.models_, 'ijn', new_axes={
                                       'm': kwargs['n_out']}, dtype=float, concatenate=True, kwargs=kwargs).persist()
        else:
            results = da.blockwise(apply_predict_to_block, 'ijnkm', x_data, 'ijkl', self.models_, 'ijn', new_axes={
                                   'm': kwargs['n_out']}, dtype=float, concatenate=True, kwargs=kwargs).persist()
        feature_coords = [i for i in range(kwargs['n_out'])]
        if 'quantile' in kwargs.keys() and kwargs['quantile'] is not None:
            feature_coords = kwargs['quantile']
        if 'threshold' in kwargs.keys() and kwargs['quantile'] is not None:
            feature_coords = kwargs['threshold']
        coords = {
            x_lat_dim: X1.coords[x_lat_dim].values,
            x_lon_dim: X1.coords[x_lon_dim].values,
            x_sample_dim: X1.coords[x_sample_dim].values,
            x_feature_dim: feature_coords,
            'ND': [i for i in range(self.ND)]
        }

        dims = [x_lat_dim, x_lon_dim, 'ND', x_sample_dim, x_feature_dim]
        attrs = X1.attrs
        attrs.update(
            {'generated_by': 'XCast Classifier - {}'.format(self.model_type)})
        return xr.DataArray(name='predicted', data=results, coords=coords, dims=dims, attrs=attrs).mean('ND')
