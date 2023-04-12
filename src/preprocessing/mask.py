import scipy.stats as ss
from ..core.utilities import check_all, guess_coords
import numpy as np
import xarray as xr 
import dask.array as da
from .onehot import quantile
import pandas as pd 

def drymask(X, dry_threshold=0.001, quantile_threshold=0.33, method='quantile', x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None):
    x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(
        X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
    check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

    if X.dims[0] != x_lat_dim or X.dims[1] != x_lon_dim or X.dims[2] != x_sample_dim or X.dims[3] != x_feature_dim:
        X = X.transpose(x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

    assert method.lower() in ['quantile', 'mean'], "Invalid method for drymask - must be one of ['quantile', 'mean']"

    mask2 = X.mean(x_feature_dim).mean(x_sample_dim)
    mask2 = xr.ones_like(mask2).where(~np.isnan(mask2), other=0) # 1 over present, 0 over missing

    if method.lower() == 'quantile':
        bn = quantile(X, quantile_threshold, method='midpoint', x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim)
        mask = bn.where(bn < dry_threshold, other=np.nan) # quant where dry, nan where not
        mask = mask + mask2
        return xr.ones_like(mask).where(np.isnan(mask), other=np.nan).mean(x_feature_dim)
    elif method.lower() == 'mean':
        bn = X.mean(x_sample_dim)
        mask = bn.where(bn < dry_threshold, other=np.nan) # quant where dry, nan where not
        mask = mask + mask2
        return xr.ones_like(mask).where(np.isnan(mask), other=np.nan).mean(x_feature_dim)

def reformat(X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None):
    x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(
        X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
    check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

    X = X.assign_coords({x_lon_dim: [ i - 360 if i > 180 else i for i in X.coords[x_lon_dim].values] }).sortby(x_lon_dim).sortby(x_lat_dim)
    return X 

def remove_climatology(X, method='monthly', x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None):
    assert method.lower() in ['monthly'], 'invalid method, must be either daily or monthly '
    x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(
        X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
    check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

    monthly_climatology = X.groupby('{}.month'.format(x_sample_dim)).mean()
    toconcat = []
    for year in sorted(list(set( [ pd.Timestamp(i).year for i in X.coords[x_sample_dim].values] ))):
        dct = {x_sample_dim: slice(pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12,31))}
        ds_yearly = X.sel(**dct).groupby('{}.month'.format(x_sample_dim)).mean() - monthly_climatology
        ds_yearly = ds_yearly.assign_coords({'month': [ pd.Timestamp(year, j, 1) for j in ds_yearly.coords['month'].values ] } ).rename({'month': x_sample_dim})
        toconcat.append(ds_yearly)
    monthly_anom = xr.concat(toconcat, x_sample_dim).sortby(x_sample_dim)
    return monthly_anom