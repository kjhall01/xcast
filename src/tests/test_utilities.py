import pytest
from .. import * 
import xarray as xr
import numpy as np

def test_regrid_coarse():
	# Hindcast Files
	cans_hcst = xr.open_dataset(Path(__file__).absolute().parents[0] /'test_data/SASCOF/cansipsv2_hcst.nc', decode_times=False, chunks='auto')
	cfsv2_hcst = xr.open_dataset(Path(__file__).absolute().parents[0] /'test_data/SASCOF/cfsv2_hcst.nc', decode_times=False, chunks='auto')
	cola_hcst = xr.open_dataset(Path(__file__).absolute().parents[0] /'test_data/SASCOF/cola_hcst.nc', decode_times=False, chunks='auto')
	nasa_hcst = xr.open_dataset(Path(__file__).absolute().parents[0] /'test_data/SASCOF/nasa_hcst.nc', decode_times=False, chunks='auto')
	X = xr.concat([cans_hcst, cfsv2_hcst, cola_hcst, nasa_hcst], 'M').assign_coords({'M':[0,1,2,3]}).mean('L').prec

	# Observation File
	Y = xr.open_dataset('test_data/SASCOF/observed_rainfall.nc', decode_times=False, chunks='auto').expand_dims({'M':[0]}).rf / (30.33)
	# first, demonstrate regridding to a coarser grid
	lats2x2 = Y.coords['LATITUDE'].values[::2]
	lons2x2 = Y.coords['LONGITUDE'].values[::2]
	regridded2x2 = regrid(Y, lons2x2, lats2x2, x_lat_dim='LATITUDE', x_lon_dim='LONGITUDE', x_sample_dim='time', x_feature_dim='M').isel(time=-1, M=0)

def test_regrid_fine():
	cans_hcst = xr.open_dataset(Path(__file__).absolute().parents[0] /'test_data/SASCOF/cansipsv2_hcst.nc', decode_times=False, chunks='auto')
	cfsv2_hcst = xr.open_dataset(Path(__file__).absolute().parents[0] /'test_data/SASCOF/cfsv2_hcst.nc', decode_times=False, chunks='auto')
	cola_hcst = xr.open_dataset(Path(__file__).absolute().parents[0] /'test_data/SASCOF/cola_hcst.nc', decode_times=False, chunks='auto')
	nasa_hcst = xr.open_dataset(Path(__file__).absolute().parents[0] /'test_data/SASCOF/nasa_hcst.nc', decode_times=False, chunks='auto')
	X = xr.concat([cans_hcst, cfsv2_hcst, cola_hcst, nasa_hcst], 'M').assign_coords({'M':[0,1,2,3]}).mean('L').prec

	# Observation File
	Y = xr.open_dataset(Path(__file__).absolute().parents[0] /'test_data/SASCOF/observed_rainfall.nc', decode_times=False, chunks='auto').expand_dims({'M':[0]}).rf / (30.33)
	lats0p25x0p25 = np.linspace(np.min(Y.coords['LATITUDE'].values), np.max(Y.coords['LATITUDE'].values), int((np.max(Y.coords['LATITUDE'].values)- np.min(Y.coords['LATITUDE'].values) ) // 0.25 + 1) )
	lons0p25x0p25 = np.linspace(np.min(Y.coords['LONGITUDE'].values), np.max(Y.coords['LONGITUDE'].values), int((np.max(Y.coords['LONGITUDE'].values)- np.min(Y.coords['LONGITUDE'].values) ) // 0.25 + 1) )
	regridded0p25x0p25 = regrid(Y, lons0p25x0p25, lats0p25x0p25, x_lat_dim='LATITUDE', x_lon_dim='LONGITUDE', x_sample_dim='time', x_feature_dim='M').isel(time=-1, M=0)

def test_gaussian_blur():
	Y = xr.open_dataset(Path(__file__).absolute().parents[0] /'test_data/SASCOF/observed_rainfall.nc', decode_times=False, chunks='auto').expand_dims({'M':[0]}).rf / (30.33)
	blurred = gaussian_smooth(Y, x_lat_dim='LATITUDE', x_lon_dim='LONGITUDE', x_sample_dim='time', x_feature_dim='M', kernel=(3,3)).isel(time=-1, M=0)


def test_minmax():
	Y = xr.open_dataset(Path(__file__).absolute().parents[0] /'test_data/SASCOF/observed_rainfall.nc', decode_times=False, chunks='auto').expand_dims({'M':[0]}).rf / (30.33)
	# demonstrate MinMax Scaling
	minmax = MinMax()
	minmax.fit(Y, x_lat_dim='LATITUDE', x_lon_dim='LONGITUDE', x_sample_dim='time', x_feature_dim='M')
	mmscaled = minmax.transform(Y, x_lat_dim='LATITUDE', x_lon_dim='LONGITUDE', x_sample_dim='time', x_feature_dim='M').isel(M=0, time=-1)


def test_normal_scaler():
	Y = xr.open_dataset('test_data/SASCOF/observed_rainfall.nc', decode_times=False, chunks='auto').expand_dims({'M':[0]}).rf / (30.33)
	normal = Normal()
	normal.fit(Y, x_lat_dim='LATITUDE', x_lon_dim='LONGITUDE', x_sample_dim='time', x_feature_dim='M')
	nscaled = normal.transform(Y, x_lat_dim='LATITUDE', x_lon_dim='LONGITUDE', x_sample_dim='time', x_feature_dim='M').isel(M=0, time=-1)


def test_model_pca():
	cans_hcst = xr.open_dataset(Path(__file__).absolute().parents[0] /'test_data/SASCOF/cansipsv2_hcst.nc', decode_times=False, chunks='auto')
	cfsv2_hcst = xr.open_dataset(Path(__file__).absolute().parents[0] /'test_data/SASCOF/cfsv2_hcst.nc', decode_times=False, chunks='auto')
	cola_hcst = xr.open_dataset(Path(__file__).absolute().parents[0] /'test_data/SASCOF/cola_hcst.nc', decode_times=False, chunks='auto')
	nasa_hcst = xr.open_dataset(Path(__file__).absolute().parents[0] /'test_data/SASCOF/nasa_hcst.nc', decode_times=False, chunks='auto')
	X = xr.concat([cans_hcst, cfsv2_hcst, cola_hcst, nasa_hcst], 'M').assign_coords({'M':[0,1,2,3]}).mean('L').prec

	PCA = PrincipalComponentsAnalysis(n_components=3)
	PCA.fit(X, x_lat_dim='Y', x_lon_dim='X', x_feature_dim='M', x_sample_dim='S')
	transformed = PCA.transform(X, x_lat_dim='Y', x_lon_dim='X', x_feature_dim='M', x_sample_dim='S')

def test_spatial_pca():
	Y = xr.open_dataset(Path(__file__).absolute().parents[0] /'test_data/SASCOF/observed_rainfall.nc', decode_times=False, chunks='auto').expand_dims({'M':[0]}).rf / (30.33)
	SPCA = SpatialPCA()
	SPCA.fit(Y, x_lat_dim='LATITUDE', x_lon_dim='LONGITUDE', x_feature_dim='M', x_sample_dim='time')
	eofs = SPCA.eofs()
