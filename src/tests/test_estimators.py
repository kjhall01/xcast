import pytest
from .. import * 
import xarray as xr
from pathlib import Path 

def make_regtestdata():
	# Forecast Files
	cans_fcst = xr.open_dataset(Path(__file__).absolute().parents[0] / 'test_data/SASCOF/cansipsv2_fcst.nc', decode_times=False, chunks='auto')
	cfsv2_fcst = xr.open_dataset(Path(__file__).absolute().parents[0] /'test_data/SASCOF/cfsv2_fcst.nc', decode_times=False, chunks='auto')
	cola_fcst = xr.open_dataset(Path(__file__).absolute().parents[0] /'test_data/SASCOF/cola_fcst.nc', decode_times=False, chunks='auto')
	nasa_fcst = xr.open_dataset(Path(__file__).absolute().parents[0] /'test_data/SASCOF/nasa_fcst.nc', decode_times=False, chunks='auto')
	F = xr.concat([cans_fcst, cfsv2_fcst, cola_fcst, nasa_fcst], 'M').assign_coords({'M':[0,1,2,3]}).mean('L').prec
	# Hindcast Files
	cans_hcst = xr.open_dataset(Path(__file__).absolute().parents[0] /'test_data/SASCOF/cansipsv2_hcst.nc', decode_times=False, chunks='auto')
	cfsv2_hcst = xr.open_dataset(Path(__file__).absolute().parents[0] /'test_data/SASCOF/cfsv2_hcst.nc', decode_times=False, chunks='auto')
	cola_hcst = xr.open_dataset(Path(__file__).absolute().parents[0] /'test_data/SASCOF/cola_hcst.nc', decode_times=False, chunks='auto')
	nasa_hcst = xr.open_dataset(Path(__file__).absolute().parents[0] /'test_data/SASCOF/nasa_hcst.nc', decode_times=False, chunks='auto')
	X = xr.concat([cans_hcst, cfsv2_hcst, cola_hcst, nasa_hcst], 'M').assign_coords({'M':[0,1,2,3]}).mean('L').prec
	# Observation File
	Y = xr.open_dataset(Path(__file__).absolute().parents[0] /'test_data/SASCOF/observed_rainfall.nc', decode_times=False, chunks='auto').expand_dims({'M':[0]}).rf / (30.33)
	return X, Y

def make_clftestdata():
	# Forecast Files
	cans_fcst = xr.open_dataset(Path(__file__).absolute().parents[0] /'test_data/SASCOF/cansipsv2_fcst.nc', decode_times=False, chunks='auto')
	cfsv2_fcst = xr.open_dataset(Path(__file__).absolute().parents[0] /'test_data/SASCOF/cfsv2_fcst.nc', decode_times=False, chunks='auto')
	cola_fcst = xr.open_dataset(Path(__file__).absolute().parents[0] /'test_data/SASCOF/cola_fcst.nc', decode_times=False, chunks='auto')
	nasa_fcst = xr.open_dataset(Path(__file__).absolute().parents[0] /'test_data/SASCOF/nasa_fcst.nc', decode_times=False, chunks='auto')
	F = xr.concat([cans_fcst, cfsv2_fcst, cola_fcst, nasa_fcst], 'M').assign_coords({'M':[0,1,2,3]}).mean('L').prec
	# Hindcast Files
	cans_hcst = xr.open_dataset(Path(__file__).absolute().parents[0] /'test_data/SASCOF/cansipsv2_hcst.nc', decode_times=False, chunks='auto')
	cfsv2_hcst = xr.open_dataset(Path(__file__).absolute().parents[0] /'test_data/SASCOF/cfsv2_hcst.nc', decode_times=False, chunks='auto')
	cola_hcst = xr.open_dataset(Path(__file__).absolute().parents[0] /'test_data/SASCOF/cola_hcst.nc', decode_times=False, chunks='auto')
	nasa_hcst = xr.open_dataset(Path(__file__).absolute().parents[0] /'test_data/SASCOF/nasa_hcst.nc', decode_times=False, chunks='auto')
	X = xr.concat([cans_hcst, cfsv2_hcst, cola_hcst, nasa_hcst], 'M').assign_coords({'M':[0,1,2,3]}).mean('L').prec
	# Observation File
	Y = xr.open_dataset(Path(__file__).absolute().parents[0] /'test_data/SASCOF/observed_rainfall.nc', decode_times=False, chunks='auto').expand_dims({'M':[0]}).rf / (30.33)

	oh = RankedTerciles()
	oh.fit(Y, x_sample_dim='time', x_lat_dim='LATITUDE', x_lon_dim='LONGITUDE')
	Y = oh.transform(Y, x_sample_dim='time', x_lat_dim='LATITUDE', x_lon_dim='LONGITUDE')
	return X, Y

@pytest.mark.parametrize("reg,x,y", [(reg, *make_regtestdata()) for reg in all_regressors])
def test_regressors(reg,x,y):
	if reg in [rPoissonRegression, rGammaRegression]:
		reg = reg()
		x = x.where(x > 0, other=0.00000001)
		y = y.where(y > 0, other=0.00000001)
	else:
		reg = reg()
	reg.fit(x, y, x_sample_dim='S', y_sample_dim='time', y_lat_dim='LATITUDE', y_lon_dim='LONGITUDE')
	preds = reg.predict(x, x_sample_dim='S')

@pytest.mark.parametrize("clf,x,y", [(clf, *make_clftestdata()) for clf in arbitrary_classifiers])
def test_classifiers(clf,x,y):
	clf  = clf()
	clf.fit(x, y, x_sample_dim='S', y_sample_dim='time', y_lat_dim='LATITUDE', y_lon_dim='LONGITUDE', y_feature_dim='C')
	preds = clf.predict(x, x_sample_dim='S' )

@pytest.mark.parametrize("clf,x,y", [(clf, *make_clftestdata()) for clf in tercile_classifiers])
def test_tercile_classifiers(clf, x, y):
	clf  = clf()
	clf.fit(x, y, x_sample_dim='S', y_sample_dim='time', y_lat_dim='LATITUDE', y_lon_dim='LONGITUDE', y_feature_dim='C')
	preds = clf.predict(x, x_sample_dim='S' )
