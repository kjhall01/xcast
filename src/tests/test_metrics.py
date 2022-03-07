import pytest
from .. import  * 
import xarray as xr
import numpy as np

@pytest.mark.parametrize("mtr", flat_classification_metrics)
def test_flat_classifier_metrics(mtr):
	pred = np.asarray([[0.45, 0.35, 0.20], [.33,.33, .33], [.4, .33, .27], [.15, .3, .55], [.2, .4, .4]])
	test = np.asarray([[0,1,0], [0,0,1], [1,0,0], [0,0,1], [0,1,0]])
	res = mtr(pred, test)


@pytest.mark.parametrize("mtr", flat_regression_metrics)
def test_flat_regressor_metrics(mtr):
	pred = np.arange(10).reshape(-1,1)
	test = np.arange(10).reshape(-1,1) + 11.0
	res = mtr(pred, test)

def make_clftestdata():
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

	oh = RankedTerciles()
	oh.fit(X.isel(M=0).expand_dims('M'), x_sample_dim='S', )
	X = oh.transform(X.isel(M=0).expand_dims('M'), x_sample_dim='S')
	return X, Y

@pytest.mark.parametrize('mtr', generalized_probabilistic_metrics)
def test_generalized_probabilistic_metrics(mtr):
	X, Y = make_clftestdata()
	mtr(X, Y, x_sample_dim='S', x_feature_dim='C', y_sample_dim='time', y_lat_dim='LATITUDE', y_lon_dim='LONGITUDE', y_feature_dim='C')

@pytest.mark.parametrize('mtr', categorical_probabilistic_metrics)
def test_categorical_probabilistic_metrics(mtr):
	X, Y = make_clftestdata()
	mtr(X, Y, x_sample_dim='S', x_feature_dim='C', y_sample_dim='time', y_lat_dim='LATITUDE', y_lon_dim='LONGITUDE', y_feature_dim='C')

def make_regtestdata():
	cans_hcst = xr.open_dataset(Path(__file__).absolute().parents[0] /'test_data/SASCOF/cansipsv2_hcst.nc', decode_times=False, chunks='auto')
	cfsv2_hcst = xr.open_dataset(Path(__file__).absolute().parents[0] /'test_data/SASCOF/cfsv2_hcst.nc', decode_times=False, chunks='auto')
	cola_hcst = xr.open_dataset(Path(__file__).absolute().parents[0] /'test_data/SASCOF/cola_hcst.nc', decode_times=False, chunks='auto')
	nasa_hcst = xr.open_dataset(Path(__file__).absolute().parents[0] /'test_data/SASCOF/nasa_hcst.nc', decode_times=False, chunks='auto')
	X = xr.concat([cans_hcst, cfsv2_hcst, cola_hcst, nasa_hcst], 'M').assign_coords({'M':[0,1,2,3]}).mean('L').prec.isel(M=0).expand_dims('M')
	# Observation File
	Y = xr.open_dataset(Path(__file__).absolute().parents[0] /'test_data/SASCOF/observed_rainfall.nc', decode_times=False, chunks='auto').expand_dims({'M':[0]}).rf / (30.33)
	return X, Y

@pytest.mark.parametrize('mtr', single_output_deterministic_metrics)
def test_single_output_deterministic_metrics(mtr):
	X, Y = make_regtestdata()
	mtr(X, Y, x_sample_dim='S', y_sample_dim='time', y_lat_dim='LATITUDE', y_lon_dim='LONGITUDE')

@pytest.mark.parametrize('mtr', multiple_output_deterministic_metrics)
def test_multioutput_deterministic_metrics(mtr):
	X, Y = make_regtestdata()
	mtr(X, Y, x_sample_dim='S', y_sample_dim='time', y_lat_dim='LATITUDE', y_lon_dim='LONGITUDE')
