import pytest
from .. import  * 
import xarray as xr
import numpy as np

@pytest.mark.skill
@pytest.mark.parametrize("mtr", flat_classification_metrics)
def test_flat_classifier_metrics(mtr):
	pred = np.asarray([[0.45, 0.35, 0.20], [.33,.33, .33], [.4, .33, .27], [.15, .3, .55], [.2, .4, .4]])
	test = np.asarray([[0,1,0], [0,0,1], [1,0,0], [0,0,1], [0,1,0]])
	res = mtr(pred, test)

@pytest.mark.skill
@pytest.mark.parametrize("mtr", flat_regression_metrics)
def test_flat_regressor_metrics(mtr):
	pred = np.arange(10).reshape(-1,1)
	test = np.arange(10).reshape(-1,1) + 11.0
	res = mtr(pred, test)


def make_test_data():
	X = xr.open_dataset(Path(__file__).absolute().parents[0] /'test_data/NMME-INDIA-JJAS-PR.nc', decode_times=False, chunks='auto').prec
	Y = xr.open_dataset(Path(__file__).absolute().parents[0] /'test_data/IMD-PR-JJAS-1982-2018.nc', decode_times=False, chunks='auto').rf
	oh = RankedTerciles()
	oh.fit(Y)
	T = oh.transform(Y)

	oh = RankedTerciles()
	oh.fit(X.isel(M=0).expand_dims({'M':[0]}))
	T2 = oh.transform(X.isel(M=0).expand_dims({'M':[0]}) )
	return X, Y, T, T2

@pytest.mark.skill
@pytest.mark.parametrize('mtr', generalized_probabilistic_metrics)
def test_generalized_probabilistic_metrics(mtr):
	X, Y, T, T2 = make_test_data()
	mtr(T, T2)

@pytest.mark.skill
@pytest.mark.parametrize('mtr', categorical_probabilistic_metrics)
def test_categorical_probabilistic_metrics(mtr):
	X, Y, T, T2 = make_test_data()
	mtr(T,T2)

@pytest.mark.skill
@pytest.mark.parametrize('mtr', single_output_deterministic_metrics)
def test_single_output_deterministic_metrics(mtr):
	X, Y, T, T2 = make_test_data()
	mtr(Y, Y)

@pytest.mark.skill
@pytest.mark.parametrize('mtr', multiple_output_deterministic_metrics)
def test_multioutput_deterministic_metrics(mtr):
	X, Y, T, T2= make_test_data()
	mtr(Y, Y)
