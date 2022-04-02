import pytest
from .. import * 
import xarray as xr
from pathlib import Path 

def make_test_data():
	X = xr.open_dataset(Path(__file__).absolute().parents[0] /'test_data/NMME-INDIA-JJAS-PR.nc', chunks='auto').prec
	Y = xr.open_dataset(Path(__file__).absolute().parents[0] /'test_data/IMD-PR-JJAS-1982-2018.nc',  chunks='auto').rf
	oh = RankedTerciles()
	oh.fit(Y)
	T = oh.transform(Y)
	return X, Y, T

@pytest.mark.parametrize("reg,x,y,t", [(reg, *make_test_data()) for reg in regressors])
def test_regressors(reg,x,y,t):
	if reg in [rPoissonRegression, rGammaRegression]:
		reg = reg()
		x = x.where(x > 0, other=0.00000001)
		y = y.where(y > 0, other=0.00000001)
	else:
		reg = reg()
	print(x, y)
	reg.fit(x, y)
	preds = reg.predict(x)

@pytest.mark.parametrize("clf,x,y,t", [(clf, *make_test_data()) for clf in classifiers])
def test_classifiers(clf,x,y,t):
	clf  = clf()
	clf.fit(x, t)
	preds = clf.predict_proba(x)

