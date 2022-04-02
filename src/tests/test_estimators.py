import pytest
from .. import * 
import xarray as xr
from pathlib import Path 
from .data import NMME_IMD_ISMR

make_test_data = NMME_IMD_ISMR

@pytest.mark.parametrize("reg,x,y,t", [(reg, *make_test_data()) for reg in regressors])
def test_regressors(reg,x,y,t):
	if reg in [rPoissonRegression, rGammaRegression]:
		reg = reg()
		x = x.where(x > 0, other=0.00000001)
		y = y.where(y > 0, other=0.00000001)
	else:
		reg = reg()
	reg.fit(x, y)
	preds = reg.predict(x)

@pytest.mark.parametrize("clf,x,y,t", [(clf, *make_test_data()) for clf in classifiers])
def test_classifiers(clf,x,y,t):
	clf  = clf()
	clf.fit(x, t)
	preds = clf.predict_proba(x)

