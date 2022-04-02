import xarray as xr 
from .. import *

def NMME_IMD_ISMR():
	X = xr.open_dataset(Path(__file__).absolute().parents[0] /'test_data/NMME-INDIA-JJAS-PR.nc', chunks='auto').prec
	Y = xr.open_dataset(Path(__file__).absolute().parents[0] /'test_data/IMD-PR-JJAS-1982-2018.nc',  chunks='auto').rf
	oh = RankedTerciles()
	oh.fit(Y)
	T = oh.transform(Y)
	return X, Y, T
