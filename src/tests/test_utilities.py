import pytest
from .. import * 
import xarray as xr
import numpy as np
from pathlib import Path 

@pytest.mark.utility
def test_regrid_coarse():
	X = xr.open_dataset(Path(__file__).absolute().parents[0] /'test_data/NMME-INDIA-JJAS-PR.nc', decode_times=False, chunks='auto').prec
	Y = xr.open_dataset(Path(__file__).absolute().parents[0] /'test_data/IMD-PR-JJAS-1982-2018.nc', decode_times=False, chunks='auto').rf

	lats2x2 = Y.coords['Y'].values[::2]
	lons2x2 = Y.coords['X'].values[::2]
	regridded2x2 = regrid(Y, lons2x2, lats2x2)

@pytest.mark.utility
def test_regrid_fine():
	X = xr.open_dataset(Path(__file__).absolute().parents[0] /'test_data/NMME-INDIA-JJAS-PR.nc', decode_times=False, chunks='auto').prec
	Y = xr.open_dataset(Path(__file__).absolute().parents[0] /'test_data/IMD-PR-JJAS-1982-2018.nc', decode_times=False, chunks='auto').rf

	lats0p25x0p25 = np.linspace(np.min(Y.coords['Y'].values), np.max(Y.coords['Y'].values), int((np.max(Y.coords['Y'].values)- np.min(Y.coords['Y'].values) ) // 0.25 + 1) )
	lons0p25x0p25 = np.linspace(np.min(Y.coords['X'].values), np.max(Y.coords['X'].values), int((np.max(Y.coords['X'].values)- np.min(Y.coords['X'].values) ) // 0.25 + 1) )
	regridded0p25x0p25 = regrid(Y, lons0p25x0p25, lats0p25x0p25)

@pytest.mark.utility
def test_gaussian_blur():
	Y = xr.open_dataset(Path(__file__).absolute().parents[0] /'test_data/IMD-PR-JJAS-1982-2018.nc', decode_times=False, chunks='auto').rf
	blurred = gaussian_smooth(Y, kernel=(3,3))

@pytest.mark.utility
def test_minmax():
	Y = xr.open_dataset(Path(__file__).absolute().parents[0] /'test_data/IMD-PR-JJAS-1982-2018.nc', decode_times=False, chunks='auto').rf
	# demonstrate MinMax Scaling
	minmax = MinMax()
	minmax.fit(Y)
	mmscaled = minmax.transform(Y)

@pytest.mark.utility
def test_normal_scaler():
	Y = xr.open_dataset(Path(__file__).absolute().parents[0] /'test_data/IMD-PR-JJAS-1982-2018.nc', decode_times=False, chunks='auto').rf
	normal = Normal()
	normal.fit(Y)
	nscaled = normal.transform(Y)

@pytest.mark.utility
def test_model_pca():
	X = xr.open_dataset(Path(__file__).absolute().parents[0] /'test_data/NMME-INDIA-JJAS-PR.nc', decode_times=False, chunks='auto').prec

	PCA = PrincipalComponentsAnalysis(n_components=3)
	PCA.fit(X)
	transformed = PCA.transform(X)
	
@pytest.mark.utility
def test_spatial_pca():
	Y = xr.open_dataset(Path(__file__).absolute().parents[0] /'test_data/IMD-PR-JJAS-1982-2018.nc', decode_times=False, chunks='auto').rf
	SPCA = SpatialPCA()
	SPCA.fit(Y)
	eofs = SPCA.eofs()
