import src as xc 
import xarray as xr 
import matplotlib.pyplot as plt 

X = xr.open_dataset('NMME-INDIA-JJAS-PR.nc').prec
Y = xr.open_dataset('IMD-PR-JJAS-1982-2018.nc').rf 

ohc = xc.RankedTerciles()
ohc.fit(Y)
T = ohc.transform(Y)

mlr = xc.cPOELM(lat_chunks=4, lon_chunks=4)
mlr.fit(X, T, rechunk=True)
p = mlr.predict(X, rechunk=True)

