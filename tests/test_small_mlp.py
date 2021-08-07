import src as xc
import xarray as xr
import numpy as np

bd = xc.open_CsvDataset('cyclones.csv', tlabels=True)
y = bd.isel(M=0).expand_dims({'M':[0]})
x = bd.isel(M=slice(1,None))

xval1 = xc.cross_validate(xc.ProbabilisticELM, x.climate_var, y.climate_var,  window=1, verbose=True, ND=3, an_thresh=4, bn_thresh=3, explicit=True)
xval2 = xc.cross_validate(xc.MultiExtendedLogisticRegression, x.climate_var, y.climate_var,  window=1, verbose=True, ND=3, an_thresh=4, bn_thresh=3, explicit=True)
xval3 = xc.cross_validate(xc.ExtremeLearningMachine, x.climate_var, y.climate_var,  window=1, verbose=True, ND=3)

vals = np.hstack( [ xval1.hindcasts.isel(X=0, Y=0).values, xval2.hindcasts.isel(X=0, Y=0).values, xval3.hindcasts.isel(X=0, Y=0).values ] )
np.savetxt('cyclone_results.csv',vals, delimiter=',' )
