import src as xc
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import datetime as dt

# Forecast Files
cans_fcst = xr.open_dataset('test_data/SASCOF/cansipsv2_fcst.nc', decode_times=False, chunks='auto')
cfsv2_fcst = xr.open_dataset('test_data/SASCOF/cfsv2_fcst.nc', decode_times=False, chunks='auto')
cola_fcst = xr.open_dataset('test_data/SASCOF/cola_fcst.nc', decode_times=False, chunks='auto')
nasa_fcst = xr.open_dataset('test_data/SASCOF/nasa_fcst.nc', decode_times=False, chunks='auto')
F = xr.concat([cans_fcst, cfsv2_fcst, cola_fcst, nasa_fcst], 'M').assign_coords({'M':[0,1,2,3]}).mean('L').prec

# Hindcast Files
cans_hcst = xr.open_dataset('test_data/SASCOF/cansipsv2_hcst.nc', decode_times=False, chunks='auto')
cfsv2_hcst = xr.open_dataset('test_data/SASCOF/cfsv2_hcst.nc', decode_times=False, chunks='auto')
cola_hcst = xr.open_dataset('test_data/SASCOF/cola_hcst.nc', decode_times=False, chunks='auto')
nasa_hcst = xr.open_dataset('test_data/SASCOF/nasa_hcst.nc', decode_times=False, chunks='auto')
X = xr.concat([cans_hcst, cfsv2_hcst, cola_hcst, nasa_hcst], 'M').assign_coords({'M':[0,1,2,3]}).mean('L').prec

# Observation File
Y = xr.open_dataset('test_data/SASCOF/observed_rainfall.nc', decode_times=False, chunks='auto').expand_dims({'M':[0]}).rf


#print('{} STARTING REGRESSOR TESTS'.format(dt.datetime.now()))
#for modeltype in xc.deterministic_mmes:
	#try:
#	print('{}    ATTEMPTING TO MAKE {} FORECAST'.format(dt.datetime.now(), modeltype.__name__))
#	model = modeltype(ND=1)
#	model.fit(X, Y, x_sample_dim='S', y_sample_dim='time', y_lat_dim='LATITUDE', y_lon_dim='LONGITUDE', lat_chunks=10, lon_chunks=10, parallel_in_memory=False, verbose=True)
#	preds = model.predict(X, x_sample_dim='S', lat_chunks=10, lon_chunks=10,  verbose=True)
	#except:
	#	print('\n{} FAILED TO MAKE {} FORECAST'.format(dt.datetime.now(), modeltype.__name__))

print('\n{} STARTING CLASSIFIER TESTS'.format(dt.datetime.now()))
for modeltype in xc.probabilistic_mmes:
	#try:
	print('{}    ATTEMPTING TO MAKE {} FORECAST'.format(dt.datetime.now(), modeltype.__name__))
	model = modeltype(ND=1)
	model.fit(X, Y, x_sample_dim='S', y_sample_dim='time', y_lat_dim='LATITUDE', y_lon_dim='LONGITUDE', lat_chunks=10, lon_chunks=10, parallel_in_memory=False, verbose=True)
	preds = model.predict(X, x_sample_dim='S', lat_chunks=10, lon_chunks=10,  verbose=True)
	print(preds)
	print('\n')
	#except:
	#	print('\n{} FAILED TO MAKE {} FORECAST'.format(dt.datetime.now(), modeltype.__name__))
