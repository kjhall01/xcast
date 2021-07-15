import xarray as xr
from pathlib import Path
from .data_validation import *

def open_XrDataset(filename):
	"""Opens a NetCDF file with Xarray, tries to decode times then tries without"""
	assert Path(filename).absolute().is_file(), 'Cant find {}'.format(Path(filename).absolute())
	try:
		ds = xr.open_dataset(str(Path(filename).absolute()), chunks='auto')
	except:
		ds = xr.open_dataset(str(Path(filename).absolute()), decode_times=False, chunks='auto')
	return ds


def open_CsvDataset(filename, delimiter=',', M='M', T='T', tlabels=False, varnames=False, parameter='climate_var'):
	assert Path(filename).absolute().is_file(), 'Cant find {}'.format(Path(filename).absolute())

	with open(str(Path(filename).absolute()), 'r') as f:
		content = f.read()
	content = [line.strip().split(delimiter) for line in content.split('\n') if len(line) > 0 ]

	# check for variable names
	for i in range(len(content[0])):
		if i != 0:
			try:
				x = float(content[0][i])
			except:
				varnames = True

	# check for time labels
	for i in range(len(content)):
		if i != 0:
			try:
				x = float(content[i][0])
			except:
				tlabels = True

	# extract  variable names
	if varnames:
		varnames = content.pop(0)
		if not tlabels:
			varnames = [ str(i.strip()) for i in varnames ]
		else:
			varnames = [str(varnames[i].strip()) for i in range(1, len(varnames))]
	else:
		varnames = [i+1 for i in range(max([len(content[j]) for j in range(len(content))]))]

	# extract time labels
	if tlabels:
		tlabels = []
		for i in range(len(content)):
			tlabels.append(content[i].pop(0))
	else:
		tlabels = [i+1 for i in range(len(content))]

	# check shape of array - pad to len(varnames) with np.nan, and then cut off extra
	for i in range(len(content)):
		while len(content[i]) < len(varnames):
			content[i].append(np.nan)
		content[i] = content[i][:len(varnames)]
		content[i] = [ float(content[i][j]) for j in range(len(content[i]))]

	content = np.asarray(content)
	content = content.reshape((len(tlabels), len(varnames), 1 , 1 ))
	coords = {M: varnames, T: tlabels, 'X':[0], 'Y': [0]}
	data_vars = {parameter: ([T, M, 'X', 'Y'], content)}
	return xr.Dataset(data_vars, coords=coords)
