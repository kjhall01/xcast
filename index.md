## Welcome to XCast

XCast is a High-Performance Data Science toolkit for the Earth Sciences. It allows one to perform gridpoint-wise statistical and machine learning analyses in an efficient way using [Dask Parallelism](https://dask.org/), through an API that closely mirrors that of [SciKit-Learn](https://scikit-learn.org/stable/), with the exception that XCast produces and consumes Xarray DataArrays, rather than two-dimensional NumPy arrays. 

Our goal is to lower the barriers to entry to Earth Science (and, specifically, climate forecasting) by bridging the gap between Python's Gridded Data utilities (Xarray, NetCDF4, etc) and its Data Science utilities (Scikit-Learn, Scipy, OpenCV), which are normally incompatible. Through XCast, you can use all your favorite estimators, skill metrics, etc with NetCDF, Grib2, Zarr, and other types of gridded data. 

XCast also lets you scale your gridpoint-wise earth science machine learning approaches to institutional supercomputers and computer clusters with ease. Its compatibility with Dask-Distributed's client schedulers make scalability a non-issue. 

### Installing XCast

XCast is distributed on Anaconda.org, and is installable with the following command: 

```
conda install -c conda-forge -c hallkjc01 xcast
```

However, we have found that it is more reliable / better practice to install into a new conda environment: 

```
conda create -c conda-forge -c hallkjc01 -n xcast_env xcast jupyter ipykernel cartopy netcdf4
conda activate xcast_env
```
This command gives you some extras- jupyter & ipykernel for working with jupyter notebooks, as many earth scientists do, cartopy for plotting gridded data on maps, and netcdf4 as an xarray backend (Xarray is a dependency of XCast). 

If you want to then add ```xcast_env``` as a jupyter kernel, use the following (and if you get 'permission denied' - use sudo): 

```
python -m ipykernel install --name=xcast_env
```

Then you'll be able to select ```xcast_env``` as a kernel in Jupyter Notebook, and be able to import xcast. 

XCast is currently unavailable on windows because my windows machine is too old to build the package - anyone got one I can borrow? (kidding- I'm working on CI) 



