<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
[![DOI](https://zenodo.org/badge/386326352.svg)](https://zenodo.org/badge/latestdoi/386326352)




<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/kjhall01/xcast/">
    <h1 align="center"><img src="XCastLogo.png" align="center" alt="Logo" width="60" height="60">  XCAST</h1>
  </a>
  <p align="center" fontsize=6> Kyle Hall & Nachiketa Acharya </p>

</p>


## Welcome to XCast

XCast is a High-Performance Data Science toolkit for the Earth Sciences. It allows one to perform gridpoint-wise statistical and machine learning analyses in an efficient way using [Dask Parallelism](https://dask.org/), through an API that closely mirrors that of [SciKit-Learn](https://scikit-learn.org/stable/), with the exception that XCast produces and consumes Xarray DataArrays, rather than two-dimensional NumPy arrays. 

Our goal is to lower the barriers to entry to Earth Science (and, specifically, climate forecasting) by bridging the gap between Python's Gridded Data utilities (Xarray, NetCDF4, etc) and its Data Science utilities (Scikit-Learn, Scipy, OpenCV), which are normally incompatible. Through XCast, you can use all your favorite estimators, skill metrics, etc with NetCDF, Grib2, Zarr, and other types of gridded data. 

XCast also lets you scale your gridpoint-wise earth science machine learning approaches to institutional supercomputers and computer clusters with ease. Its compatibility with Dask-Distributed's client schedulers make scalability a non-issue. 


<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h3 style="display: inline-block">Table of Contents</h3></summary>
  <ul>
    <li><a href="#installing-xcast">Installation</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#preprocessing">Preprocessing</a></li>
    <li><a href="#model-training">Model Training</a></li>
    <li><a href="#validation-and-skill">Cross Validation & Skill Metrics</a></li>
    <li><a href="#parallelism-in-xcast">Parallelism In XCast</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ul>
</details>


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

### Getting Started 

The first thing to know about XCast is its required input format. We try to give some flexibility, but have to draw the line somewhere- and so all XCast functions and object methods are designed to produce and consume 4-dimensional [Xarray DataArrays](https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html). The four dimensions correspond to X (ie, longitude), Y (ie, latitude), Samples (ie, time), and Features (ie, ensemble members, or different physical variables used as predictors). They must all be present, if only of size one. The coordinates must also be one-to-one with the dimensions (no unlabeled dimensions, no extraneous coordinates - it might work, but you'd be getting lucky). 

Once it has verified the format of your input, XCast will then attempt to detect the name of each dimension. There are usually only so many standard names of these things, and XCast usually does a decent job, but in the off chance you get a 'detection failed' error, you can also just specify the names of each dimension by passing keyword arguments called x_lat_dim, x_lon_dim, x_feature_dim, x_sample_dim (or y_lat_dim, y_lon_dim, y_sample_dim, and y_feature_dim, depending which argument - predictors or predictands). 

You can view some sample data using: 

```
X, Y, T = xc.NMME_IMD_ISMR() # loads XCast sample Data from North American Multi-model Ensemble and India Meteorological Department
```

Next, you can train a model and make predictions! 

```
X, Y, T = xc.NMME_IMD_ISMR()
mlr = xc.rMultipleLinearRegression()
mlr.fit(X, Y) # no need to specify the names since detection succeeds
preds = mlr.predict(X) 
```

### Preprocessing 

Oftentimes, before training a model, it is good/desirable to preprocess data. In fact, sometimes it's required. XCast has a number of utilities for preprocessing, including: 

###### Regridding: 
Use linear, quadratic, cubic, etc interpolation to regrid your data onto some other dataarray's coordinates. Powered by SciPy.interpolate.interp2d and accepts all the same keyword args. 
```
# regrid coarser: 
X, Y, T = xc.NMME_IMD_ISMR()
lats2x2 = Y.coords['Y'].values[::2]
lons2x2 = Y.coords['X'].values[::2]
regridded2x2 = xc.regrid(Y, lons2x2, lats2x2)

# regrid finer: 
lats0p25x0p25 = np.linspace(np.min(Y.coords['Y'].values), np.max(Y.coords['Y'].values), int((np.max(Y.coords['Y'].values)- np.min(Y.coords['Y'].values) ) // 0.25 + 1) )
lons0p25x0p25 = np.linspace(np.min(Y.coords['X'].values), np.max(Y.coords['X'].values), int((np.max(Y.coords['X'].values)- np.min(Y.coords['X'].values) ) // 0.25 + 1) )
regridded0p25x0p25 = xc.regrid(Y, lons0p25x0p25, lats0p25x0p25)
```

###### Gaussian Kernel Smoothing:
Smooth by averaging over a NxN pixel kernel. Powered by OpenCV
```
X, Y, T = xc.NMME_IMD_ISMR()
blurred = xc.gaussian_smooth(Y, kernel=(3,3))
```

###### Rescaling: 
Apply standard anomaly scaling by removing the mean and dividing by std. dev, or MinMax scale to the interval [-1, 1]. 
```
X, Y, T = xc.NMME_IMD_ISMR()

# MinMax Scaling: 
minmax = xc.MinMax()
minmax.fit(Y)
mmscaled = minmax.transform(Y)

# Standard Anomaly Scaling: 
normal = xc.Normal()
normal.fit(Y)
nscaled = normal.transform(Y)
```

###### Decomposition: 
Apply Principal Components Analysis across the feature dimension, or extract spatial loadings across both spatial dimensions.
```
# Principal Components (feature dimension): 
X, Y, T = xc.NMME_IMD_ISMR()
PCA = xc.PrincipalComponentsAnalysis(n_components=3)
PCA.fit(X)
transformed = PCA.transform(X)

# Principal Components (Spatial): 
SPCA = SpatialPCA()
SPCA.fit(Y)
eofs = SPCA.eofs()
```

###### One-Hot Encoding: 
Encode tercile categories for each sample at each point. A new dimension is created in place of the feature dimension, whose new coordinates will be 0,1,2 where 0 = below normal, 1 = near normal, and 2 = above normal. Needed for all classifiers. 
```
X, Y, T = xc.NMME_IMD_ISMR()

# Tercile OHC by ranking: 
ohc = xc.RankedTerciles()
ohc.fit(Y) 
T = ohc.transform(Y) 

# Tercile OHC by gaussian assumption: 
ohc = xc.NormalTerciles():
ohc.fit(Y)
T= ohc.transform(Y) 
``` 

### Model Training
Model training and prediction mostly follow the same API as SciKit-learn- there are `.fit`, `.predict`, and `.predict_proba` methods on each estimator. Be careful which you try to use- some don't have predict, and some don't have predict_proba. Here are examples of each prepackaged estimator: 

They also accept all the same keyword args as their SciKit-Learn counterparts (except for POELM, ELM and ELR - those are XCast implementations) 

```
X, Y, T = xc.NMME_IMD_ISMR()

# Classifiers - T is just Y but one-hot encoded
mvlr = xc.cMultivariateLogisticRegression()
mvlr.fit(X, T) 
preds = mvlr.predict(X) 
probs = mvlr.predict_proba(X) 

elr = xc.cExtendedLogisticRegression()
elr.fit(X, T) 
preds = elr.predict(X) 
probs = elr.predict_proba(X) 

mlpc = xc.cMultiLayerPerceptron()
mlpc.fit(X, T) 
preds = mlpc.predict(X) 
probs = mlpc.predict_proba(X) 

nbc = xc.cNaiveBayes()
nbc.fit(X, T) 
preds = nbc.predict(X) 
probs = nbc.predict_proba(X) 

rfc = xc.cRandomForest()
rfc.fit(X, T) 
preds = rfc.predict(X) 
probs = rfc.predict_proba(X) 

poelm = xc.cPOELM()
poelm.fit(X, T) 
preds = poelm.predict(X) 
probs = poelm.predict_proba(X) 

# Regressors: 
model = xc.EnsembleMean()
model.fit(X, Y)
preds= model.predict(X) 

model = xc.BiasCorrectedEnsembleMean()
model.fit(X, Y)
preds= model.predict(X) 

model = xc.rMultipleLinearRegression()
model.fit(X, Y)
preds= model.predict(X) 

model = xc.rPoissonRegression()
model.fit(X, Y)
preds= model.predict(X) 

model = xc.rGammaRegression()
model.fit(X, Y)
preds= model.predict(X) 

model = xc.rMultiLayerPerceptron()
model.fit(X, Y)
preds= model.predict(X) 

model = xc.rRandomForest()
model.fit(X, Y)
preds= model.predict(X) 

model = xc.rRidgeRegression()
model.fit(X, Y)
preds= model.predict(X) 

model = xc.rExtremeLearningMachine()
model.fit(X, Y)
preds= model.predict(X) 
```

### Validation And Skill 

If you want to meaningfully interpret the skill of a statistical model or neural network at predicting out-of-sample data, it is highly recommended to use Leave-N-Out Cross Validation to reconstruct a 'cross-validated' hindcast dataset. During cross-validation, the training data is split into N windows- each of which includes all of the training data, except one (or more) years reserved for testing. Models are fit, and used to make predictions on the data they left out, and then those predictions are reassembled into one dataset, which can then be meaningfully compared with the target data in a skill function. 


Here is an example using Leave-One-Out Cross-Validation to construct a hindcast dataset, and then using XCast (powered by scikit-learn and scipy) to compute skill maps: 
```
X, Y, T = xc.NMME_IMD_ISMR()
crossvalidation_window=1 # number of samples to be left out - must be odd
ND=10  # ND represents the number of random initializations to use, to counteract non-determinism in the method

poelm_xval = []
for x_train, y_train, x_test, y_test in xc.CrossValidator(X, Y, window=crossvalidation_window):
    ohc = xc.RankedTerciles()
    ohc.fit(y_train)
    ohc_y_train = ohc.transform(y_train)
    
    poelm = xc.cPOELM(ND=ND, hidden_layer_size=hidden_layer_size, activation=activation)
    poelm.fit(x_train, ohc_y_train)
    poelm_preds = poelm.predict_proba(x_test)
    poelm_xval.append(poelm_preds.isel(S=crossvalidation_window // 2))

poelm = xr.concat(poelm_xval, 'S').mean('ND')

groc = xc.GeneralizedROC(poelm, india_ohc)
poelm_rps = xc.RankProbabilityScore(poelm, india_ohc)
climatological_odds = xr.ones_like(poelm) * 0.33 
climo_rps = xc.RankProbabilityScore(climatological_odds, india_ohc)
rpss = 1 - ( poelm_rps / climo_rps)
```

XCast implements a lot of metrics - view the rest of them [here](https://github.com/kjhall01/xcast/blob/main/src/verification/metrics.py) 


### Parallelism In XCast

Under the hood, XCast implements gridpoint operations using Dask, a powerful python multiprocessing / lazy execution library. Dask makes it extremely easy to scale XCast solutions to institutional computing resources and supercomputer clusters - see [Dask's Distributed Library](https://distributed.dask.org/en/stable/client.html) for details on how to hook XCast into a cluster scheduler. 

Parallelism in XCast can be a huge performance win, even just on your laptop- most come with multiple cores. XCast is capable, through dask, of using multiple processes to run gridpoint-wise operations in parallel by distributing 'chunks' of the data and computation to each process. That is done like this: 

```
from dask.distributed import Client 
client = Client(n_workers=4) # this sets up a local cluster with dask 
X, Y, T = xc.NMME_IMD_ISMR()
X, Y = xc.align_chunks(X, Y, 5, 5) # this will split the data into 5 chunks along latitude, and 5 chunks along longitude for a total of 25 (+/- some) 
mlr = xc.rMultipleLinearRegression()
mlr.fit(X, Y, rechunk=False) # the rechunk=False tells Xcast not to rechunk this data again - it does it by default so you don't have to care
mlr.predict(X, rechunk=False) 
```

This has been shown to dramatically improve computation speed / decrease time. 


### Contact: 
corresponding author: Kyle Hall (hallkjc01@gmail.com) - apologies if I take a long time to get back to you - this is a side project of mine

### Acknowledgements: 
Many thanks to all of the following: 
- Nachiketa Acharya for vital advice & guidance, as well as design help, feedback, and being a true friend
- PanGEO, SciKit-Learn, Xarray, Scipy, and the entire Python Open Source community for building all of XCast's dependencies
- the North American Multimodel Ensemble (NMME) and India Meteorological Department, because I used NMME & IMD data as a test cast while developing XCast 
- NCAR UCAR S.E.A for giving us a concrete goal to work toward! 


[contributors-shield]: https://img.shields.io/github/contributors/kjhall01/xcast.svg?style=for-the-badge
[contributors-url]: https://github.com/kjhall01/xcast/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/kjhall01/xcast.svg?style=for-the-badge
[forks-url]: https://github.com/kjhall01/xcast/network/members
[stars-shield]: https://img.shields.io/github/stars/kjhall01/xcast.svg?style=for-the-badge
[stars-url]: https://github.com/kjhall01/xcast/stargazers
[issues-shield]: https://img.shields.io/github/issues/kjhall01/xcast.svg?style=for-the-badge
[issues-url]: https://github.com/kjhall01/xcast/issues
[license-shield]: https://img.shields.io/github/license/kjhall01/xcast.svg?style=for-the-badge
[license-url]: https://github.com/kjhall01/xcast/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/kjhall01
