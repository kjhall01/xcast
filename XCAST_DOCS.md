<!--
*** This README comes from here: https://github.com/othneildrew/Best-README-Template/edit/master/BLANK_README.md - thanks ! 
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/kjhall01/xcast/">
    <img src="images/logo.png" alt="Logo" width="200" height="200">
  </a>
  <h3 align="center">XCast: Documentation </h3>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li><a href="#background">Background</a></li>
    <li>
      <a href="#utilities">Utilities</a>
      <ol>
        <li><a href="#open-CSV-Dataset">open_CsvDataset</a></li>
        <li><a href="#fill-constant">fill_constant</a></li>
        <li><a href="#fill-time-mean">fill_time_mean</a></li>
        <li><a href="#fill-space-mean">fill_space_mean</a></li>
        <li><a href="#validation">Validation</a></li>
        <li><a href="#verification">Verification With XSkillScore</a></li>
        <li><a href="#extending-basemme">Extending BaseMME</a></li>
      </ol>
    </li>
    <li><a href="#data-preprocessing">Data Preprocessing</a></li>
    <li><a href="#training-models">Training Models</a></li>
    <li><a href="#validation">Validation</a></li>
    <li><a href="#verification">Verification With XSkillScore</a></li>
    <li><a href="#extending-basemme">Extending BaseMME</a></li>
  </ol>
</details>


<!-- background -->
## Background

As described in the [README](https://github.com/kjhall01/xcast/edit/main/README.md), XCast is designed to facilitate the application of standard Python statistical and machine learning tools to spatial gridded data on a "gridpoint-wise" basis-- meaning one point at a time, independently. While XCast is much more than "just some for-loops", (as it implements chunked, out-of-memory, and parallel computation Dask, as one example) it can be conceptualized as such.

Imagine that you designed a normal statistical workflow, regressing thirty years of rainfall totals on thirty years of sea surface temperature and geopotential height measurements, averaged over a spatial region. Now imagine you wanted to do that process, except instead of averaging over the spatial region, you wanted to do it independently at each point in space- that's what XCast does! It abstracts out the fact that you're really working with many independent statistical models, and lets you treat them all as one unified datatype. 

That being said, the difficulty lies in the fact that standard Python statistical tools are not built to work with the gridded data format that we're hoping to use for our analysis. Gridded Earth Science Data is generally, instead of two-dimensional, N_FEATURES (sst & gph) x N_SAMPLES (30 years) layouts, in  four-dimensional  N_FEATURES x N_SAMPLES x N_LATITUDES x N_LONGITUDES layouts. Standard Python statistical tools are designed to work with the prior, implemented as two-dimensional NumPy Arrays. Gridded Earth Science Data most commonly comes as high-dimensional Xarray DataArrays, which are incompatible with those tools. It is possible to convert between the two, but managing dimensionality can be confusing, especially for novice python programmers. Requiring all of the data to be in-memory at once also dramatically limits the size of the datasets one can work with. Both problems are barriers to entry to earth science.

### XCast Dimensionality
While Earth Science Data often comes as high-dimensional datatypes, XCast works exclusively with 4D data. Perhaps at some point in the future we will add support for higher dimensionality, but we had to draw the lines somewhere. Those four dimensions each serve very specific purposes: 

1. Spatial Dimension One: The outer dimension over which operations will loop (ex: Latitude) 
2. Spatial Dimension Two: The inner dimension over which operations will loop (ex: Longitude) 
3. Feature Dimension: The dimension that represents the multiple features of a single sample  (ex: independent predictors, SST and GPH ) 
4. Sample Dimension: The dimension that represents the multiple samples on which a statistical method will be applied (ex: Years) 

In most XCast functions and class methods, the names of each of the above dimensions on the input data must be specified. This may seem like a pain, but it also allows XCast to accomodate any possible given standard or convention. 

<!-- utilities -->
## Utilities

XCast implements a few utilities to assist in the preparation of datasets for use in statistical modeling and machine learning: 

### Open CSV Dataset
Opens a .csv file formatted as N_SAMPLES rows of N_FEATURES columns, i.e., Each row is a sample and each column is a features. 
```ds = xc.open_CsvDataset(filename, delimiter=',', M='M', T='T', tlabels=False, varnames=False, parameter='climate_var')```
1. filename - string, representing path to desired file
2. delimiter - string, representing delimiter between cells of the csv file
3. M - string, representing desired name of returned Xarray Dataset's Feature dimension
4. T - string, representing desired name of returned Xarray Dataset's Sample dimension
5. tlabels - bool, whether or not the .csv file contains labels for each sample in the first column (i.e., years) 
6. varnames - bool, whether or not the .csv file contains labels for each feature in the first row (i.e., SST, GPH, etc) 
7. parameter - string, desired name of Xarray DataArray contained within returned Xarray DataSet 

Returns: Xarray DataSet with DataArray named as parameter 

### Fill Constant
Fills all of the values of X that are equal to missing_value with val
```X = xc.fill_constant(X, val, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', missing_value=np.nan)```
1. X - Xarray DataArray, satisfying XCast dimensionality and format requirements
2. val - numeric type (float, int), desired fill-value
3. x_lat_dim - string, name of latitude dimension on X 
4. x_lon_dim - string, name of longitude dimension on X 
5. x_sample_dim - string, name of sample dimension on X
6. x_feature_dim - string, name of feature dimension on X 
7. missing_value - numeric type, value to be replaced

Returns: Xarray DataArray with missing_values filled with val 

### Fill Time Mean
Fills all of the values of X that are equal to missing_value with the mean of X along the sample dimension at for each lat/lon/feature
```X = xc.fill_time_mean(X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', missing_value=np.nan)```
1. X - Xarray DataArray, satisfying XCast dimensionality and format requirements
2. x_lat_dim - string, name of latitude dimension on X 
3. x_lon_dim - string, name of longitude dimension on X 
4. x_sample_dim - string, name of sample dimension on X
5. x_feature_dim - string, name of feature dimension on X 
6. missing_value - numeric type, value to be replaced

Returns: Xarray DataArray with missing_values filled

### fill_space_mean(X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', missing_value=np.nan)
Fills all of the values of X that are equal to missing_value with the mean of X along the spatial dimensions for that sample/feature
1. X - Xarray DataArray, satisfying XCast dimensionality and format requirements
2. x_lat_dim - string, name of latitude dimension on X 
3. x_lon_dim - string, name of longitude dimension on X 
4. x_sample_dim - string, name of sample dimension on X
5. x_feature_dim - string, name of feature dimension on X 
6. missing_value - numeric type, value to be replaced

Returns: Xarray DataArray with missing_values filled


### regrid(X, lons, lats, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', use_dask=False, feat_chunks=1, samp_chunks=1 )
"Regrids" X onto new latitude and longitude coordinates, presumably of a different resolution using SciPy interp2d (bivariate spline interpolation) 
1. X - Xarray DataArray, satisfying XCast dimensionality and format requirements
2. lons - array-like, new longitude coordinates to be used by scipy.interp2d
3. lats - array-like, new latitude coordinates to be used by scipy.interp2d
4. x_lat_dim - string, name of latitude dimension on X 
5. x_lon_dim - string, name of longitude dimension on X 
6. x_sample_dim - string, name of sample dimension on X
7. x_feature_dim - string, name of feature dimension on X 
8. use_dask - bool, whether or not to perform operations on a 'chunked' basis, out-of-memory, using Dask. 
9. feat_chunks - int, desired number of chunks across the feature dimension to use for out-of-memory computation with dask. 
10. samp_chunks - int, desired number of chunks across the sample dimension to use for out-of-memory computation with dask. 

Returns: Xarray DataArray with X interpolated onto new spatial dimensions specified by lons and lats. 

### gaussian_smooth(X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', kernel=(9,9), use_dask=False, feature_chunks=1, sample_chunks=1 )
Applies gaussian smoothing over a kernel of size 'kernel' in the LATxLON dimensions. 
1. X - Xarray DataArray, satisfying XCast dimensionality and format requirements
2. x_lat_dim - string, name of latitude dimension on X 
3. x_lon_dim - string, name of longitude dimension on X 
4. x_sample_dim - string, name of sample dimension on X
5. x_feature_dim - string, name of feature dimension on X 
6. kernel - tuple, length-2 tuple indicating size of kernel in latitude and longitude dimensions.
7. use_dask - bool, whether or not to perform operations on a 'chunked' basis, out-of-memory, using Dask. 
8. feat_chunks - int, desired number of chunks across the feature dimension to use for out-of-memory computation with dask. 
9. samp_chunks - int, desired number of chunks across the sample dimension to use for out-of-memory computation with dask. 

Returns: Xarray DataArray of X with gaussian smoothing applied. 


<!-- data preprocessing -->
## Data Preprocessing

<!-- Training Models -->
## Training Models

<!-- Validation -->
## Validation

<!-- Verification -->
## Verification With XSkillScore

<!-- Extending Base Classes -->
## Extending Base Classes


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/kjhall01/xcast.svg?style=for-the-badge
[contributors-url]: https://github.com/kjhall01/xcast/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/kjhall01/xcast.svg?style=for-the-badge
[forks-url]: https://github.com/kjhall01/xcast/network/members
[stars-shield]: https://img.shields.io/github/stars/kjhall01/xcast.svg?style=for-the-badge
[stars-url]: https://github.com/kjhall01/xcast/stargazers
[issues-shield]: https://img.shields.io/github/issues/kjhall01/xcast.svg?style=for-the-badge
[issues-url]: https://github.com/kjhall01/xcast/issues
[license-shield]: https://img.shields.io/github/license/kjhall01/xcast.svg?style=for-the-badge
[license-url]: https://github.com/kjhall01/xcast/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/kjhall01
