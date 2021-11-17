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
  <h3 align="center">XCast: Documentation (Warning: Out of Date as of 11/17/21)</h3>
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
        <li><a href="#regrid">regrid</a></li>
        <li><a href="#gaussian-smoothing">gaussian_smooth</a></li>
      </ol>
    </li>
    <li>
      <a href="#data-preprocessing">Data Preprocessing</a>
      <ol>
        <li><a href="#dimensionality-reduction">Dimensionality Reduction</a></li>
        <li><a href="#scaling">Scaling</a></li>
      </ol>
    </li>
    <li>
      <a href="#statistical-models">Model Training</a>
      <ol>
        <li>
          <a href="#deterministic-models">Deterministic Models </a>
          <ol>
            <li><a href="#ensemble-mean">Ensemble Mean</a></li>
            <li><a href="#bias-corrected-ensemble-mean">Bias-Corrected Ensemble Mean</a></li>
            <li><a href="#multiple-linear-regression">Multiple Linear Regression</a></li>
            <li><a href="#principal-components-regression">Principal Components Regression</a></li>
            <li><a href="#poisson-regression">Poisson Regression</a></li>
            <li><a href="#gamma-regression">Gamma Regression</a></li>
            <li><a href="#ridge-regression">Ridge Regression</a></li>
            <li><a href="#random-forest">Random Forest</a></li>
            <li><a href="#multi-layer-perceptron">Multi-Layer Perceptron</a></li>
            <li><a href="#extreme-learning-machine">Extreme Learning Machine</a></li>
            <li><a href="#extreme-learning-machine-with-pca">ELM with PCA</a></li>
         </ol>
       </li> 
       <li><a href="#probabilistic-models">Probabilistic Models </a></li> 
     </ol> 
    </li>
    <li><a href="#validation">Validation</a></li>
    <li><a href="#verification">Verification With XSkillScore</a></li>
    <li><a href="#extending-basemme">Extending BaseMME</a></li>
  </ol>
</details>


<!-- background -->
## Background

For tutorials and help with Xarray, please visit [their webpage!](http://xarray.pydata.org/en/stable/)

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


### XCast Format Requirements
While XCast works directly with Xarray DataArrays, there are some extra requirements. 
1. Each dimension must have cooresponding coordinates that match the dimension exactly in name and size 
2. There can be no extra dimensions of size 1 outside of the 4 XCast dimensions, and no un-used coordinates 
3. All four XCast dimensions must be present, if only of size 1. 


<!-- utilities -->
## Utilities

XCast implements a few utilities to assist in the preparation of datasets for use in statistical modeling and machine learning: 

#### Open CSV Dataset
Opens a .csv file formatted as N_SAMPLES rows of N_FEATURES columns, i.e., Each row is a sample and each column is a features. 

    ds = xc.open_CsvDataset(filename, delimiter=',', M='M', T='T', tlabels=False, varnames=False, parameter='climate_var')
     
1. filename - string, representing path to desired file
2. delimiter - string, representing delimiter between cells of the csv file
3. M - string, representing desired name of returned Xarray Dataset's Feature dimension
4. T - string, representing desired name of returned Xarray Dataset's Sample dimension
5. tlabels - bool, whether or not the .csv file contains labels for each sample in the first column (i.e., years) 
6. varnames - bool, whether or not the .csv file contains labels for each feature in the first row (i.e., SST, GPH, etc) 
7. parameter - string, desired name of Xarray DataArray contained within returned Xarray DataSet 

Returns: Xarray DataSet with DataArray named as parameter 

#### Fill Constant
Fills all of the values of X that are equal to missing_value with val

    X = xc.fill_constant(X, val, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', missing_value=np.nan)

1. X - Xarray DataArray, satisfying XCast dimensionality and format requirements
2. val - numeric type (float, int), desired fill-value
3. x_lat_dim - string, name of latitude dimension on X 
4. x_lon_dim - string, name of longitude dimension on X 
5. x_sample_dim - string, name of sample dimension on X
6. x_feature_dim - string, name of feature dimension on X 
7. missing_value - numeric type, value to be replaced

Returns: Xarray DataArray with missing_values filled with val 

#### Fill Time Mean
Fills all of the values of X that are equal to missing_value with the mean of X along the sample dimension at for each lat/lon/feature

    X = xc.fill_time_mean(X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', missing_value=np.nan)

1. X - Xarray DataArray, satisfying XCast dimensionality and format requirements
2. x_lat_dim - string, name of latitude dimension on X 
3. x_lon_dim - string, name of longitude dimension on X 
4. x_sample_dim - string, name of sample dimension on X
5. x_feature_dim - string, name of feature dimension on X 
6. missing_value - numeric type, value to be replaced

Returns: Xarray DataArray with missing_values filled

#### Fill Space Mean
Fills all of the values of X that are equal to missing_value with the mean of X along the spatial dimensions for that sample/feature

    X = xc.fill_space_mean(X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', missing_value=np.nan)

1. X - Xarray DataArray, satisfying XCast dimensionality and format requirements
2. x_lat_dim - string, name of latitude dimension on X 
3. x_lon_dim - string, name of longitude dimension on X 
4. x_sample_dim - string, name of sample dimension on X
5. x_feature_dim - string, name of feature dimension on X 
6. missing_value - numeric type, value to be replaced

Returns: Xarray DataArray with missing_values filled


#### Regrid 
"Regrids" X onto new latitude and longitude coordinates, presumably of a different resolution using SciPy interp2d (bivariate spline interpolation) 

    X = xc.regrid(X, lons, lats, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', use_dask=False, feat_chunks=1, samp_chunks=1 )

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

#### Gaussian Smoothing
Applies gaussian smoothing over a kernel of size 'kernel' in the LATxLON dimensions. 

    X = xc.gaussian_smooth(X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', kernel=(9,9), use_dask=False, feature_chunks=1, sample_chunks=1 )

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
XCast implements a number of desirable, gridpoint-wise data preprocessing tools based on the standard Python data science tools. 

### Dimensionality Reduction 
#### Principal Components Analysis
Applies sklearn.decomposition.PCA across the feature dimension of an XCast dataset. 

    pca = xc.PrincipalComponentsAnalysis(n_components=2, use_dask=False)
    pca.fit(X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', lat_chunks=1, lon_chunks=1, verbose=False )
    X = pca.transform(X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', lat_chunks=1, lon_chunks=1 , verbose=False) 
    
1. n_components - int, number of principal components to keep after dimensionality reduction
2. use_dask - bool, whether or not to perform operations on a 'chunked' basis, out-of-memory, using Dask.
3. X - Xarray DataArray, satisfying XCast dimensionality and format requirements
4. x_lat_dim - string, name of latitude dimension on X 
5. x_lon_dim - string, name of longitude dimension on X 
6. x_sample_dim - string, name of sample dimension on X
7. x_feature_dim - string, name of feature dimension on X 
8. lat_chunks - int, desired number of chunks across the latitude dimension to use for out-of-memory computation with dask
9. lon_chunks - int, desired number of chunks across the longitude dimension to use for out-of-memory computation with dask
10. verbose - bool, whether or not to print progressbar representing operations


#### NMF
Applies sklearn.decomposition.NMF across the feature dimension of an XCast dataset. 

    nmf = xc.NMF(n_components=2, use_dask=False)
    nmf.fit(X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', lat_chunks=1, lon_chunks=1, verbose=False )
    X = nmf.transform(X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', lat_chunks=1, lon_chunks=1 , verbose=False) 
    
1. n_components - int, number of components to keep after dimensionality reduction
2. use_dask - bool, whether or not to perform operations on a 'chunked' basis, out-of-memory, using Dask.
3. X - Xarray DataArray, satisfying XCast dimensionality and format requirements
4. x_lat_dim - string, name of latitude dimension on X 
5. x_lon_dim - string, name of longitude dimension on X 
6. x_sample_dim - string, name of sample dimension on X
7. x_feature_dim - string, name of feature dimension on X 
8. lat_chunks - int, desired number of chunks across the latitude dimension to use for out-of-memory computation with dask
9. lon_chunks - int, desired number of chunks across the longitude dimension to use for out-of-memory computation with dask
10. verbose - bool, whether or not to print progressbar representing operations

#### Factor Analysis
Applies sklearn.decomposition.FactorAnalysis across the feature dimension of an XCast dataset. 

    fa = xc.FactorAnalysis(n_components=2, use_dask=False)
    fa.fit(X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', lat_chunks=1, lon_chunks=1, verbose=False )
    X = fa.transform(X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', lat_chunks=1, lon_chunks=1 , verbose=False) 
    
1. n_components - int, number of components to keep after dimensionality reduction
2. use_dask - bool, whether or not to perform operations on a 'chunked' basis, out-of-memory, using Dask.
3. X - Xarray DataArray, satisfying XCast dimensionality and format requirements
4. x_lat_dim - string, name of latitude dimension on X 
5. x_lon_dim - string, name of longitude dimension on X 
6. x_sample_dim - string, name of sample dimension on X
7. x_feature_dim - string, name of feature dimension on X 
8. lat_chunks - int, desired number of chunks across the latitude dimension to use for out-of-memory computation with dask
9. lon_chunks - int, desired number of chunks across the longitude dimension to use for out-of-memory computation with dask
10. verbose - bool, whether or not to print progressbar representing operations


#### Dictionary Learning
Applies sklearn.decomposition.DictionaryLearning across the feature dimension of an XCast dataset. 

    dl = xc.DictionaryLearning(n_components=2, use_dask=False)
    dl.fit(X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', lat_chunks=1, lon_chunks=1, verbose=False )
    X = dl.transform(X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', lat_chunks=1, lon_chunks=1 , verbose=False) 
    
1. n_components - int, number of components to keep after dimensionality reduction
2. use_dask - bool, whether or not to perform operations on a 'chunked' basis, out-of-memory, using Dask.
3. X - Xarray DataArray, satisfying XCast dimensionality and format requirements
4. x_lat_dim - string, name of latitude dimension on X 
5. x_lon_dim - string, name of longitude dimension on X 
6. x_sample_dim - string, name of sample dimension on X
7. x_feature_dim - string, name of feature dimension on X 
8. lat_chunks - int, desired number of chunks across the latitude dimension to use for out-of-memory computation with dask
9. lon_chunks - int, desired number of chunks across the longitude dimension to use for out-of-memory computation with dask
10. verbose - bool, whether or not to print progressbar representing operations

### Scaling 
#### Normal Scaling 
Scales a dataset to mean=0, std. dev=1 at each gridpoint by subtracting the time-stddev and dividing by the time-mean for each feature.

    n = xc.Normal()
    n.fit(X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M') 
    X_Norm = n.transform(X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M') 
    n.inverse_transform(X_Norm, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M') == X # Returns True 
    
1. X - Xarray DataArray, satisfying XCast dimensionality and format requirements
2. x_lat_dim - string, name of latitude dimension on X 
3. x_lon_dim - string, name of longitude dimension on X 
4. x_sample_dim - string, name of sample dimension on X
5. x_feature_dim - string, name of feature dimension on X 

#### MinMax Scaling 
Scales a dataset to \[min, max\] at each gridpoint by subtracting the minimum over time, dividing by the range over time, then multiplying by (max-min) and adding min for each feature. 

    mn = xc.MinMax(min=-1, max=1)
    mn.fit(X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M') 
    X_mn = mn.transform(X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M') 
    mn.inverse_transform(X_Norm, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M') == X # Returns True 
    
1. min - numeric type, represents minimum of desired scaled range
2. max - numeric type, represents maximum of desired scaled range 
3. X - Xarray DataArray, satisfying XCast dimensionality and format requirements
4. x_lat_dim - string, name of latitude dimension on X 
5. x_lon_dim - string, name of longitude dimension on X 
6. x_sample_dim - string, name of sample dimension on X
7. x_feature_dim - string, name of feature dimension on X 

<!-- Training Models -->
## Statistical Models 
### Deterministic Models 
See [XCastDeterministic.ipynb](https://github.com/kjhall01/xcast/blob/main/XCastDeterministic.ipynb) for examples of using the XCast models. 
All XCast deterministic models currently use fill_space_mean to deal with missing data, and regrid X onto Y's coordinates in order to make sure they're the same size and shape. (Size and shape must be the same in order to use gridpoint-wise statistical and machine learning methods.) 
Also, all the same transformations applied to predictors before model fitting are applied to predictors for prediction as well, using the same, saved, scaling objects. That way the operation stays 'in-sample', ie, the model doesn't get extra information about the test set from the scaling. 

#### Ensemble Mean
Produces a deterministic estimate by taking the mean over the values in the feature dimension at each sample, latitude, longitude. 

#### Bias Corrected Ensemble Mean 
Standardizes input features using xc.Normal() on each independently, then proceeds to produce an estimate by taking the mean at each sample, latitude, and longitude, then xc.Normal.inverse_transform's that estimate onto the distribution of the predictand data. 

#### Multiple Linear Regression 
Scales input features to \[-1, 1\] using xc.MinMax, each independently, then fits an sklearn.linear_model.LinearRegression between the transformed inputs and raw predictands. 

#### Poisson Regression 
Scales input features to \[-1, 1\] using xc.MinMax, each independently, then fits an sklearn.linear_model.PoissonRegressor between the transformed inputs and raw predictands. 

#### Gamma Regression 
Scales input features to \[0.00000001, 2\] using xc.MinMax, each independently, then fits an sklearn.linear_model.GammaRegressor between the transformed inputs and raw predictands. 

#### Principal Components Regression
Scales input features to \[-1, 1\] using xc.MinMax, each independently, then applies PCA to the predictors, then fits an sklearn.linear_model.LinearRegression between the transformed inputs and raw predictands. 

#### Multi Layer Perceptron
\* Warning Slow \* Scales input features to \[-1, 1\] using xc.MinMax, each independently, and scales the predictands to 'normal', then fits an sklearn.neural_network.MLPRegressor between the transformed inputs and transformed predictands. Finally, scales predictions back to distribution of raw predictands. 

#### Random Forest 
\* Warning: Slow \* Scales input features to \[-1, 1\] using xc.MinMax, each independently, and scales the predictands to 'normal', then fits an sklearn.ensemble.RandomForestRegressor between the transformed inputs and transformed predictands. Finally, scales predictions back to distribution of raw predictands. 

#### Ridge Regression 
Scales input features to \[-1, 1\] using xc.MinMax, each independently, then fits an sklearn.linear_model.Ridge between the transformed inputs and raw predictands. 

#### Extreme Learning Machine 
Scales input features to \[-1, 1\] using xc.MinMax, each independently, and scales the predictands to 'normal', then fits an hpelm.ELM() between the transformed inputs and transformed predictands. Finally, scales predictions back to distribution of raw predictands. 

#### Extreme Learning Machine with PCA
Scales input features to \[-1, 1\] using xc.MinMax, each independently, then applies PCA to the predictors, and scales the predictands to 'normal', then fits an hpelm.ELM() between the transformed inputs and transformed predictands. Finally, scales predictions back to distribution of raw predictands. 


### Probabilistic Models 
In the future, XCast will implement probabilistic statistical models as well :)

<!-- Validation -->
## Validation
XCast implements leave-n-out cross-validation in order to allow you to accurately and easily evaluate the skill of the models you build. 

    xval_hindcasts = xc.cross_validate( xc.MultipleLinearRegression, X, Y, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', y_lat_dim='Y', y_lon_dim='X', y_sample_dim='T', y_feature_dim='M',  window=3, verbose=0, ND=1, **kwargs )
    
1. MME - XCast MME Type, one of the deterministic models 
2. X - Predictors, Xarray DataArray, satisfying XCast dimensionality and format requirements
3. Y - Predictands, Xarray DataArray, satisfying XCast dimensionality and format requirements
4. x_lat_dim - string, name of latitude dimension on X 
5. x_lon_dim - string, name of longitude dimension on X 
6. x_sample_dim - string, name of sample dimension on X
7. x_feature_dim - string, name of feature dimension on X 
8. y_lat_dim - string, name of latitude dimension on Y
9. y_lon_dim - string, name of longitude dimension on Y 
10. y_sample_dim - string, name of sample dimension on Y
11. y_feature_dim - string, name of feature dimension on Y
12. window - int, size of cross validation window (N of leave-N-out) 
13. verbose - int, how much printing do you want to happen? 
14. ND - int, number of times to train model, for non-deterministic statistical models 

Returns: Xarray Dataset containing ND-means and standard deviations for the cross-validated hindcasts


<!-- Verification -->
## Verification With XSkillScore
Verification can be done with XSkillScore. the xc.to_xss function will help you transfer naming conventions. 


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
