---
title: 'XCast: A Gridpoint-Wise Statistical Modeling Python Library for the Earth Sciences'
tags:
  - Python
  - Machine learning 
  - Artificial Intelligence
  - Xarray 
  - Spatial Data
  - Gridded Data 
  - Forecasting 
authors:
  - name: Kyle Hall
    orcid: 0000-0003-3723-0662
    affiliation: 1
  - name: Nachiketa Acharya
    orcid: 0000-0003-3010-2158
    affiliation: 2
affiliations:
 - name: International Research Institute for Climate & Society, Columbia University
   index: 1
 - name: Center for Earth System Modeling, Analysis, & Data (ESMAD), Department of Meteorology and Atmospheric Science, The Pennsylvania State University
   index: 2
date: 17 September 2021
bibliography: paper.bibtex
---

# Summary

XCast is a free and open source Python package for climate data analytics, machine learning, and forecasting. It is designed to facilitate the application of two-dimensional Python statistical and machine learning models to four-dimensional Xarray gridded data on a gridpoint-wise basis for the purpose of climate forecasting, but 

Standard Python statistics and machine learning implementations are incompatible with the mainstream climate data format, the Xarray DataArray, and require large amounts of data wrangling to be used for climate forecasting. XCast allows climate data scientists to use these traditional two-dimensional data science tools directly with four-dimensional Xarray-formatted climate data, which allows them in turn to use the powerful features of the Xarray Library. 

XCast builds on standard Python data science tools to implement data preprocessing techniques like dimensionality reduction, filling missing data, regridding, interpolation, and one-hot encoding, as well as model training and leave-n-out cross validation. XCast currently implements twelve (12) deterministic forecasting methodologies like Ensemble Mean, Multiple Linear Regression, Artificial Neural Networks, and Extreme Learning Machine. It is designed specifically to be extended to new methods easily. 


# Statement of need

In climate science and climate forecasting, some approaches use spatially-dependent statistical techniques, i.e., spatial principal components regression. These types of models save time and computational effort by fitting over a region in space all at once. Other approaches use gridpoint-wise models, where one model is fit at each point in space. 

The second type of approach is difficult to implement in Python. While there are open source implementations of numerous statistical tools and machine learning approaches available in Python, they largely work with two dimensional data of “N” samples and “M” features. They are not built to work with four-dimensional, spatial data, or to synergize with mainstream climate data formats, like the Xarray DataArray (NetCDF). Climate data, even if in the Xarray format, may also implement different coordinate and variable naming standards and conventions, or be too large for in-memory computation

XCast solves all of these problems at once. It manages the creation of the numerous model instances necessary to implement gridpoint-wise statistical methods, and coordinates the extraction of the appropriate region of the dataset used by each instance. XCast allows the programmer to store their data as four-dimensional Xarray data types while building gridpoint-wise statistical and machine learning models, which facilitates parallel, distributed, and out-of-core big-data computation with Dask. XCast accommodates arbitrary coordinate and variable naming conventions by allowing the user to specify the names of the sample, feature, latitude and longitude dimensions of their datasets. By allowing programmers to stay within the Xarray ecosystem, XCast opens big-data climate science and climate forecasting to a world of synergy with tools like OPEnDAP, Intake, ClimPred, XClim, and the entire PanGEO stack.  


# 


# Acknowledgements

Thank you to the developers of SciKit-Learn, NumPy, SciPy, Xarray, Dask, and the PanGEO stack without whom this project would be impossible.

# References 
