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

XCast is a powerful, open-source library for building gridpoint-wise statistical and machine learning models and manipulating gridded geospatial data. It leverages the same high-performance computing library as Xarray’s core functionality, and brings gridpoint-wise parallelism and out-of-core computation to the Earth Science community. XCast allows the user to apply gridpoint-wise machine learning methods to Xarray-formatted data without losing access to Xarray’s basic functions, or losing its synergy with the rest of the Python data science ecosystem. 



# Statement of need

Many Earth Science research problems are approached by regressing one geospatial variable on others, at a given gridpoint. Climate forecasters, for example, often regress observation data on General Circulation Model (GCM) output to construct multi-model ensembles (MMEs), correct GCM bias, and produce operational forecasts. [@Acharya:2021], [@Kirtman:2014]  In scientific contexts involving gridded geospatial data, like climate forecasting, it can be very useful to produce gridded results and maps by applying statistical operations on a gridpoint-wise basis: once at every point, independently.  

In Python, most geospatial data is manipulated with Xarray, an open-source library for working with labelled multi-dimensional data. [@hoyer2017xarray] Xarray is convenient and easy to use, and most of its basic operations are powered by Dask, a high-performance computing library. [@dask] Python’s statistical and machine learning libraries, however, are designed to work with two-dimensional, unlabelled data, and are incompatible with Xarray. [@scikit-learn], [@Harris:2020] In order to use Python statistical and machine learning libraries with geospatial data, the user must first extract it from Xarray. Unfortunately this results in the loss of the high-performance Xarray operations implemented with Dask. Parallelism and out-of-core computation on gridded datasets are still possible, but require a prohibitively large amount of effort to implement.

Since gridpoint-wise statistical operations can benefit so much from parallelism, this is a huge loss. Luckily, XCast provides a convenient, flexible solution to the problems faced by the geospatial data science community. By allowing programmers to stay within the Xarray ecosystem, XCast opens big-data climate science and climate forecasting to a world of synergy with tools like OPEnDAP, Intake, ClimPred, XClim, and the entire PanGEO stack.  


# High Performance Computing

XCast implements two types of high performance computation- local out-of-core processing, and in-memory parallelism. It lets the user split their dataset into a number of chunks along each spatial dimension, such that each chunk fits in memory. In the case of out-of-core processing, operations are applied to each chunk serially, so as not to load more than one chunk into memory at once.  With in-memory parallelism, operations are applied to multiple chunks at once. The number of chunks that can be processed in parallel depends on the available resources, which can be controlled easily with Dask’s distributed computing client library. Dask’s distributed client module also allows XCasts in-memory parallel model training to be scaled to supercomputer clusters, to make big-data computation possible. However, when XCast’s gridpoint-wise statistical and machine learning models make predictions, there is no reasonable way to use in-memory parallel computing, since data returned through Dask must all fit in memory at once. Predictions therefore always use out-of-core computation, to avoid oversubscribing resources. 

# Statistical Methods 
XCast implements numerous gridpoint-wise statistical models. Each prepackaged model fills missing data and regrids the predictors to the spatial resolution of the predictands. The currently implemented models are listed below. 

* **Ensemble Mean** EM averages over the predictor dataset's feature dimension. 
* **Bias-Corrected Ensemble Mean** BCEM averages over the predictor dataset's feature dimension after standardizing each feature independently. It then inverts the standardization of predictions using the mean and standard deviation of the predictands. 
* **Multiple Linear Regression** MLR fits a Multiple Linear Regression between the predictand dataset and the predictor dataset. 
* **Poisson Regression** PR fits a Poisson Regression between the predictand dataset and the predictor dataset. 
* **Gamma Regression** GR fits a Gamma Regression between the predictand dataset and the predictor dataset. 
* **Principal Components Regression** PCR fits a Multiple Linear Regression between the principal components of the predictor dataset features and the predictand datset. 
* **Multi-Layer Perceptron** MLP fits an Artificial Neural Network between the predictand dataset and the predictor dataset using backpropagation. 
* **Random Forest** RF fits numerous Decision Tree models between the predictand dataset and the predictor dataset.
* **Ridge Regression**  RR fits a Ridge Regression between the predictand dataset and the predictor dataset. 
* **Extreme Learning Machine** ELM fits an Extreme Learning Machine model between the predictand dataset and the predictor dataset. [@Huang:2004]
* **Principal Components ELM** PCELM  fits an Extreme Learning Machine model between the principal components of the predictor dataset features and the predictand datset. [@Jinkwon:2007]

# Extending XCast 

Since XCast cannot possibly implement every possible statistical and machine learning approach, it is designed to be easily extensible to new types of models. All a user needs to do is implement a new XCast BaseMME subclass, override its constructor method, and set its `model_type` attribute to the target machine learning model class. The machine learning model class must implement `.fit` and `.predict` methods that work with unlabeled, two-dimensional arrays. More detail is available in the XCast documentation. 


# Acknowledgements

Thank you to the developers of all of the Python geospatial data science libraries without whom this project would be impossible.

# References 
