## Statement of Need 

In climate science and climate forecasting, some approaches use spatially-dependent statistical techniques, i.e., spatial principal components regression. These types of models save time and computational effort by fitting over a region in space all at once. Other approaches use gridpoint-wise models, where one model is fit at each point in space. 

The second type of approach is difficult to implement in Python. While there are open source implementations of numerous statistical tools and machine learning approaches available in Python, they largely work with two dimensional data of “N” samples and “M” features. They are not built to work with four-dimensional, spatial data, or to synergize with mainstream climate data formats, like the Xarray DataArray (NetCDF). Climate data, even if in the Xarray format, may also implement different coordinate and variable naming standards and conventions, or be too large for in-memory computation

XCast solves all of these problems at once. It manages the creation of the numerous model instances necessary to implement gridpoint-wise statistical methods, and coordinates the extraction of the appropriate region of the dataset used by each instance. XCast allows the programmer to store their data as four-dimensional Xarray data types while building gridpoint-wise statistical and machine learning models, which facilitates parallel, distributed, and out-of-core big-data computation with Dask. XCast accommodates arbitrary coordinate and variable naming by allowing the user to specify the names of the sample, feature, latitude and longitude dimensions of their datasets. By allowing programmers to stay within the Xarray ecosystem, XCast opens big-data climate science and climate forecasting to a world of synergy with tools like OPEnDAP, Intake, ClimPred, XClim, and the entire PanGEO stack.  


## Overview 

XCast is a free and open source Python package for big-data climate analytics, machine learning, and forecasting. It is designed to facilitate the application of two-dimensional Python statistical and machine learning models to four-dimensional Xarray data on a gridpoint-wise basis for multi-model ensemble deterministic and probabilistic forecasting. 

Standard Python statistics and machine learning implementations are incompatible with the mainstream climate data format, the Xarray DataArray, and require large amounts of data wrangling to be used for climate forecasting. XCast allows climate data scientists to use these traditional two-dimensional data science tools directly with four-dimensional Xarray-formatted climate data, which allows them in turn to use the powerful features of the Xarray Library. 

XCast builds on standard Python data science tools to implement data preprocessing techniques like dimensionality reduction, filling missing data, regridding, interpolation, and one-hot encoding, as well as model training and leave-n-out cross validation. XCast currently implements twelve (12) deterministic forecasting methodologies like Ensemble Mean, Multiple Linear Regression, Artificial Neural Networks, and Extreme Learning Machine, and ten (10) probabilistic forecasting methodologies including Member Counting, Extended Logistic Regression, Probabilistic Artificial Neural Network, and Probabilistic Extreme Learning Machine. It is, however, designed specifically to extend to new methods easily. 






