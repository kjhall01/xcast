<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5627478.svg)](https://doi.org/10.5281/zenodo.5627478)




<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/kjhall01/xcast/">
    <h2 align="center"><img src="XCastLogo.png" align="center" alt="Logo" width="50" height="50">  XCast</h2>
  </a>
    <a href="https://github.com/kjhall01/xcast/issues">Report Bug</a>
    ·
    <a href="https://github.com/kjhall01/xcast/issues">Request Feature</a>
  </p>
</p>




## Welcome to XCast

XCast is a High-Performance Data Science toolkit for the Earth Sciences. It allows one to perform gridpoint-wise statistical and machine learning analyses in an efficient way using [Dask Parallelism](https://dask.org/), through an API that closely mirrors that of [SciKit-Learn](https://scikit-learn.org/stable/), with the exception that XCast produces and consumes Xarray DataArrays, rather than two-dimensional NumPy arrays. 

Our goal is to lower the barriers to entry to Earth Science (and, specifically, climate forecasting) by bridging the gap between Python's Gridded Data utilities (Xarray, NetCDF4, etc) and its Data Science utilities (Scikit-Learn, Scipy, OpenCV), which are normally incompatible. Through XCast, you can use all your favorite estimators, skill metrics, etc with NetCDF, Grib2, Zarr, and other types of gridded data. 

XCast also lets you scale your gridpoint-wise earth science machine learning approaches to institutional supercomputers and computer clusters with ease. Its compatibility with Dask-Distributed's client schedulers make scalability a non-issue. 


<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h3 style="display: inline-block">Table of Contents</h3></summary>
  <ol>
    <li><a href="#installing-xcast">Installation</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
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
