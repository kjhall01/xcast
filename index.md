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

**THIS PAGE HAS BEEN MOVED TO XCAST'S NEW HOME, [XCAST-LIB.GITHUB.IO!](https://xcast-lib.github.io)**

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
