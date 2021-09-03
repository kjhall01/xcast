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
    <li><a href="#utilities">Utilities</a></li>
    <li><a href="#data-preprocessing">Data Preprocessing</a></li>
    <li><a href="#training-models">Training Models</a></li>
    <li><a href="#validation">Validation</a></li>
    <li><a href="#verification">Verification With XSkillScore</a></li>
    <li><a href="#extending-basemme">Extending BaseMME</a></li>
  </ol>
</details>


<!-- background -->
## Background

As described in the [README](https://github.com/kjhall01/xcast/edit/main/README.md), XCast is designed to facilitate the application of standard Python statistical and machine learning tools to spatial gridded data on a "gridpoint-wise" basis-- meaning one point at a time, independently. While XCast is much more than some "for-loops", (implementing chunked, out-of-memory, and parallel computation with dask for example) it can be conceptualized as such.

Imagine that you designed a normal statistical workflow, regressing thirty years of rainfall totals on thirty years of sea surface temperature and geopotential height measurements, averaged over a spatial region. Now imagine you wanted to do that process, except instead of averaging over the spatial region, you wanted to do it independently at each point in space- that's what XCast does! It abstracts out the fact that you're really working with NLATITUDE x NLONGITUDE statistical models, and lets you treat them all as one datatype. 

That being said, the difficulty lies in the fact that standard Python statistical tools are not built to work with the gridded data format that we're hoping to use for our analysis. Our data is now, instead of a two-dimensional, N_Features (sst & gph) x N_Samples (30 years) layout, in a four-dimensional  N_Features x N_Samples x N_Latitudes x N_Longitudes layout. Standard Python statistical tools are designed to work with the first layout, as two-dimensional NumPy Arrays; but in the Earth Sciences, gridded data most commonly comes as high-dimensional Xarray DataArrays, which are incompatible. It is possible to convert between the two, but managing dimensionality can be confusing, especially for novice python programmers. Requiring all of the data to be in-memory at once also dramatically limits the size of the datasets one can work with. Both problems are barriers to entry to earth science.

### XCast Dimensionality
While Earth Science Data often comes as high-dimensional datatypes, XCast works exclusively with 4D data. Perhaps at some point in the future we will add support for higher dimensionality, but we had to draw the lines somewhere. Those four dimensions each serve very specific purposes: 

1. Spatial Dimension One: The outer dimension over which operations will loop (ex: Latitude) 
2. Spatial Dimension Two: The inner dimension over which operations will loop (ex: Longitude) 
3. Feature Dimension: The dimension that represents the multiple features of a single sample  (ex: independent predictors, SST and GPH in the above example scenario) 
4. Sample Dimension: The dimension that represents the multiple samples on which a statistical method will be applied (ex: Years) 

In most XCast functions and class methods, the names of each of the above dimensions on the input data must be specified. This may seem like a pain, but it also allows XCast to accomodate any possible given standard or convention. 

<!-- utilities -->
## Utilities

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
