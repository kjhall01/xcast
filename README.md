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

  <h3 align="center">XCast: A Gridpoint-Wise Statistical Modeling Library for the Earth Sciences </h3>
  
  XCast is a free and open source (passion) project designed to help Earth Scientists scale single-point-in-space regression approaches to spatial gridded data using the popular Earth Science data tool, Xarray. XCast provides a set of tools useful for manipulating and preprocessing Xarray datasets, and implements a 
"fit-predict" training and prediction framework similar to those of the traditional Python statistical tools. More than just a "double-for-loop" wrapper for machine learning libraries, XCast is designed to be high-performance, intuitive, and easily extensible. It is our hope that XCast will serve to bridge the gap between the two-dimensional world of Python Data Science (Samples x Features), and the four-dimensional world of climate data (Samples x Features x Latitude x Longitude).
    <br />
    <a href="https://github.com/kjhall01/xcast/blob/main/XCAST_DOCS.md"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/kjhall01/xcast/blob/main/XCastDeterministic.ipynb">View Demo</a>
    ·
    <a href="https://github.com/kjhall01/xcast/issues">Report Bug</a>
    ·
    <a href="https://github.com/kjhall01/xcast/issues">Request Feature</a>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li><a href="#why-xcast">Why XCast?</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
    </li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>


<!-- Why XCast -->
## Why XCast?

Numerous problems in the Earth Sciences are solved by finding statistical relationships (multiple-regression type) between variables measured at a given point in space, across time. Often, it's desirable to apply these approaches at many points in space, on a 'gridpoint-wise' basis. While Python has numerous [statistical](http://www.scipy.org/) and [machine learning](http://scikit-learn.org/stable/) libraries, none are designed to accomodate fitting more than one statistical model at once, i.e., at many points in space, as is required by this gridpoint-wise approach. 

XCast enables users to apply Python's various statistical tools to spatial gridded data on a gridpoint-wise basis, without having to manually track and manage different dimensions, lists of model instances, or metadata. Built on [Xarray](http://xarray.pydata.org/en/stable/) and [Dask](https://dask.org/), two powerful data science libraries, XCast is capable of analyzing "Big-Data" that won't fit in RAM, and can be scaled to supercomputer clusters. It is designed to be extended to accomodate new statistical libraries easily, and to maximize synergy with the PanGEO stack and other Earth Science data analytics packages like XClim, ClimPred, and XSkillScore. 

<!-- GETTING STARTED -->
## Getting Started

1. Install with [Anaconda](https://anaconda.org/)
   ```sh
   conda install -c hallkjc01 xcast
   ```
2. Read the [Documentation](https://github.com/kjhall01/xcast/)
3. Check out our [blog](blogwebsite.org)


<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/kjhall01/xcast/issues) for a list of proposed features (and known issues).


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- CONTACT -->
## Contact
Email: kjhall@iri.columbia.edu (This is a side project, so it may take a while to get back to you)

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* [SciKit-Learn](https://scikit-learn.org/stable/)
* [HPELM](https://hpelm.readthedocs.io/en/latest/)
* This README template comes from [here](https://github.com/othneildrew/Best-README-Template/edit/master/BLANK_README.md) - thank you!

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
