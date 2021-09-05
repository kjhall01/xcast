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

  <h3 align="center">PyMME: Gridpoint-Wise Multi-Model Ensemble Forecasting with XCast </h3>
  
  PyMME is an effort to use non-traditional statistical methods to construct gridpoint-wise Multi-Model Ensemble forecasts (MMEs) and examine their relative skill. MME construction is a common way to improve the skill of forecasts made by General Circulation Models (GCMs). Traditionally a simple ensemble mean is used, but there is potential for fitted statistical models to improve MME skill.  Although the scope of XCast is not limited to MME construction, PyMME is implemented with XCast and serves as XCast's primary research objective.  
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
    <li><a href="#why-pymme">Why PyMME?</a></li>
    <li><a href="#timeline">Timeline</a></li>
    <li><a href="#contact">Research Goals</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>


<!-- Why PyMME -->
## Why PyMME?

In the age of the climate crisis, seasonal and subseasonal climate variability have become more and more severe. Dry seasons have become drier, wet seasons wetter, and all, harder to predict. Supporting agricultural industries through climate services and information is more important than ever. Exploration of new ways to produce multi-model ensemble forecasts is a critical component of the research that supports those programs. 

<!-- GETTING STARTED -->
## Timeline
* Spring 2019 - PyELM-MME Ideation & Design Work begins
* September 2020 - Development work begins on [PyELM-MME](https://github.com/kjhall01/PyELM-MME), a predecessor to the PyMME project and XCast 
* Fall/Winter 2020 - Development on [PyELM-MME](https://github.com/kjhall01/PyELM-MME) 
* March 2021 - [PyELM-MME](https://github.com/kjhall01/PyELM-MME) presented at the [UCAR SEA's Improving Scientific Software Conference](https://sea.ucar.edu/conference/2021), XCast & PyMME Ideation 
* Summer 2021 - Development of [XCast](https://github.com/kjhall01/xcast/) begins
* August 2021 - Publication of PyELM-MME paper in the [PROCEEDINGS OF THE 2021 IMPROVING SCIENTIFIC SOFTWARE CONFERENCE](https://opensky.ucar.edu/islandora/object/technotes:589)
* September 2021 - [PyMME](https://github.com/kjhall01/xcast/blob/main/PYMME.md) presented at the [3rd Annual NOAA Workshop on Leveraging AI in the Environmental Sciences](https://2021noaaaiworkshop.sched.com/event/lSAN/virtual-poster-walk-part-vii) 


<!-- CONTRIBUTING -->
## Research Goals 
The primary goal of the PyMME project is to explore the use of new and interesting statistical methods for the construction of multi-model ensemble forecasts. A secondary goal aims to develop a "general purpose" gridpoint-wise, xarray-based python statistical toolkit named "XCast" to serve as a partner library to ClimPred, XClim, and XSkillScore, and expand the Python Earth Science Machine Learning ecosystem. A tertiary goal is to identify specific combinations of GCM datasets, spatial regions, and statistical/ai methods that exhibit high predictive skill in seasonal or subseasonal forecasts. 


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- CONTACT -->
## Contact
Email: kjhall@iri.columbia.edu (This is a side project, so it may take a while to get back to you)

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* add acknowledgements - pyelm-mme, xcast, xarray, pangeo, dask, iri data library, datasets 

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
