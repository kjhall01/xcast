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
![installs](https://img.shields.io/conda/dn/hallkjc01/xcast?color=light-green&label=Installations&style=for-the-badge)
[![DOI](https://zenodo.org/badge/386326352.svg)](https://zenodo.org/badge/latestdoi/386326352)



<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/kjhall01/xcast/">
    <img src="images/XCastLogo.png" alt="Logo" width="200" height="200">
  </a>

  <h3 align="center">XCast: A Climate Forecasting Toolkit  </h3>
  <h3 align="center">**working on xcast v2 -- keep an eye out** </h3>

  XCast is a free and open source climate forecasting toolkit written by Kyle Hall & Nachiketa Acharya, designed to help forecasters and earth scientists apply state-of-the-art postprocessing techniques to gridded data sets. 
    <br />
    <a href="https://xcast-lib.github.io/"><strong>Explore the docs»</strong></a>
    <br />
    <a href="https://github.com/kjhall01/xcast/issues">Report Bug</a>
  </p>
</p>


<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li><a href="#why-xcast">Why XCast?</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>


## Installation 
XCast is distributed on Anaconda , and can be installed like any other Python library  with the following command:

```
conda install -c conda-forge -c hallkjc01 xcast 
```

to set up an XCast environment for use with Jupyter notebook, please use the following commands: 

```
conda create -n xcast_env -c conda-forge -c hallkjc01 xcast xarray netcdf4 jupyter ipykernel 
conda activate xcast_env 
python -m ipykernel install --name=xcast_env --user 
```
you'll then be able to select `xcast_env` from the list of available jupyter kernels in your jupyter notebook

<!-- LICENSE -->
## License
Distributed under the MIT License. See `LICENSE` for more information.

<!-- CONTACT -->
## Contact
Please make an issue [here](https://github.com/kjhall01/xcast/issues). 

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
[license-url]: https://github.com/kjhall01/xcast/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/kjhall01
