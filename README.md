# Searching for gravitational self-lensing in PSR J1816-4510
Klara Kurucova | Summer Internship at Max Planck Institute for Gravitational Physics (AEI) 2026

## About 
In this project we study the spider pulsar binary system PSRJ1816-4510. There are multiple reasons that motivate the study of this spider pulsar system: the pulsar is significantly brighter than in other systems, it is more massive and we have evidence for gamma-ray eclipsing that indicates a high inclination of the system. All of these factors predict that this system could exhibit gravitational self-lensing, shown as a milli-magnitude increase in the systems light curve during the companion's superior conjunction. **This would be the first ever detection of gravitational self-lensing in a compact pulsar binary.**

## Data acquired
The observations of this spider pulsar were taken by the HiPERCAM instrument on the Gran Telescopio Canarias (GTC) in the ugriz bands. Data acquisition was conducted in multiple observing blocks (currently 24 acquired observing blocks) as the observing plan is to cover:
1. The full orbit of the companion (11 observing blocks)
2. Additional observations of the companion's superior conjuction (so far 11 observing blocks).

This observing strategy has been adopted as we are trying to detect a milli-magnitude feature in the companion's superior conjuction. With a large number of observing blocks we can improve the signal to noise ratio and bring out the desired feature. Please note that some additional observing blocks for the companion's superior conjuction are still expected. Note that here I only provide the output .dat files to be able to run GL_search.py, as uploading all the files would simply take up too much storage and is not necessary to showcase the functioning of this code.

## Data organisation 
There is a seprate folder for each observation run containing the HiPERCAM raw data, reduction files generated through the reduction HCAM pipeline and .dat files for each of the bands. Each folder has a separate README.md as a guide to the files. 

## Magnitude conversion for ensemble photometry 
For ensemble photomtery the Pan-STARRS P1 catalogue was used. To convert this to the HiPERCAM superSDSS, we use the function P1tosuper.py. This function takes an input of a .csv file and converts the magnitude based on linear model cofficients from [JL Tonry et al.](http://dx.doi.org/10.1088/0004-637X/750/2/99). For clarification, the same ensemble of sources is used for all the observation runs so a magnitude conversion only really needs to happen once. 

## Reductions
The acquired data has been reduced using the HiPERCAM pipeline. We use C. Clark's function log2lc.py to calibrate instrumental magnitudes using our ensemble of sources and obtain separate light curves for each band (ugriz) for each observation block.

## GL_search.py
This python code is a "compact" and user-friendly way to search for the gravitational lensing in the companion's superior conjuction. There are two main sections to this function:

1. **Calibrating observation runs**
As most of our observation runs have been take on different nights, there are subtle magnitude differences that we encounter between them. As the lensing feature we are looking for is expected to be mili-magnitude, we want to make sure that what our program detects is lensing and not just calibration differences between the observation runs. For each band, we find a "baseline model" which is a simple double-peak, and calibrate each of the observation runs to this. This gives us a clean image that mitigates instrumental differences between the nights. A plot is made for each band called "calibration_{band}.png" to show the process.

2. **Searching for gravitational lensing using match template filtering**
We create a simple boxcar which we fit to a large number of possible phases and widths and see which correlates best with our data. It is up to the user to specifiy a certain range to these values. Additionally, the this match template filtering accounts for fitting and additional linear and quadratic to our data as an attempt to de-trend our data even further. We get an output of:
    - A correlation surface showing the best chosen phase and width of boxcar ("correlation_surface.png")
    - A plot showing all points for all bands and the fitted boxcar ("boxcar_all_bands.png")
    -   And most importantly, a plot containing separate subplots for each of the requested bands with the best fitted boxcar on each, and in the title the best centre phase, best width and amplitude of the boxcar as well as the score that can be regarded as a SNR ("GL_lensing_bands.png")

It is exactly the last plot that shows evidence for our sucessful detection of gravitational self-lensing in the PSR 1816-4510.

### Running GL_search.py 
The user should use config.txt to specify paths to reduction files, bands to perform the GL search on, orbital parameters and suchlike. Please preserve the current layout of the config.txt to avoid any issues with running the code. To run the file, it is enough to simply call "python GL_search.py config.txt" on command line. 

Note: See how the paths are structured in the config file and try to put the reduction file in a similar folder structure as the code creates keys for each of the observation runs from the grandparent directory rather than the parent. You might run into significant issues if the keys are assigned incorrectly.

## MCMC_sampler.py
Is an emcee-based MCMC sampler designed to run with the light curve binary sythesis code ICARUS that finds best fitting models for spider pulsar binaries.

## sigma_clipping.py 
Is a function that sigma clips our observation runs (note that this is now implemented in GL_search.py).

Happy coding!