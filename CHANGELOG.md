# Changelog

All notable changes to BEAT will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).


## [1.2.2] 28 October 2022
Contributors: Hannes Vasyura-Bathke @hvasbath

### Added
- plotting.marginals.traceplot: source_idxs argument can be slice e.g. 1:10 to take mean of patch attributes
- heart.seis_derivative: function to numerically calculate derivatives wrt source attributes for waveforms
- heart.py, config.py: allow for version control for fomosto backends through "version" argument at *gf_config*
- utility: functions for slice to string conversion and vice versa
- config; NonlinearGFConfig added version attribute to specify backend version to use for GF calculation

### Changed
- plotting.marginals.traceplot: CDF plotting of multiple distributions in a single subplot marks quantiles only once

### Fixed
- plotting.marginals.traceplot: multipage output repeating variables


## [1.2.1] 14 September 2022
Contributors: Hannes Vasyura-Bathke @hvasbath

### Added
**FFI**
- add calculation of coupling to derived variables at *Fault* for the summarize method

**plotting**
- *plot_projection* argument available for *stage_posteriors* plot: cdf, pdf, kde
- update Example 8 to showcase the use of this functionality

### Fixed
- stage_posterior plot supports multipage output, figsize fixed to fractions of A4
- multievent waveform_fits plot returns separate figures for each sub-event


## [1.2.0] 21 August 2022
Contributors: Mahdi Hamidbeygi @mahdihamidbeygi, Hannes Vasyura-Bathke @hvasbath

### Added
**Polarity**
- polarity module/composite/dataset-type for inference of wave onset polarities
- joint inference of P, Sh and Sv wave-polarities
- drawing of piercing points on fuzzyBB plot if polarity data exists
- drawing of fuzzyBBs for Sh and Sv waves
- add tutorial Example 8 for polarity inference

**Amplitude Spectra**
- added string-choice `domain` to WaveMaps to choose `time` or `frequency` domain for inference
- plotting: added spectra fits in waveform fits plot

**Continuous Integration (CI)**
- adapted installation scheme to fulfill PEP517 through pip and .toml
- providing beat binary packages on PyPi
- github actions workflow for code formatting and wheel building
- pre-commit and hooks for yaml, spellchecking, tailing whitespaces and eof

### Changed
- split plotting module into plotting directory with submodules
- plotting: stage_posteriors change hist color to grey if only single density is shown
- plotting: fuzzyBB dashed white and black line instead of red for MAP
- docs: updated (short) installation instructions to use the package manager pip instead of setup.py

## [1.1.1] 6th January 2022

### Added
**FFI**
- chain initialisation with lsq solution for seismic data
- resolution based discretization:
  * add improved method of Atzori et al. 2019
  * automatic damping estimation
  * add Tutorial 4b to demonstrate the usage and functionality

**Plotting**
- 3d_slip_distribution: allows plotting of selected segments, slip-deficit, coupling, slip_variation
- gnss_fits: added Variance Reduction histograms if nensemble > 1
- slip_distribution: common colorscale and spatial scale across subfaults
- moment_rate: adjusted size and formatting to be publication ready
- correlation_hist: plot source related correlations for nsources > 1, colorcoding

### Changed
- docs: rename former example 4 to example 4a

### Fixed
- Bin backend: loading of traces with corrupt file header
- beat import: just import sampled variables
- RingFaultSource: allow sampling of slip-direction (sign)

## [1.1.0] 12th April 2021

### General

Documentation moved to https://pyrocko.org/beat and is made version dependent.
E.g. https://pyrocko.org/beat/v1.0 to view older documentation versions.

### Added
**General**
- Wavemap attribute "quantity" to sample acceleration, velocity or displacement waveforms
- added export method to geodetic composite to enable GNSS and SAR data / results export to CSV, YAML

**Tensile Dislocations**
- added the opening_fraction argument to the RectangularSource to allow to model tensile opening/closing
- for geometry mode, as well as ffi mode to estimate distributed opening / closing

**Finite Fault**
- discretization options for the fault into patches:
  + uniform
  + resolution based (after Atzori & Antonioli 2011 GJI) for geodetic data valid only
- support for multiple subfaults in FFI mode:
  + Laplacian smoothing with different functional forms:
    * distance based:
      ** Gaussian
      ** Exponential (after Radiget et al. 2011)
    * only neighbors:
      ** nearest_neighbor for single subfault only
  + geodetic data
  + seismic data with nucleation point for each subfault
- allow for station corrections
- subfault wise, prior bounds definition
- easier fault geometry setup:
  + beat check --what=geometry allows to pipe fault setup into Talpa for GUI interactive fault editing
  + beat import --results=saved_geometry.yaml --mode=ffi imports the saved_geometry.yaml to the beat config_ffi

**Hierarchicals**
- generalized hierarchical corrections to support various types:
  + geodetic:
    * orbital ramp
    * Euler pole rotation

**Plotting**
- slip_distribution allows for variable patch-size
- plot gnss_fits for horizontal and vertical GNSS components
- waveform misfits:
  + with plot_projection =individual allows to get individual source contributions for geometry mode
  + added time-shift histograms if these were sampled
  + added variance reduction histograms
  + added amplitude scales
- new station map with GMT, allows time shift plotting
- FFI exports geometry object for fast and easy interactive inspection in pyrocko.sparrow
- Lune plot for Moment Tensors (Tape and Tape 2012, 2015)

### Fixed
- extended support for GNSS data import from globk (GAMMIT)
- SMC saving of sampler state does not dump full model graph
- multiprocess FFI performance increase
- MTQTSource magnitude scaling

### Changed
- fit_ramp option at the geodetic_config is now an Hierarchical Correction
- laplacian has now configuration arguments for functional form, nearest neighbor as previously now only for single fault
- moved beat command level export to dataset composites export method
- MTQTSource input argument u removed in favor of w
- filterer attribute of Wavemap is now list of filters, allows chaining of filters
- beat import mode now cleanly refers to the current project directory not anymore to the results directory to be imported from
- beat import got additional --import_from_mode to choose the mode of the results to import from (was previously mode)

## [v.1.0.0]  18.06.2019
Initial release:

### Added
- estimate non-linear parameters of elastic deformation sources
- finite fault inversion with distributed slip on uniformly discretized planar faults
- in layered/homogeneous elastic media
- supported datatypes:
  + geodetic (InSAR, GNSS)
  + seismic (seismic waveforms)
- Bayesian Inference with hierarchical residual estimation
- several sampling algorithms:
  + Adaptive Metropolis Hastings
  + Parallel Tempering (Replica Exchange)
  + Sequential Monte Carlo
- included plotting
