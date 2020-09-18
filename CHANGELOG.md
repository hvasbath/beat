# Changelog

All notable changes to BEAT will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [unreleased]

### Added
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
- waveform misfits with plot_projection =individual allows to get individual source contributions for geometry mode
- new station map with GMT, allows time shift plotting
- FFI exports geometry object for fast and easy interactive inspection in pyrocko.sparrow

### Fixed
- extended support for GNSS data import from globk (GAMMIT)
- SMC saving of sampler state does not dump full model graph
- multiprocess FFI performance increase
- MTQTSource magnitude scaling

### Changed
- fit_ramp option at the geoedetic_config is now an Hierarchical Correction
- laplacian has now configuration arguments for functional form, nearest neighbor as previously now only for single fault
- moved beat command level export to dataset composites export method
- MTQTSource input argument u removed in favor of w

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
