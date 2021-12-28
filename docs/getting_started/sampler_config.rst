
Notes on sampler configuration
------------------------------

Sequential Monte Carlo (SMC) Sampler
====================================

This document discusses points to consider for the configuration of the SMC sampler. The most important
point to consider is the **number of unknown parameters**. The higher the number of unknown parameters the configured model setup contains, the higher the sampler parameters need to be set.

What are unknown parameters in the configured problem that affect the setup? 

 * the parameterization of the source (e.g. MT, RectangularSource, FFI ...) 
 * station corrections, i.e. time shifts on data traces (number of data traces per waveform fit config, e.g. 21 waveforms fitting the P wave on the Z component) 
 * noise scalings affected by *dataset_specific_residual_noise estimation*:

   + "True" - one scaling parameter per dataset (e.g. 21 scalings for 21 waveforms fitting the P wave on 
     the  Z component, or 3 scaling parameters for 3 displacement maps, or 5 scaling parameters for 5 GNSS stations)
   + "False" - one parameter per dataset group (e.g. 1 scaling parameter for 21 waveforms fitting the P 
     wave on the Z component, 1 scaling parameter for 3 displacement maps, or 1 scaling parameter for 5 GNSS stations)
 * other hierarchical parameters, e.g. *ramp corrections* (InSAR), *euler_pole corrections* ...

The following table shows some example model configurations and the respective recommended SMC sampler configuration:

+------------+------------------+-------------+----------+---------+----------+
|           Model configuration               |   Total  |  Sampler parameters|
+------------+------------------+-------------+----------+---------+----------+
| source     |noise scalings    |hierarchicals|Parameters|*n_steps*|*n_chains*|
+============+==================+=============+==========+=========+==========+
| MTQTsource |                  |             |          |         |          |
| location   | 21 waveforms     | 21 waveforms|          |         |          |
| magnitude  | False            | False       |          |         |          |
+------------+------------------+-------------+----------+---------+----------+
| 5 + 3 + 1  | 1                | 1           | 11       | 500     | 200      |
+------------+------------------+-------------+----------+---------+----------+
| 2 sources  | - 3 SAR (True)   | - ramp      |          |         |          |
| Rectangular| - 5 GNSS (True)  |   (True)    |          |         |          |
| location   | - 15 P waveform  | - P: True   |          |         |          |
| STF        |   (True) - Z     | - S: True   |          |         |          |
|            | - 8 S waveforms  |             |          |         |          |
|            |   (True) - T     |             |          |         |          |
+------------+------------------+-------------+----------+---------+----------+
| 2 *        | 3 + 5 + 15 + 8   | 3 * 3 +     | 86       | 300     | 1500     |
|(6 + 3 + 2) |                  | 15 + 8      |          |         |          |
+------------+------------------+-------------+----------+---------+----------+
| FFI        | - 3 SAR (True)   | - ramp      |          |         |          |
| 150 patches| - 5 GNSS (True)  |   (False)   |          |         |          |
| nucleation | - 15 P waveform  | - P: True   |          |         |          |
| point      |   (True) - Z     | - S: True   |          |         |          |
|            | - 8 S waveforms  | - laplacian |          |         |          |
|            |   (True) - T     |             |          |         |          |
+------------+------------------+-------------+----------+---------+----------+
| 150 * 4 + 2| 3 + 5 + 15 + 8   | 15 + 8 + 1  | 675      | 300     | 10000    |
+------------+------------------+-------------+----------+---------+----------+

.. note:: The aim of the configuration is, to make convergence of the sampler likely. Nevertheless, it may happed that the sampler does not converge and values need to be adjusted. Each model setup is individual and success or failure to reach convergence also depends on other factors e.g. choice of source parameterization, noise parameterization, available data ... 
 