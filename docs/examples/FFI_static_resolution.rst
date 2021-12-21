
Example 4b: Static finite-fault estimation, resolution based patch discretization
---------------------------------------------------------------------------------

In this example we will determine a variable slip distribution for the L'aquila 2009 earthquake by using static InSAR data.
In contrast to `Example 4a <https://pyrocko.org/beat/docs/current/examples/FFI_static.html#>` we will use resolution based
discretization of fault patches following the approach of [Atzori&Antonioli2011]_. It is assumed that the reader has Example 4a completed before following this example, in order to be familiar with the concepts and commands for uniform discretized faults.

Clone
^^^^^
Please clone the config_ffi.yaml from the previous uniform static FFI run for the Laquila earthquake::

  beat clone Laquila Laquila_resolution --mode=ffi --datatypes=geodetic --copy_data

Calculate Greens Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^
In the uniformly discretized fault variable slip estimation we did define the reference fault geometry based on the results of `Example 3 <https://pyrocko.org/beat/docs/current/examples/Rectangular.html#>` where we did estimate the fault geometry parameters. Based on the reference fault and the available data observations the model resolution matrix can be calculated and the fault can be divided into patches such that a defined threshold of resolution is met. For details on the algorithm I refer the reader to the original article of [Atzori&Antonioli2011]_.

In this example we want to discretize the fault surface using such a resolution based discretization. Todo so, please set the *discretization* attribute of the *gf_config* to *resolution* and run the update command to display changes to the config::

  beat update Laquila_resolution --mode=ffi --diff

Rerun without --diff to apply the changes::

  beat update Laquila_resolution --mode=ffi

The discretization config should look now like this:

    discretization: resolution
    discretization_config: !beat.ResolutionDiscretizationConfig
      extension_widths:
      - 0.1
      extension_lengths:
      - 0.1
      epsilon: 0.005
      resolution_thresh: 0.95
      depth_penalty: 3.5
      alpha: 0.3
      patch_widths_min:
      - 1.0
      patch_widths_max:
      - 5.0
      patch_lengths_min:
      - 1.0
      patch_lengths_max:
      - 5.0


References
^^^^^^^^^^
.. [Atzori&Antonioli2011] Atzori, S. and Antonioli, A. (2011). Optimal fault resolution in geodetic inversion of coseismic data. Geophysical Journal International, 185:529–538
