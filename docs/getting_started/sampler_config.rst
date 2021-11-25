
Notes on sampler configuration
------------------------------

Sequential Monte Carlo (SMC) Sampler
====================================

This question comes up a lot and I will write this week a longer documentation for the webpage. Short points to consider:

- number of unknown parameters: The more unknown parameters you have the higher you need to set your sampler parameters.

What is the problem you are estimating? Only MT, with/without station corrections? with/without noise scalings?

E.g. if you have around 10 parameters (location, MT components, moment magnitude) you can significantly reduce the default sampler parameters to e.g.:

nchains=200, nsteps=400, buffer_thinning=20, workers= as many as your hardware allows (e.g. 10)

You may be able to further decrease these numbers, I havent had the chance to systematically test that yet. I tend to be conservative on those. But this could be something you could investigate yourself.

With 20hrs of runtime for an MT problem you likely have used very high sampler parameters, but thus you can almost be certain that you arrived at the global maximum likelihood.

Now you could use the same setup simply changing the sampler parameters to lower as suggested and mybe even lower later until you notice significant deviation from your originl well sempled setup.

If you do this I would be very interested to hear your findings and if you agree to include your statements on the webpage. I and all the other users would appreciate your efforts very much!

Of course your names would be acknowledged.


If you will find you have around 100 parameters you rather need to go to 1500 chains.

- sampling rate of the data (seismic only) - do you really need high sampling rate? or are you only doing teleseismic body waveform inversion up to 0.1 Hz, thus 1Hz sample rate might be sufficient

- taper duration (a-d)- the longer the more costly the forward model

- number of observations (seismic/geodetic stations): are there redundant stations close to each other where one could be removed?


