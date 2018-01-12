import theano


def GF_stacking_2scan_complete_rtint(GFLibrary_parr, GFLibrary_perp, GFTimes, CutInterval, Risetimes, StartTimes, Weights_parr, Weights_perp, deltat):
    ''' Time-Shift, weight and sum Greensfunctions for one target for all patches (sources).
        T and Z component done. Cutting before stacking. Risetime interpolation.
        GFLibrary - Greensfunction matrix (patch_num, ydata)
        CutInterval - Integer [tmin, tmax in s] of cutout around phase arrival
        Risetimes - Risetimes of the patches
        StartTimes - Rupture-onset from Fast_sweeping
        Weights    - Scalar slip [m] as symbolic variable
        delta_t    - sample rate of Greensfunctions '''
        
    n_targets, n_patches, n_rt, GF_l  = GFLibrary_parr.shape
    CutInterval = CutInterval/deltat

    # Iterator Matrix
    PATCHES = tt.arange(n_patches, dtype='int16')
    TARGETS = tt.arange(n_targets, dtype='int16')

    def write_one_patch(p, Ceil_Risetime, IntFacST, IntFacRT, I_start, I_end, Weight_parr, Weight_perp, Shifted_GFs, GFLibrary_parr, GFLibrary_perp):
        # interpolation in time samples and cutting around phase arival, interpolation in risetime
        output_parr_ceil = IntFacRT[1] * ((GFLibrary_parr[p, Ceil_Risetime, I_start:I_end] * Weight_parr * IntFacST[1]) + \
                       (GFLibrary_parr[p, Ceil_Risetime, I_start - 1:I_end - 1] * Weight_parr * IntFacST[0]))
        output_parr_floor = IntFacRT[0] * ((GFLibrary_parr[p, Ceil_Risetime - 1, I_start:I_end] * Weight_parr * IntFacST[1]) + \
                       (GFLibrary_parr[p, Ceil_Risetime - 1, I_start - 1:I_end -1 ] * Weight_parr * IntFacST[0]))

        output_perp_ceil = IntFacRT[1] * ((GFLibrary_perp[p, Ceil_Risetime, I_start:I_end] * Weight_perp * IntFacST[1]) + \
                       (GFLibrary_perp[p, Ceil_Risetime, I_start - 1:I_end - 1] * Weight_parr * IntFacST[0]))
        output_perp_floor = IntFacRT[0] * ((GFLibrary_perp[p, Ceil_Risetime - 1, I_start:I_end] * Weight_perp * IntFacST[1]) + \
                       (GFLibrary_perp[p, Ceil_Risetime -1, I_start - 1:I_end - 1] * Weight_parr * IntFacST[0]))
        return Shifted_GFs + (output_parr_ceil + output_parr_floor + output_perp_ceil + output_perp_floor)

    def write_one_target(T, Synthetics, PATCHES, GFLibrary_parr, GFLibrary_perp, GFTimes, 
                         Risetimes, StartTimes, Weights_parr, Weights_perp, 
                         deltat, n_patches, CutInterval):
        # allocate empty trace
        nsamples = tt.cast(tt.ceil(CutInterval.sum()), dtype='int16')
        Shifted_GFs = tt.alloc(0, nsamples).astype(dtype=config.floatX)

        # Calculate Interpolation Factors for Risetime
        Ceil_Risetimes = tt.ceil(Risetimes/deltat).astype('int16')
        IntFacsRT = tt.concatenate([((Ceil_Risetimes-(Risetimes/deltat)), 
                              (1.- (Ceil_Risetimes-(Risetimes/deltat))))]).T

        ## APPLY TIME SHIFTS AND CALCULATE TIME INTERPOLATION FACTORS
        # column wise index referencing --> no common time anymore as cut around
        # phase during GF calculation
        delayed_Times = (GFTimes[T] - tt.repeat((GFTimes[T,:,0] - StartTimes), 3).reshape((n_patches, 3)))
        
        Ceil_ArrTimes = tt.ceil(delayed_Times[:,1])
        IntFacsST = tt.concatenate([((Ceil_ArrTimes - delayed_Times[:,1]), 
                              (1.- (Ceil_ArrTimes - delayed_Times[:,1])))]).T
        SampleArr = tt.ceil(delayed_Times[:,1]/deltat).astype('int16')
        RefArr = SampleArr.min()
        I_starts = (2*RefArr - SampleArr - CutInterval[0]).astype('int16')
        I_ends = I_starts + nsamples

        [results, _] = theano.scan(fn=write_one_patch,
                        sequences = [PATCHES, Ceil_Risetimes, IntFacsST, IntFacsRT, I_starts, I_ends, Weights_parr, Weights_perp],
                        outputs_info = [Shifted_GFs],
                        non_sequences = [GFLibrary_parr[T], GFLibrary_perp[T]],
                        strict = True)

        output = results[-1].flatten()
        Synthetics = tt.set_subtensor(Synthetics[(T):(T+1),:], output)
        return Synthetics
    
    # create output matrix
    samples_arrivals = tt.cast(CutInterval.sum(), dtype='int16')
    Synthetics = tt.alloc(0, n_targets, samples_arrivals).astype(config.floatX)
    # loop targets
    [results, _] = theano.scan(fn=write_one_target,
                        sequences = [TARGETS],
                        outputs_info = [Synthetics],
                        non_sequences = [PATCHES, 
                                        GFLibrary_parr, 
                                        GFLibrary_perp, 
                                        GFTimes,
                                        Risetimes, StartTimes, Weights_parr, Weights_perp, 
                                        deltat, n_patches, CutInterval],
                        strict = True)
    
    return results[-1]
