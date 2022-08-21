"""
Fast_sweeping algorithm

Determines rupture onset times of patches along planar rectangular fault with
respect to an initial nucleation point.

References
----------
.. [Zhao2004] Zhao, Hongkai (2004), A fast sweeping method for eikonal
    equations, MATHEMATICS OF COMPUTATION, 74(250), 603-627,
    S 0025-5718(04)01678-3
"""

import numpy as num
import theano
import theano.tensor as tt
from theano.ifelse import ifelse

import fast_sweep_ext

km = 1000.0


def get_rupture_times_c(
    slowness, patch_size, n_patch_strike, n_patch_dip, nuc_x, nuc_y
):
    """
    C Implementation wrapper

    Slowness array has to be a flat array (1d).

    Parameters
    ----------
    slowness : :class:`numpy.NdArray`
        Matrix (2d, ( 1 x n_patch_dip * n_patch_strike) of slownesses of
        rupture on patches 1 / rupture_velocity [s / km]
    patch_size : float
        Size of slip patches [km]
    n_patch_strike : int
        Number of patches in strike direction of fault-plane
    n_patch_dip : int
        Number of patches in dip direction of fault-plane
    nuc_x : int
        Nucleation point of rupture in patch coordinate system on fault
        along strike [integer 0 left, n_patch_str right]
    nuc_y : int
        Nucleation point of rupture in patch coordinate system on fault
        along dip [integer 0 top n_patch_dip bottom]

    Returns
    -------
    tzero : :class:`numpy.NdArray` 1d (n_patch_dip * n_patch_strike)
        rupture onset times in s after hypocentral time

    Notes
    -----
    Here we call the C-implementation on purpose with swapped
    strike and dip directions, because we need the
    fault dipping in row directions of the array.
    The C-implementation has it along columns!!!
    """
    return fast_sweep_ext.fast_sweep(
        slowness, patch_size, nuc_y, nuc_x, n_patch_dip, n_patch_strike
    )


def get_rupture_times_numpy(
    Slowness, patch_size, n_patch_strike, n_patch_dip, nuc_x, nuc_y
):
    """
    Numpy implementation for reference.

    Parameters
    ----------
    Slowness : :class:`numpy.NdArray`
        Matrix (2d, n_patch_dip x n_patch_strike) of slownesses of
        rupture on patches 1 / rupture_velocity [s / km]
    patch_size : float
        Size of slip patches [km]
    n_patch_strike : int
        Number of patches in strike direction of fault-plane
    n_patch_dip : int
        Number of patches in dip direction of fault-plane
    nuc_x : int
        Nucleation point of rupture in patch coordinate system on fault
        along strike [integer 0 left, n_patch_str right]
    nuc_y : int
        Nucleation point of rupture in patch coordinate system on fault
        along dip [integer 0 top n_patch_dip bottom]

    Returns
    -------
    tzero : :class:`numpy.NdArray` (n_patch_dip, n_patch_strike)
        rupture onset times in s after hypocentral time
    """

    StartTimes = num.ones((n_patch_dip, n_patch_strike)) * 1e8
    StartTimes[nuc_y, nuc_x] = 0

    def upwind(
        dip_ind, str_ind, StartTimes, Slowness, patch_sz, n_patch_dip, n_patch_strike
    ):
        s1 = str_ind - 1
        d1 = dip_ind - 1
        s2 = str_ind + 1
        d2 = dip_ind + 1

        # if a < b return b
        if s1 < 0:
            checked_s1 = 0
        else:
            checked_s1 = s1

        if d1 < 0:
            checked_d1 = 0
        else:
            checked_d1 = d1

        # if a =< b return a-1
        if s2 >= n_patch_strike:
            checked_s2 = n_patch_strike - 1
        else:
            checked_s2 = s2

        if d2 >= n_patch_dip:
            checked_d2 = n_patch_dip - 1
        else:
            checked_d2 = d2

        ST_xmin = num.min(
            (StartTimes[checked_d1, str_ind], StartTimes[checked_d2, str_ind])
        )
        ST_ymin = num.min(
            (StartTimes[dip_ind, checked_s1], StartTimes[dip_ind, checked_s2])
        )

        ### Eikonal equation solver ###
        # The unique solution to the equation
        # [(x-a)^+]^2 + [(x-b)^+]^2 = f^2 * h^2
        # where a = u_xmin, b = u_ymin, is
        #
        #         | min(a,b) + f*h,                           |a-b|>= f*h
        # xnew =  |
        #         |0.5 * [ a+b+sqrt( 2*f^2*h^2 - (a-b)^2 ) ], |a-b| < f*h

        if num.abs(ST_xmin - ST_ymin) >= Slowness[dip_ind, str_ind] * patch_sz:
            start_new = (
                num.min((ST_xmin, ST_ymin)) + Slowness[dip_ind, str_ind] * patch_sz
            )
        else:
            start_new = (
                ST_xmin
                + ST_ymin
                + num.sqrt(
                    2
                    * num.power(Slowness[dip_ind, str_ind], 2)
                    * num.power(patch_sz, 2)
                    - num.power((ST_xmin - ST_ymin), 2)
                )
            ) / 2

        # if a < b return a

        if start_new >= StartTimes[dip_ind, str_ind]:
            start_new = StartTimes[dip_ind, str_ind]

        return start_new

    ### start main loop here ...
    num_iter = 1
    epsilon = 0.1
    err = 1e6
    while err > epsilon:
        Old_Times = StartTimes.copy()
        for ii in range(4):
            if ii == 0:
                for i in range(n_patch_dip):
                    for j in range(n_patch_strike):
                        StartTimes[i, j] = upwind(
                            i,
                            j,
                            StartTimes,
                            Slowness,
                            patch_size,
                            n_patch_dip,
                            n_patch_strike,
                        )
            if ii == 1:
                for i in range(n_patch_dip - 1, -1, -1):
                    for j in range(n_patch_strike):
                        StartTimes[i, j] = upwind(
                            i,
                            j,
                            StartTimes,
                            Slowness,
                            patch_size,
                            n_patch_dip,
                            n_patch_strike,
                        )

            if ii == 2:
                for i in range(n_patch_dip - 1, -1, -1):
                    for j in range(n_patch_strike - 1, -1, -1):
                        StartTimes[i, j] = upwind(
                            i,
                            j,
                            StartTimes,
                            Slowness,
                            patch_size,
                            n_patch_dip,
                            n_patch_strike,
                        )

            if ii == 3:
                for i in range(n_patch_dip):
                    for j in range(n_patch_strike - 1, -1, -1):
                        StartTimes[i, j] = upwind(
                            i,
                            j,
                            StartTimes,
                            Slowness,
                            patch_size,
                            n_patch_dip,
                            n_patch_strike,
                        )

        err = num.sum(num.sum(num.power((StartTimes - Old_Times), 2)))
        num_iter = num_iter + 1

    return StartTimes


def get_rupture_times_theano(slownesses, patch_size, nuc_x, nuc_y):
    """
    Does the same calculation as get_rupture_times_numpy
    just with symbolic variable input and output for theano graph
    implementation optimization.
    """
    [step_dip_max, step_str_max] = slownesses.shape
    StartTimes = tt.ones((step_dip_max, step_str_max)) * 1e8
    StartTimes = tt.set_subtensor(StartTimes[nuc_y, nuc_x], 0)

    # Stopping check var
    epsilon = theano.shared(0.1)
    err_val = theano.shared(1e6)

    # Iterator matrixes
    dip1 = tt.repeat(tt.arange(step_dip_max), step_str_max)
    str1 = tt.tile(tt.arange(step_str_max), step_dip_max)

    dip2 = tt.repeat(tt.arange(step_dip_max), step_str_max)
    str2 = tt.tile(tt.arange(step_str_max - 1, -1, -1), step_dip_max)

    dip3 = tt.repeat(tt.arange(step_dip_max - 1, -1, -1), step_str_max)
    str3 = tt.tile(tt.arange(step_str_max - 1, -1, -1), step_dip_max)

    dip4 = tt.repeat(tt.arange(step_dip_max - 1, -1, -1), step_str_max)
    str4 = tt.tile(tt.arange(step_str_max), step_dip_max)

    DIP = tt.concatenate([dip1, dip2, dip3, dip4])
    STR = tt.concatenate([str1, str2, str3, str4])

    ### Upwind scheme ###
    def upwind(dip_ind, str_ind, StartTimes, slownesses, patch_size):
        [n_patch_dip, n_patch_str] = slownesses.shape
        zero = theano.shared(0)
        s1 = str_ind - 1
        d1 = dip_ind - 1
        s2 = str_ind + 1
        d2 = dip_ind + 1

        # if a < b return b
        checked_s1 = ifelse(tt.lt(s1, zero), zero, s1)
        checked_d1 = ifelse(tt.lt(d1, zero), zero, d1)

        # if a =< b return a-1
        checked_s2 = ifelse(tt.le(n_patch_str, s2), n_patch_str - 1, s2)
        checked_d2 = ifelse(tt.le(n_patch_dip, d2), n_patch_dip - 1, d2)

        ST_xmin = tt.min(
            (StartTimes[checked_d1, str_ind], StartTimes[checked_d2, str_ind])
        )
        ST_ymin = tt.min(
            (StartTimes[dip_ind, checked_s1], StartTimes[dip_ind, checked_s2])
        )

        ### Eikonal equation solver ###
        # The unique solution to the equation
        # [(x-a)^+]^2 + [(x-b)^+]^2 = f^2 * h^2
        # where a = u_xmin, b = u_ymin, is
        #
        #         | min(a,b) + f*h,                           |a-b|>= f*h
        # xnew =  |
        #         |0.5 * [ a+b+sqrt( 2*f^2*h^2 - (a-b)^2 ) ], |a-b| < f*h
        start_new = ifelse(
            tt.le(
                slownesses[dip_ind, str_ind] * patch_size, tt.abs_(ST_xmin - ST_ymin)
            ),
            tt.min((ST_xmin, ST_ymin)) + slownesses[dip_ind, str_ind] * patch_size,
            (
                ST_xmin
                + ST_ymin
                + tt.sqrt(
                    2 * tt.pow(slownesses[dip_ind, str_ind], 2) * tt.pow(patch_size, 2)
                    - tt.pow((ST_xmin - ST_ymin), 2)
                )
            )
            / 2,
        )

        # if a < b return a
        output = ifelse(
            tt.lt(start_new, StartTimes[dip_ind, str_ind]),
            start_new,
            StartTimes[dip_ind, str_ind],
        )
        return tt.set_subtensor(
            StartTimes[dip_ind : dip_ind + 1, str_ind : str_ind + 1], output
        )

    def loop_upwind(StartTimes, PreviousTimes, err_val, iteration, epsilon):
        [results, updates] = theano.scan(
            fn=upwind,
            sequences=[DIP, STR],
            outputs_info=[StartTimes],
            non_sequences=[slownesses, patch_size],
        )

        StartTimes = results[-1]
        err_val = tt.sum(tt.sum(tt.pow((StartTimes - PreviousTimes), 2)))

        PreviousTimes = StartTimes.copy()
        return (
            (StartTimes, PreviousTimes, err_val, iteration + 1),
            theano.scan_module.until(err_val < epsilon),
        )

    # while loop until err < epsilon
    iteration = theano.shared(0)
    PreviousTimes = StartTimes.copy()
    ([result, PreviousTimes, errs, Iteration], updates) = theano.scan(
        fn=loop_upwind,
        outputs_info=[StartTimes, PreviousTimes, err_val, iteration],
        non_sequences=[epsilon],
        n_steps=500,
    )  # arbitrary set, stops after few iterations
    return result[-1]
