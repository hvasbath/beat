# coding=utf-8
import numpy as num
import logging
import os
import shutil
import signal

from tempfile import mkdtemp
from subprocess import Popen, PIPE
from os.path import join as pjoin

from pyrocko.guts import Float, Int, Tuple, List, Object, String
from pyrocko import gf


km = 1000.

guts_prefix = 'pf'

Timing = gf.meta.Timing

logger = logging.getLogger('beat')

# how to call the programs
program_bins = {
    'pscmp.2008a': 'fomosto_pscmp2008a',
}

psgrn_displ_names = ('uz', 'ur', 'ut')
psgrn_stress_names = ('szz', 'srr', 'stt', 'szr', 'srt', 'stz')
psgrn_tilt_names = ('tr', 'tt', 'rot')
psgrn_gravity_names = ('gd', 'gr')


def str_float_vals(vals):
    return ' '.join('%e' % val for val in vals)


def str_int_vals(vals):
    return ' '.join('%i' % val for val in vals)


def str_str_vals(vals):
    return ' '.join("'%s'" % val for val in vals)


def dsin(value):
    return num.sin(value * num.pi / 180.)


def dcos(value):
    return num.cos(value * num.pi / 180.)


def distributed_fault_patches_to_config(patches):
    '''
    Input: List of PsCmpRectangularSource(s)
    '''
    srows = []
    for i, patch in enumerate(patches):
            srows.append('%i %s' % (i + 1, patch.string_for_config()))

    return '\n'.join(srows), len(srows)


class PsCmpObservation(Object):
    pass


class PsCmpScatter(PsCmpObservation):
    lats = List.T(Float.T(), optional=True, default=[10.4, 10.5])
    lons = List.T(Float.T(), optional=True, default=[12.3, 13.4])

    def string_for_config(self):
        srows = []
        for lat, lon in zip(self.lats, self.lons):
            srows.append('(%15f, %15f)' % (lat, lon))

        self.sw = 0
        return ' %i' % (len(srows)), '\n'.join(srows)


class PsCmpProfile(PsCmpObservation):
    n_steps = Int.T(default=10)
    start_distance = Float.T(
        default=0.,
        help='minimum distance from source [m]')
    end_distance = Float.T(
        default=100. * km,
        help='minimum distance from source [m]')

    def string_for_config(self):
        self.sw = 1
        self.start_distance /= km   # convert to km as pscmp takes [km]
        self.end_distance /= km
        return ' %i' % (self.n_steps), \
            ' ( %(start_distance)15f, %(end_distance)15f )' % self.__dict__


class PsCmpArray(PsCmpObservation):
    n_steps_x = Int.T(default=10)
    n_steps_y = Int.T(default=10)
    start_distance_x = Float.T(
        default=0.,
        help='minimum distance in x-direction (E) from source [m]')
    end_distance_x = Float.T(
        default=100. * km,
        help='maximum distance in x-direction (E) from source [m]')
    start_distance_y = Float.T(
        default=0.,
        help='minimum distance in y-direction (N) from source [m]')
    end_distance_y = Float.T(
        default=100. * km,
        help='minimum distance in y-direction (N) from source [m]')
    def string_for_config(self):
        self.sw = 2
        self.start_distance_x /= km   # convert to km as pscmp takes [km]
        self.end_distance_x /= km
        self.start_distance_y /= km
        self.end_distance_y /= km

        return ' %(n_steps_x)i %(start_distance_x)15f %(end_distance_x)15f ' % self.__dict__, \
               ' %(n_steps_y)i %(start_distance_y)15f %(end_distance_y)15f ' % self.__dict__


class PsCmpRectangularSource(gf.Location, gf.seismosizer.Cloneable):
    '''
    Input parameters have to be in:
    [deg] for reference point (lat, lon) and angles (rake, strike, dip)
    [m] shifting with respect to reference position
    [m] for fault dimensions and source depth. The default shift of the
    origin (pos_s, pos_d) with respect to the reference coordinates
    (lat, lon) is zero.
    '''
    length = Float.T(default=6.0 * km)
    width = Float.T(default=5.0 * km)
    strike = Float.T(default=0.0)
    dip = Float.T(default=90.0)
    rake = Float.T(default=0.0)
    torigin = Float.T(default=0.0)

    slip = Float.T(optional=True, default=1.0)
    strike_slip = Float.T(optional=True, default=None)
    dip_slip = Float.T(optional=True, default=None)
    pos_s = Float.T(optional=True, default=None)
    pos_d = Float.T(optional=True, default=None)
    opening = Float.T(default=0.0)

    def update(self, **kwargs):
        '''Change some of the source models parameters.

        Example::

          >>> from pyrocko import gf
          >>> s = gf.DCSource()
          >>> s.update(strike=66., dip=33.)
          >>> print s
          --- !pf.DCSource
          depth: 0.0
          time: 1970-01-01 00:00:00
          magnitude: 6.0
          strike: 66.0
          dip: 33.0
          rake: 0.0

        '''
        for (k, v) in kwargs.iteritems():
            self[k] = v

    def convert_slip(self):
        dip_slip = float(self.slip * dsin(self.rake) * (-1))
        strike_slip = float(self.slip * dcos(self.rake))
        return dip_slip, strike_slip

    def string_for_config(self):
        self.__dict__['effective_lat'] = self.effective_lat
        self.__dict__['effective_lon'] = self.effective_lon

        if self.strike_slip or self.dip_slip is None:
            self.dip_slip, self.strike_slip = self.convert_slip()

        if self.pos_s or self.pos_d is None:
            self.pos_s = 0.
            self.pos_d = 0.

        self.length /= km
        self.width /= km
        return '%(effective_lat)15f %(effective_lon)15f %(depth)15f' \
               '%(length)15f %(width)15f %(strike)15f' \
               '%(dip)15f 1 1 %(torigin)15f \n %(pos_s)15f %(pos_d)15f ' \
               '%(strike_slip)15f %(dip_slip)15f %(opening)15f' % self.__dict__


class PsCmpCoulombStress(Object):
    pass


class PsCmpCoulombStressMasterFault(PsCmpCoulombStress):
    friction = Float.T(default=0.7)
    skempton_ratio = Float.T(default=0.0)
    master_fault_strike = Float.T(default=300.)
    master_fault_dip = Float.T(default=15.)
    master_fault_rake = Float.T(default=90.)
    sigma1 = Float.T(default=1.e6)
    sigma2 = Float.T(default=-1.e6)
    sigma3 = Float.T(default=0.0)

    def string_for_config(self):
        return '%(friction)15e %(skempton_ratio)15e %(master_fault_strike)15f' \
               '%(master_fault_dip)15f %(master_fault_rake)15f' \
               '%(sigma1)15e %(sigma2)15e %(sigma3)15e' % self.__dict__


class PsCmpConfig(Object):

    pscmp_version = String.T(default='2008a')
    # scatter, profile or array
    observation = PsCmpObservation.T(default=PsCmpScatter.D())

    pscmp_outdir = String.T(default='./')
    psgrn_outdir = String.T(default='./psgrn_functions/')

    los_vector = Tuple.T(3, Float.T(), optional=True)

    times_snapshots = List.T(Float.T(), optional=True)

    rectangular_source_patches = List.T(default=PsCmpRectangularSource.D())


class PsCmpConfigFull(PsCmpConfig):

    sw_los_displacement = Int.T(default=0)
    sw_coulomb_stress = Int.T(default=0)
    coulomb_master_field = PsCmpCoulombStress.T(
        optional=True,
        default=PsCmpCoulombStressMasterFault.D())

    displ_sw_output_types = Tuple.T(3, Int.T(), default=(1, 1, 1))
    stress_sw_output_types = Tuple.T(6, Int.T(), default=(0, 0, 0, 0, 0, 0))
    tilt_sw_output_types = Tuple.T(3, Int.T(), default=(0, 0, 0))
    gravity_sw_output_types = Tuple.T(2, Int.T(), default=(0, 0))

    displ_filenames = Tuple.T(3, String.T(), default=psgrn_displ_names)
    stress_filenames = Tuple.T(6, String.T(), default=psgrn_stress_names)
    tilt_filenames = Tuple.T(3, String.T(), default=psgrn_tilt_names)
    gravity_filenames = Tuple.T(2, String.T(), default=psgrn_gravity_names)

    snapshot_basefilename = String.T(default='snapshot')

    @staticmethod
    def example():
        conf = PsCmpConfigFull()
        conf.psgrn_outdir = 'TEST_psgrn_functions/'
        conf.pscmp_outdir = 'TEST_pscmp_output/'
        conf.rectangular_source_patches = [PsCmpRectangularSource(
                                lat=10., lon=10., slip=2.,
                                width=5., length=10.,
                                strike = 45, dip=30, rake=-90)]
        conf.observation = PsCmpArray(
                start_distance_x=9.5, end_distance_x=10.5, n_steps_x=150,
                start_distance_y=9.5, end_distance_y=10.5, n_steps_y=150)
        conf.times_snapshots = [0]
        return conf

    def get_output_filenames(self, rundir):
        return [pjoin(rundir, self.snapshot_basefilename+'_'+str(i+1)+'.txt')
                for i in range(len(self.times_snapshots))]

    def string_for_config(self):
        d = self.__dict__.copy()

        patches_str, n_patches = distributed_fault_patches_to_config(
                        self.rectangular_source_patches)

        d['patches_str'] = patches_str
        d['n_patches'] = n_patches

        str_npoints, str_observation = self.observation.string_for_config()
        d['str_npoints'] = str_npoints
        d['str_observation'] = str_observation
        d['sw_observation_type'] = self.observation.sw

        if self.times_snapshots:
            str_times_dummy = []
            for i, time in enumerate(self.times_snapshots):
                 str_times_dummy.append("%f  '%s_%i.txt'\n" %(time, self.snapshot_basefilename, i+1))

            str_times_dummy.append('#')
            d['str_times_snapshots'] = ''.join(str_times_dummy)
            d['n_snapshots'] = len(str_times_dummy) -1
        else:
            d['str_times_snapshots'] = '# '
            d['n_snapshots'] = 0

        if self.sw_los_displacement == 1:
            d['str_los_vector'] = str_float_vals(self.los_vector)
        else:
            d['str_los_vector'] = ''

        if self.sw_coulomb_stress == 1:
            d['str_coulomb_master_field'] = self.coulomb_master_field.string_for_config()
        else:
            d['str_coulomb_master_field'] = ''

        d['str_psgrn_outdir'] = "'../../%s'" % self.psgrn_outdir
        d['str_pscmp_outdir'] = "'%s'" % './'

        d['str_displ_filenames'] = str_str_vals(self.displ_filenames)
        d['str_stress_filenames'] = str_str_vals(self.stress_filenames)
        d['str_tilt_filenames'] = str_str_vals(self.tilt_filenames)
        d['str_gravity_filenames'] = str_str_vals(self.gravity_filenames)

        d['str_displ_sw_output_types'] = str_int_vals(self.displ_sw_output_types)
        d['str_stress_sw_output_types'] = str_int_vals(self.stress_sw_output_types)
        d['str_tilt_sw_output_types'] = str_int_vals(self.tilt_sw_output_types)
        d['str_gravity_sw_output_types'] = str_int_vals(self.gravity_sw_output_types)

        template = '''# autogenerated PSCMP input by pscmp.py
#===============================================================================
# This is input file of FORTRAN77 program "pscmp08" for modeling post-seismic
# deformation induced by earthquakes in multi-layered viscoelastic media using
# the Green's function approach. The earthquke source is represented by an
# arbitrary number of rectangular dislocation planes. For more details, please
# read the accompanying READ.ME file.
#
# written by Rongjiang Wang
# GeoForschungsZentrum Potsdam
# e-mail: wang@gfz-potsdam.de
# phone +49 331 2881209
# fax +49 331 2881204
#
# Last modified: Potsdam, July, 2008
#
# References:
#
# (1) Wang, R., F. Lorenzo-Martín and F. Roth (2003), Computation of deformation
#     induced by earthquakes in a multi-layered elastic crust - FORTRAN programs
#     EDGRN/EDCMP, Computer and Geosciences, 29(2), 195-207.
# (2) Wang, R., F. Lorenzo-Martin and F. Roth (2006), PSGRN/PSCMP - a new code for
#     calculating co- and post-seismic deformation, geoid and gravity changes
#     based on the viscoelastic-gravitational dislocation theory, Computers and
#     Geosciences, 32, 527-541. DOI:10.1016/j.cageo.2005.08.006.
# (3) Wang, R. (2005), The dislocation theory: a consistent way for including the
#     gravity effect in (visco)elastic plane-earth models, Geophysical Journal
#     International, 161, 191-196.
#
#################################################################
##                                                             ##
## Green's functions should have been prepared with the        ##
## program "psgrn08" before the program "pscmp08" is started.  ##
##                                                             ##
## For local Cartesian coordinate system, the Aki's convention ##
## is used, that is, x is northward, y is eastward, and z is   ##
## downward.                                                   ##
##                                                             ##
## If not specified otherwise, SI Unit System is used overall! ##
##                                                             ##
#################################################################
#===============================================================================
# OBSERVATION ARRAY
# =================
# 1. selection for irregular observation positions (= 0) or a 1D observation
#    profile (= 1) or a rectangular 2D observation array (= 2): iposrec
#
#    IF (iposrec = 0 for irregular observation positions) THEN
#
# 2. number of positions: nrec
#
# 3. coordinates of the observations: (lat(i),lon(i)), i=1,nrec
#
#    ELSE IF (iposrec = 1 for regular 1D observation array) THEN
#
# 2. number of position samples of the profile: nrec
#
# 3. the start and end positions: (lat1,lon1), (lat2,lon2)
#
#    ELSE IF (iposrec = 2 for rectanglular 2D observation array) THEN
#
# 2. number of x samples, start and end values: nxrec, xrec1, xrec2
#
# 3. number of y samples, start and end values: nyrec, yrec1, yrec2
#
#    sequence of the positions in output data: lat(1),lon(1); ...; lat(nx),lon(1);
#    lat(1),lon(2); ...; lat(nx),lon(2); ...; lat(1),lon(ny); ...; lat(nx),lon(ny).
#
#    Note that the total number of observation positions (nrec or nxrec*nyrec)
#    should be <= NRECMAX (see pecglob.h)!
#===============================================================================
 %(sw_observation_type)i
%(str_npoints)s
%(str_observation)s
#===============================================================================
# OUTPUTS
# =======
#
# 1. select output for los displacement (only for snapshots, see below), x, y,
#    and z-cosines to the INSAR orbit: insar (1/0 = yes/no), xlos, ylos, zlos
#
#    if this option is selected (insar = 1), the snapshots will include additional
#    data:
#    LOS_Dsp = los displacement to the given satellite orbit.
#
# 2. select output for Coulomb stress changes (only for snapshots, see below):
#    icmb (1/0 = yes/no), friction, Skempton ratio, strike, dip, and rake angles
#    [deg] describing the uniform regional master fault mechanism, the uniform
#    regional principal stresses: sigma1, sigma2 and sigma3 [Pa] in arbitrary
#    order (the orietation of the pre-stress field will be derived by assuming
#    that the master fault is optimally oriented according to Coulomb failure
#    criterion)
#
#    if this option is selected (icmb = 1), the snapshots will include additional
#    data:
#    CMB_Fix, Sig_Fix = Coulomb and normal stress changes on master fault;
#    CMB_Op1/2, Sig_Op1/2 = Coulomb and normal stress changes on the two optimally
#                       oriented faults;
#    Str_Op1/2, Dip_Op1/2, Slp_Op1/2 = strike, dip and rake angles of the two
#                       optimally oriented faults.
#
#    Note: the 1. optimally orieted fault is the one closest to the master fault.
#
# 3. output directory in char format: outdir
#
# 4. select outputs for displacement components (1/0 = yes/no): itout(i), i=1-3
#
# 5. the file names in char format for the x, y, and z components:
#    toutfile(i), i=1-3
#
# 6. select outputs for stress components (1/0 = yes/no): itout(i), i=4-9
#
# 7. the file names in char format for the xx, yy, zz, xy, yz, and zx components:
#    toutfile(i), i=4-9
#
# 8. select outputs for vertical NS and EW tilt components, block rotation, geoid
#    and gravity changes (1/0 = yes/no): itout(i), i=10-14
#
# 9. the file names in char format for the NS tilt (positive if borehole top
#    tilts to north), EW tilt (positive if borehole top tilts to east), block
#    rotation (clockwise positive), geoid and gravity changes: toutfile(i), i=10-14
#
#    Note that all above outputs are time series with the time window as same
#    as used for the Green's functions
#
#10. number of scenario outputs ("snapshots": spatial distribution of all above
#    observables at given time points; <= NSCENMAX (see pscglob.h): nsc
#
#11. the time [day], and file name (in char format) for the 1. snapshot;
#12. the time [day], and file name (in char format) for the 2. snapshot;
#13. ...
#
#    Note that all file or directory names should not be longer than 80
#    characters. Directories must be ended by / (unix) or \ (dos)!
#===============================================================================
 %(sw_los_displacement)i    %(str_los_vector)s
 %(sw_coulomb_stress)i    %(str_coulomb_master_field)s
 %(str_pscmp_outdir)s
 %(str_displ_sw_output_types)s
 %(str_displ_filenames)s
 %(str_stress_sw_output_types)s
 %(str_stress_filenames)s
 %(str_tilt_sw_output_types)s    %(str_gravity_sw_output_types)s
 %(str_tilt_filenames)s %(str_gravity_filenames)s
 %(n_snapshots)i
%(str_times_snapshots)s
#===============================================================================
#
# GREEN'S FUNCTION DATABASE
# =========================
# 1. directory where the Green's functions are stored: grndir
#
# 2. file names (without extensions!) for the 13 Green's functions:
#    3 displacement komponents (uz, ur, ut): green(i), i=1-3
#    6 stress components (szz, srr, stt, szr, srt, stz): green(i), i=4-9
#    radial and tangential components measured by a borehole tiltmeter,
#    rigid rotation around z-axis, geoid and gravity changes (tr, tt, rot, gd, gr):
#    green(i), i=10-14
#
#    Note that all file or directory names should not be longer than 80
#    characters. Directories must be ended by / (unix) or \ (dos)! The
#    extensions of the file names will be automatically considered. They
#    are ".ep", ".ss", ".ds" and ".cl" denoting the explosion (inflation)
#    strike-slip, the dip-slip and the compensated linear vector dipole
#    sources, respectively.
#
#===============================================================================
 %(str_psgrn_outdir)s
 %(str_displ_filenames)s
 %(str_stress_filenames)s
 %(str_tilt_filenames)s    %(str_gravity_filenames)s
#===============================================================================
# RECTANGULAR SUBFAULTS
# =====================
# 1. number of subfaults (<= NSMAX in pscglob.h): ns
#
# 2. parameters for the 1. rectangular subfault: geographic coordinates
#    (O_lat, O_lon) [deg] and O_depth [km] of the local reference point on
#    the present fault plane, length (along strike) [km] and width (along down
#    dip) [km], strike [deg], dip [deg], number of equi-size fault patches along
#    the strike (np_st) and along the dip (np_di) (total number of fault patches
#    = np_st x np_di), and the start time of the rupture; the following data
#    lines describe the slip distribution on the present sub-fault:
#
#    pos_s[km]  pos_d[km]  slip_strike[m]  slip_downdip[m]  opening[m]
#
#    where (pos_s,pos_d) defines the position of the center of each patch in
#    the local coordinate system with the origin at the reference point:
#    pos_s = distance along the length (positive in the strike direction)
#    pos_d = distance along the width (positive in the down-dip direction)
#
#
# 3. ... for the 2. subfault ...
# ...
#                   N
#                  /
#                 /| strike
#                +------------------------
#                |\        p .            \ W
#                :-\      i .              \ i
#                |  \    l .                \ d
#                :90 \  S .                  \ t
#                |-dip\  .                    \ h
#                :     \. | rake               \ 
#                Z      -------------------------
#                              L e n g t h
#
#    Simulation of a Mogi source:
#    (1) Calculate deformation caused by three small openning plates (each
#        causes a third part of the volume of the point inflation) located
#        at the same depth as the Mogi source but oriented orthogonal to
#        each other.
#    (2) Multiply the results by 3(1-nu)/(1+nu), where nu is the Poisson
#        ratio at the source depth.
#    The multiplication factor is the ratio of the seismic moment (energy) of
#    the Mogi source to that of the plate openning with the same volume change.
#===============================================================================
# n_faults
#-------------------------------------------------------------------------------
 %(n_patches)i
#-------------------------------------------------------------------------------
# n   O_lat   O_lon   O_depth length  width strike dip   np_st np_di start_time
# [-] [deg]   [deg]   [km]    [km]     [km] [deg]  [deg] [-]   [-]   [day]
#     pos_s   pos_d   slp_stk slp_ddip open
#     [km]    [km]    [m]     [m]      [m]
#-------------------------------------------------------------------------------
%(patches_str)s
#================================end of input===================================
'''  # noqa
        return template % d


class PsCmpError(gf.store.StoreError):
    pass


class Interrupted(gf.store.StoreError):
    def __str__(self):
        return 'Interrupted.'


class PsCmpRunner:

    def __init__(self, tmp=None, keep_tmp=False):
        self.tempdir = mkdtemp(prefix='pscmprun-', dir=tmp)
        self.keep_tmp = keep_tmp
        self.config = None

    def run(self, config):
        self.config = config

        input_fn = pjoin(self.tempdir, 'input')

        f = open(input_fn, 'w')
        input_str = config.string_for_config()

        logger.debug('===== begin pscmp input =====\n'
                     '%s===== end pscmp input =====' % input_str)

        f.write(input_str)
        f.close()
        program = program_bins['pscmp.%s' % config.pscmp_version]

        old_wd = os.getcwd()

        os.chdir(self.tempdir)

        interrupted = []

        def signal_handler(signum, frame):
            os.kill(proc.pid, signal.SIGTERM)
            interrupted.append(True)

        original = signal.signal(signal.SIGINT, signal_handler)
        try:
            try:
                proc = Popen(program, stdin=PIPE, stdout=PIPE, stderr=PIPE,
                             close_fds=True)

            except OSError as err:
                os.chdir(old_wd)
		logger.error('OS error: {0}'.format(err))
                raise PsCmpError('could not start pscmp: "%s"' % program)

            (output_str, error_str) = proc.communicate('input\n')

        finally:
            signal.signal(signal.SIGINT, original)

        if interrupted:
            raise KeyboardInterrupt()

        logger.debug('===== begin pscmp output =====\n'
                     '%s===== end pscmp output =====' % output_str)

        errmess = []
        if proc.returncode != 0:
            errmess.append(
                'pscmp had a non-zero exit state: %i' % proc.returncode)

        if error_str:
            errmess.append('pscmp emitted something via stderr')

        if output_str.lower().find('error') != -1:
            errmess.append("the string 'error' appeared in pscmp output")

        if errmess:
            self.keep_tmp = True

            os.chdir(old_wd)
            raise PsCmpError('''
===== begin pscmp input =====
%s===== end pscmp input =====
===== begin pscmp output =====
%s===== end pscmp output =====
===== begin pscmp error =====
%s===== end pscmp error =====
%s
pscmp has been invoked as "%s"
in the directory %s'''.lstrip() % (
                input_str, output_str, error_str, '\n'.join(errmess), program,
                self.tempdir))

        self.pscmp_output = output_str
        self.pscmp_error = error_str

        os.chdir(old_wd)

    def get_results(self, component='displ', which='snapshot', flip_z=False):
        '''
        Be careful: The z-component is downward positive!
        If flip_z=True it will be flipped upward! For displacements!
        '''
        if which == 'snapshot':
            assert self.config.times_snapshots is not None
            fns = self.config.get_output_filenames(self.tempdir)
        else:
            raise Exception(
                'get_results: which argument should be "snapshot"')

        output = []
        for fn in fns:
            if not os.path.exists(fn):
                continue

            data = num.loadtxt(fn, skiprows=1, dtype=num.float)
            nsamples, n_comp = data.shape

            if component == 'displ':
                if flip_z:
                    data[:,4] *= (-1)
                output.append(data[:,2:5])
            elif component == 'stress':
                output.append(data[:,5:11])
            elif component == 'tilt':
                output.append(data[:,11:14])
            elif component == 'gravity':
                output.append(data[:,14:16])
            else:
                raise Exception(
       'get_results: component argument should be "displ/stress/tilt/gravity"')

        return output

    def __del__(self):
        if self.tempdir:
            if not self.keep_tmp:
                shutil.rmtree(self.tempdir)
                self.tempdir = None
            else:
                logger.warn(
                    'not removing temporary directory: %s' % self.tempdir)


