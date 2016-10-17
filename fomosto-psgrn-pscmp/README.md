# PSGRN and PSCMP (packaged as fomosto backend)

Code to calculate synthetic stress/strain/tilt/gravitational fields on a
layered viscoelastic halfspace.

PSGRN and PSCMP have been written by Rongjiang Wang.

Packaging has been done by Hannes Vasyura-Bathke.

## References

- Wang, R., F. Lorenzo-Martín and F. Roth (2003), Computation of deformation
  induced by earthquakes in a multi-layered elastic crust - FORTRAN programs
  EDGRN/EDCMP, Computer and Geosciences, 29(2), 195-207.
- Wang, R., F. Lorenzo-Martin and F. Roth (2006), PSGRN/PSCMP - a new code for
  calculating co- and post-seismic deformation, geoid and gravity changes
  based on the viscoelastic-gravitational dislocation theory, Computers and
  Geosciences, 32, 527-541. DOI:10.1016/j.cageo.2005.08.006.
- Wang, R. (2005), The dislocation theory: a consistent way for including the
  gravity effect in (visco)elastic plane-earth models, Geophysical Journal
  International, 161, 191-196.

# Compile and install PSGRN and PSCMP
```
autoreconf -i   # only if 'configure' script is missing
F77=gfortran FFLAGS=-mcmodel=medium ./configure
make
sudo make install
```
