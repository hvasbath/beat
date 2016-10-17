c     GLOBAL INDEX PARAMETERS FOR DEFINING ARRAYS
c     ===========================================
c     nzmax: max. interface index;
c     lmax: max. no of total homogeneous layers (lmax <= nzmax-2);
c     nrmax: max. no of traces;
c     nfmax: max. no of frequency samples.
c     nzsmax: max. number of source depths
c
      integer lmax,nzmax,nrmax,nfmin,nfmax,ntmax,nzsmax
      parameter(lmax=100)
      parameter(nzmax=lmax+3)
      parameter(nrmax=251)
      parameter(nfmin=64)
	  parameter(nfmax=1024)
      parameter(ntmax=2*nfmax)
      parameter(nzsmax=100)
c
c     INDEX PARAMETERS FOR BESSEL FUNCTION TABLES
c     ===========================================
c
      integer dnx,nxmax,nbsjmax
      parameter(dnx=512)
      parameter(nxmax=1024)
      parameter(nbsjmax=dnx*nxmax)
c
c     GLOBAL CONSTANTS
c     ================
c
      double precision km2m,day2sec,relaxmin,grfacmin
      parameter(km2m=1.0d+03,day2sec=8.64d+04)
      parameter(relaxmin=1.0d-06,grfacmin=1.0d-03)
c
c     GRAVITY, GRAVITATIONAL CONSTANT AND EARTH RADIUS
c     ================================================
c     gamma = 4*pi*G
c
      double precision g0,gamma,rearth
      parameter(g0=9.82d+00,gamma=8.38579d-10,rearth=6.371d+06)
      double precision denswater
      parameter(denswater=1.0d+03)
c
      integer nwarn
      common /iwarning/ nwarn
c
      double precision grfac
      logical nongravity
      common /dgreff/ grfac
      common /lgreff/ nongravity
c
c     DISCRETISATION ACCURACY FOR LAYERS WITH CONSTANT GRADIENT
c     =========================================================
c     reslm: for moduli
c     resld: for density
c     reslv: for viscosity
c
      double precision reslm,resld,reslv
      parameter(reslm=0.05d0,resld=0.05d0,reslv=0.250d0)
c
c     COMMON BLOCKS
c     =============
      integer lp,nno(nzmax)
      double precision zp(nzmax),hp(nzmax)
      common /isublayer/ lp,nno
      common /dsublayer/ zp,hp
c
c     zrec: receiver depth
c     lzrec: sublayer no of receiver
c
      integer lzrec,ioc
      double precision zrec,kgmax
      common /ireceiver/ lzrec,ioc
      common /dreceiver/ zrec,kgmax
c
c     original model parameters
c
      integer l0
      double precision z1(lmax),z2(lmax)
      double precision la1(lmax),la2(lmax),mu1(lmax),mu2(lmax)
      double precision rho1(lmax),rho2(lmax),etk1(lmax),etk2(lmax)
      double precision etm1(lmax),etm2(lmax),alf1(lmax),alf2(lmax)
      common /imodel0/ l0
      common /dmodel0/ z1,z2,la1,la2,mu1,mu2,rho1,rho2,etk1,etk2,
     &                 etm1,etm2,alf1,alf2
c       
c     model parameter:
c     n0: number of homogeneous layers
c
      integer n0
      double precision h(lmax),la(lmax),mu(lmax),rho(lmax)
      double precision etk(lmax),etm(lmax),alf(lmax)
      logical elastic(lmax)
      common /imodel/ n0
      common /lmodel/ elastic
      common /dmodel/ h,la,mu,rho,etk,etm,alf
c
      double complex cla(lmax),cmu(lmax)
      common /dcpara/ cla,cmu
c
c     source parameters
c
      integer ls,ms(4),ics(4)
      double precision zs
      double complex cics(4),cms(4)
      double complex sfct0(8,4),sfct1(8,4)
      common /isource/ ls,ms,ics
      common /dsource/ zs,sfct0,sfct1
      common /csource/ cics,cms
c
c     half-space source parameters
c
      double complex sfcths0(8,4),sfcths1(8,4)
      common /dsourcehs/ sfcths0,sfcths1
c
c     table of J_n(x), n = -1, 0, 1, 2, 3
c
      double precision dxbsj,bsj(0:nbsjmax,-1:3)
      common /dbessels/ dxbsj,bsj
c
c     output data
c
      double precision r(nrmax),rs(nrmax),geow(nrmax)
      double complex du(-1:nfmax,nrmax,14,4)
      logical select(14,4)
      common /doutdata/ r,rs,geow,du
      common /loutdata/ select

