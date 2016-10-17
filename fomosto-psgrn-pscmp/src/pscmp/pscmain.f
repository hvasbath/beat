      program pscmain
      implicit none
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     this program calculates viscoelastic deformation caused by a     c
c     number of rectanglar rupture planes using the Green's function   c
c     approach.                                                        c
c                                                                      c
c     The input data will be read from an input file                   c
c                                                                      c
c     Last modified: Potsdam, July, 2008, by R. Wang                   c
c                                                                      c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
c     BEGIN DECLARATIONS
c     ==================
c~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
c     GLOBAL CONSTANTS
c     ================
c~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      include 'pscglob.h'
c~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
c     LOCAL WORK SPACES
c     =================
c~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      integer i,j,is,isc,ntr,nsc,irec,ilatrec,ilonrec,nlatrec,nlonrec
      integer nrec,ns,iptch,ieq,neq
      double precision latrec1,latrec2,lonrec1,lonrec2,dlatrec,dlonrec
      double precision sx,sy,sz
      double complex clonrec1,clonrec2
      double complex clonrec(NRECMAX)
      character*80 infile
      character*180 dataline
      logical onlysc,neweq
c~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
c     END DECLARATIONS
c     ================
c~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
c00000000000000000000000000000000000000000000000000000000000000000000000
c     BEGIN READ IN INPUT PARAMETERS
c     ==============================
c00000000000000000000000000000000000000000000000000000000000000000000000
c
      nwarn=0
c
      print *,'######################################################'
      print *,'#                                                    #'
      print *,'#                  Welcome to                        #'
      print *,'#                                                    #'
      print *,'#                                                    #'
      print *,'#     PPPP     SSSS    CCCC   M   M    PPPP          #'
      print *,'#     P   P   S       C       MM MM    P   P         #'
      print *,'#     PPPP     SSS    C       M M M    PPPP          #'
      print *,'#     P           S   C       M   M    P             #'
      print *,'#     P       SSSS     CCCC   M   M    P             #'
      print *,'#                                                    #'
      print *,'#                  Version 2008                      #'
      print *,'#                                                    #'
      print *,'#                      by                            #'
      print *,'#                                                    #'
      print *,'#                 Rongjiang Wang                     #'
      print *,'#              (wang@gfz-potsdam.de)                 #'
      print *,'#                                                    #'
      print *,'#           GeoForschungsZentrum Potsdam             #'
      print *,'#                     July, 2008                     #'
      print *,'######################################################'
      print *,'                                                      '
      write(*,'(a,$)')' Please type the file name of input data: '
      read(*,'(a)')infile
      open(10,file=infile,status='old')
c00000000000000000000000000000000000000000000000000000000000000000000000
c     READ IN PARAMETERS FOR OBSERVATION ARRAY
c     ========================================
c00000000000000000000000000000000000000000000000000000000000000000000000
      call getdata(10,dataline)
      read(dataline,*)ilonrec
      if(ilonrec.eq.0)then
c
c       irregular observation positions
c
        call getdata(10,dataline)
        read(dataline,*)nrec
        if(nrec.lt.1)then
          stop ' Error: wrong input for nrec!'
        else if(nrec.gt.NRECMAX)then
          stop ' Error: NRECMAX defined too small!'
        endif
        read(10,*)(clonrec(irec),irec=1,nrec)
        do irec=1,nrec
          clonrec(irec)=clonrec(irec)
          latrec(irec)=dreal(clonrec(irec))
          lonrec(irec)=dimag(clonrec(irec))
        enddo
      else if(ilonrec.eq.1)then
c
c        1D observation profile
c
        call getdata(10,dataline)
        read(dataline,*)nrec
        call getdata(10,dataline)
        read(dataline,*)clonrec1,clonrec2
        if(nrec.lt.1)then
          stop ' Error: wrong input for nrec!'
        else if(nrec.gt.NRECMAX)then
          stop ' Error: NRECMAX defined too small!'
        endif
        latrec(1)=dreal(clonrec1)
        lonrec(1)=dimag(clonrec1)
        if(nrec.gt.1)then
          dlatrec=dreal(clonrec2-clonrec1)/dble(nrec-1)
          dlonrec=dimag(clonrec2-clonrec1)/dble(nrec-1)
        else
          dlatrec=0.d0
          dlonrec=0.d0
        endif
        do irec=1,nrec
          latrec(irec)=dreal(clonrec1)+dlatrec*dble(irec-1)
          lonrec(irec)=dimag(clonrec1)+dlonrec*dble(irec-1)
        enddo
      else if(ilonrec.eq.2)then
c
c        2D rectanglar observation array
c
        call getdata(10,dataline)
        read(dataline,*)nlatrec,latrec1,latrec2
        call getdata(10,dataline)
        read(dataline,*)nlonrec,lonrec1,lonrec2
        nrec=nlatrec*nlonrec
        if(nrec.lt.1)then
          stop ' Error: wrong input for nrec!'
        else if(nrec.gt.NRECMAX)then
          stop ' Error: NRECMAX defined too small!'
        endif
        if(nlatrec.gt.1)then
          dlatrec=(latrec2-latrec1)/dble(nlatrec-1)
        else
          dlatrec=0.d0
        endif
        if(nlonrec.gt.1)then
          dlonrec=(lonrec2-lonrec1)/dble(nlonrec-1)
        else
          dlonrec=0.d0
        endif
        irec=0
        do ilonrec=1,nlonrec
          do ilatrec=1,nlatrec
            irec=irec+1
            latrec(irec)=latrec1+dlatrec*dble(ilatrec-1)
            lonrec(irec)=lonrec1+dlonrec*dble(ilonrec-1)
          enddo
        enddo
      else
        stop' Error: wrong input for ilonrec!'
      endif
c00000000000000000000000000000000000000000000000000000000000000000000000
c      READ IN OUTPUT PARAMETERS
c      =========================
c00000000000000000000000000000000000000000000000000000000000000000000000
      call getdata(10,dataline)
      read(dataline,*)i
      if(i.eq.1)read(dataline,*)insar,xlos,ylos,zlos
      call getdata(10,dataline)
      read(dataline,*)i
      if(i.eq.1)read(dataline,*)icmb,friction,skempton,
     &                          strike0,dip0,rake0,(sigma0(j),j=1,3)
      call getdata(10,dataline)
      read(dataline,*)outdir
c
      call getdata(10,dataline)
      read(dataline,*)(itout(i),i=1,3)
      call getdata(10,dataline)
      read(dataline,*)(toutfile(i),i=1,3)
      call getdata(10,dataline)
      read(dataline,*)(itout(i),i=4,9)
      call getdata(10,dataline)
      read(dataline,*)(toutfile(i),i=4,9)
      call getdata(10,dataline)
      read(dataline,*)(itout(i),i=10,14)
      call getdata(10,dataline)
      read(dataline,*)(toutfile(i),i=10,14)
      call getdata(10,dataline)
      read(dataline,*)nsc
      if(nsc.gt.NSCMAX)then
        stop ' Error: NSCMAX defined too small!'
      endif
      do isc=1,nsc
        call getdata(10,dataline)
        read(dataline,*)tsc(isc),scoutfile(isc)
        if(tsc(isc).lt.0.d0)then
          stop ' Error: wrong scenario time!'
        endif
        tsc(isc)=tsc(isc)*DAY2SEC
      enddo
      onlysc=.true.
      do i=1,14
        onlysc=onlysc.and.itout(i).ne.1
      enddo
      if(onlysc.and.nsc.le.0)then
        stop ' No outputs have been selected!'
      endif
c00000000000000000000000000000000000000000000000000000000000000000000000
c     READ IN PARAMETERS FOR EARTH MODEL CHOICE
c     =========================================
c00000000000000000000000000000000000000000000000000000000000000000000000
      call getdata(10,dataline)
      read(dataline,*)grndir
c
      call getdata(10,dataline)
      read(dataline,*)(green(i),i=1,3)
      call getdata(10,dataline)
      read(dataline,*)(green(i),i=4,9)
      call getdata(10,dataline)
      read(dataline,*)(green(i),i=10,14)
c00000000000000000000000000000000000000000000000000000000000000000000000
c     READ IN PARAMETERS FOR RECTANGULAR SOURCES
c     ==========================================
c00000000000000000000000000000000000000000000000000000000000000000000000
      call getdata(10,dataline)
      read(dataline,*)ns
      if(ns.lt.1)then
        stop ' Error: wrong number of subfaults!'
      endif
      if(ns.gt.NSMAX)then
        stop ' Error: too large no of source rectangles!'
      endif
      neq=0
      do is=1,ns
        call getdata(10,dataline)
        read(dataline,*)i,latref(is),lonref(is),zref(is),
     &      length(is),width(is),strike(is),dip(is),
     &      nptch_s(is),nptch_d(is),tstart(is)
        if(nptch_s(is)*nptch_d(is).gt.NPTCHMAX)then
          write(*,'(a,i3,a)')' Error: too large no of of patches at ',
     &                       is,'. segment!'
          stop
        endif
        zref(is)=zref(is)*KM2M
        length(is)=length(is)*KM2M
        width(is)=width(is)*KM2M
        tstart(is)=tstart(is)*DAY2SEC
        do iptch=1,nptch_s(is)*nptch_d(is)
          call getdata(10,dataline)
          read(dataline,*)ptch_s(is,iptch),ptch_d(is,iptch),sx,sy,sz
          ptch_s(is,iptch)=ptch_s(is,iptch)*KM2M
          ptch_d(is,iptch)=ptch_d(is,iptch)*KM2M
          slip_s(is,iptch)=sx
          slip_d(is,iptch)=sy
          opening(is,iptch)=sz
        enddo
        neweq=.true.
        do ieq=1,neq
          if(tstart(is).eq.eqtime(ieq))then
            neweq=.false.
            ieqno(is)=ieq
          endif
        enddo
        if(neweq)then
          neq=neq+1
          if(neq.gt.neqmax)then
            stop ' Error: NEQMAX defined too small!'
          endif
          ieqno(is)=neq
          eqtime(neq)=tstart(is)
        endif
        if(zref(is).lt.0.d0)then
          stop ' Error: source depth zs0 < 0!'
        endif
        if(dabs(strike(is)).gt.360.d0)then
          stop ' Error: wrong strike angle!'
        endif
        if(strike(is).lt.0.d0)then
          strike(is)=strike(is)+360.d0
        endif
        if(dip(is).gt.90.d0.or.dip(is).lt.0.d0)then
          stop ' Error: wrong dip angle!'
        endif
      enddo
c
      close(10)
c00000000000000000000000000000000000000000000000000000000000000000000000
c      END READ IN INPUT PARAMETERS
c      ============================
c00000000000000000000000000000000000000000000000000000000000000000000000
      print *,'... input data successful ...'
c00000000000000000000000000000000000000000000000000000000000000000000000
c      BEGIN PROCESSING
c      ================
c00000000000000000000000000000000000000000000000000000000000000000000000
      nwarn=0
      print *,'... using the Green function approach ...'
      call pscgrn(neq,ns,nrec,ntr,onlysc,nsc)
      print *,'... outputs ...'
      call pscout(neq,nrec,ntr,onlysc,nsc)
c00000000000000000000000000000000000000000000000000000000000000000000000
c      END OF STANDARD PROCESSING
c      ==========================
c00000000000000000000000000000000000000000000000000000000000000000000000
      if(nwarn.eq.0)then
        print *,'####################################################'
        print *,'#                                                  #'
        print *,'#          End of computations with PSCMP          #'
        print *,'#                                                  #'
        print *,'####################################################'
      else
        print *,'####################################################'
        print *,'     Sorry, there have been',nwarn,' warnings.      '
        print *,'             Results may be inaccurate!             '
        print *,'####################################################'
      endif
c
      stop
      end
