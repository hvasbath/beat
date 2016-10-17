      program psgmain
      implicit none
c
      include 'psgglob.h'
c
c     work space
c
      integer i,j,l,ir,nr,it,nt,izs,nls,ierr
      integer istp,isp,nr1,nr2,nzs,nlr,nprf
      integer lend,lenf,leninp,iunit,runtime
      integer unit(14,4),idec(nrmax),nout(nrmax)
      double precision am,r1,r2,dr,dract,sampratio,rsmin
      double precision twindow,dt,kg,dratio,accuracy
      double precision zs1,zs2,dzs,zrs2,swap,vp,vs
      character*35 stype(4)
      character*35 comptxt(14)
      character*80 inputfile,fname(14),outdir
      character*163 green(14,4)
      character*180 dataline
      integer time
c
c     read input file file
c
      print *,'######################################################'
      print *,'#                                                    #'
      print *,'#                  Welcome to                        #'
      print *,'#                                                    #'
      print *,'#                                                    #'
      print *,'#     PPPP     SSSS    GGGG   RRRR    N   N          #'
      print *,'#     P   P   S       G       R   R   NN  N          #'
      print *,'#     PPPP     SSS    G GGG   RRRR    N N N          #'
      print *,'#     P           S   G   G   R R     N  NN          #'
      print *,'#     P       SSSS     GGGG   R  R    N   N          #'
      print *,'#                                                    #'
      print *,'#                  Version 2008                      #'
      print *,'#                                                    #'
      print *,'#                      by                            #'
      print *,'#                 Rongjiang Wang                     #'
      print *,'#              (wang@gfz-potsdam.de)                 #'
      print *,'#                                                    #'
      print *,'#           GeoForschungsZentrum Potsdam             #'
      print *,'#                    July 2008                       #'
      print *,'######################################################'
      print *,'                                                      '
c
      nwarn=0
      write(*,'(a,$)')' Please type the file name of input data: '
      read(*,'(a)')inputfile
      runtime=time()
      open(10,file=inputfile,status='old')
c
c     parameters for source-observation array
c     =======================================
c
      call getdata(10,dataline)
      read(dataline,*)zrec,ioc
      zrec=zrec*km2m
      call getdata(10,dataline)
      read(dataline,*)nr,r1,r2,sampratio
      if(sampratio.lt.1.d0)then
        stop 'Error: max. to min. sampling ratio < 1!'
      endif
      r1=r1*km2m
      r2=r2*km2m
      if(r1.gt.r2)then
        swap=r1
        r1=r2
        r2=swap
      endif
      if(r1.lt.0.d0.or.r2.lt.0.d0.or.nr.lt.1)then
        stop 'Error: wrong no of distance samples!'
      else if(nr.gt.nrmax)then
        stop 'Error: max. no of distance samples exceeded!'
      else if(nr.eq.1.or.r1.eq.r2)then
        r2=r1
        nr=1
        dr=0.d0
        r(1)=r1
      else if(nr.eq.2)then
        dr=r2-r1
        r(1)=r1
        r(2)=r2
      else
        dr=2.d0*(r2-r1)/dble(nr-1)/(1.d0+sampratio)
        r(1)=r1
        do i=2,nr
          dract=dr*(1.d0+(sampratio-1.d0)*dble(i-2)/dble(nr-2))
          r(i)=r(i-1)+dract
        enddo
      endif
c
      call getdata(10,dataline)
      read(dataline,*)nzs,zs1,zs2
      if(zs1.gt.zs2)then
        swap=zs1
        zs1=zs2
        zs2=swap
      endif
      zs1=zs1*km2m
      zs2=zs2*km2m
      if(zs1.lt.0.d0.or.zs2.lt.0.d0.or.nzs.lt.1)then
        stop 'Error: wrong no of source depths!'
      else if(nzs.gt.nzsmax)then
        stop 'Error: max. no of source depths exceeded!'
      else if(nzs.eq.1.or.zs1.eq.zs2)then
        nzs=1
        dzs=0.d0
      else
        dzs=(zs2-zs1)/dble(nzs-1)
        if(zs1.lt.zrec+0.5d0*dzs.and.zs2.gt.zrec-0.5d0*dzs)then
          zs=zrec+0.5d0*dzs
10        zs=zs-dzs
          if(zs.gt.zs1)goto 10
          if(zs.gt.0.d0)then
            zs1=zs
          else
            zs1=zs+dzs
          endif
        endif
        zs2=zs1+dble(nzs-1)*dzs
      endif
c
      call getdata(10,dataline)
      read(dataline,*)nt,twindow
      if(twindow.le.0.d0)then
        stop ' Error in input: wrong time window!'
      else if(nt.le.0)then
        stop ' Error in input: time sampling no <= 0!'
      endif
      twindow=twindow*day2sec
      if(nt.le.2)then
        dt=twindow
      else
        dt=twindow/dble(nt-1)
      endif
c
c     wavenumber integration parameters
c     =================================
c
      call getdata(10,dataline)
      read(dataline,*)accuracy
      if(accuracy.le.0.d0.or.accuracy.ge.1.d0)accuracy=0.1d0
c
      call getdata(10,dataline)
      read(dataline,*)grfac
      if(grfac.le.grfacmin)then
        grfac=0.d0
      endif
c
c     parameters for output files
c     ===========================
c
      call getdata(10,dataline)
      read(dataline,*)outdir
c
      do lend=80,1,-1
        if(outdir(lend:lend).ne.' ')goto 100
      enddo
100   continue
c
      if(lend.lt.1)then
        stop 'Error: wrong format for output directory!'
      endif
c
      call getdata(10,dataline)
      read(dataline,*)(fname(i),i=1,3)
      call getdata(10,dataline)
      read(dataline,*)(fname(i),i=4,9)
      call getdata(10,dataline)
      read(dataline,*)(fname(i),i=10,14)
      do i=1,14
        do lenf=80,1,-1
          if(fname(i)(lenf:lenf).ne.' ')goto 110
        enddo
110     continue
        green(i,1)=outdir(1:lend)//fname(i)(1:lenf)//'.ep'
        green(i,2)=outdir(1:lend)//fname(i)(1:lenf)//'.ss'
        green(i,3)=outdir(1:lend)//fname(i)(1:lenf)//'.ds'
        green(i,4)=outdir(1:lend)//fname(i)(1:lenf)//'.cl'
        do istp=1,4
          select(i,istp)=.true.
        enddo
      enddo
c
c     no tangential components for clvd sources
c
      select(3,1)=.false.
      select(8,1)=.false.
      select(9,1)=.false.
      select(11,1)=.false.
      select(12,1)=.false.
      select(3,4)=.false.
      select(8,4)=.false.
      select(9,4)=.false.
      select(11,4)=.false.
      select(12,4)=.false.
c
c     global model parameters
c     =======================
c
      call getdata(10,dataline)
      read(dataline,*)l
      if(l.gt.lmax)then
        stop 'the max. no of layers (lmax) too small defined!'
      endif
c
c      multilayered model parameters
c      =============================
c
      kgmax=0.d0
      do i=1,l
        call getdata(10,dataline)
        read(dataline,*)j,h(i),vp,vs,rho(i),etk(i),etm(i),alf(i)
        if(alf(i).gt.1.d0.or.alf(i).le.0.d0)then
          stop 'Error in psgmain: wrong value for parameter alpha!'
        endif
        h(i)=h(i)*km2m
        vp=vp*km2m
        vs=vs*km2m
        mu(i)=rho(i)*vs*vs
        la(i)=rho(i)*vp*vp-2.d0*mu(i)
        if(la(i).le.0.d0)then
          stop 'inconsistent Vp/Vs ratio!'
        endif
        if(etk(i).le.0.d0.or.alf(i).eq.1.d0)then
          etk(i)=0.d0
          alf(i)=1.d0
        endif
        if(etm(i).lt.0.d0)then
          etm(i)=0.d0
        endif
        kg=rho(i)*g0/(la(i)+2.d0*mu(i)/3.d0)
        kgmax=dmax1(kgmax,kg)
      enddo
      if(l.eq.1)h(l)=0.d0
c
c     end of inputs
c     =============
c
      close(10)
c
      comptxt(1)='Uz (vertical displacement)'
      comptxt(2)='Ur (radial displacement)'
      comptxt(3)='Ut (tangential displacement)'
      comptxt(4)='Szz (linear stress)'
      comptxt(5)='Srr (linear stress)'
      comptxt(6)='Stt (linear stress)'
      comptxt(7)='Szr (shear stress)'
      comptxt(8)='Srt (shear stress)'
      comptxt(9)='Stz (shear stress)'
      comptxt(10)='Tr (tilt -dUr/dz)'
      comptxt(11)='Tt (tilt -dUt/dz)'
      comptxt(12)='Rot (rotation ar. z-axis)'
      comptxt(13)='Gd (geoid changes)'
      comptxt(14)='Gr (gravity changes)'
c
c     determine upper und lower parameter values of each layer
c
      l0=1
      z1(l0)=0.d0
      do i=2,l
        if(h(i).gt.h(i-1))then
          z1(l0)=h(i-1)
          la1(l0)=la(i-1)
          mu1(l0)=mu(i-1)
          rho1(l0)=rho(i-1)
          etk1(l0)=etk(i-1)
          etm1(l0)=etm(i-1)
          alf1(l0)=alf(i-1)
c
          z2(l0)=h(i)
          la2(l0)=la(i)
          mu2(l0)=mu(i)
          rho2(l0)=rho(i)
          etk2(l0)=etk(i)
          etm2(l0)=etm(i)
          alf2(l0)=alf(i)
          l0=l0+1
        else
          z1(l0)=h(i)
          la1(l0)=la(i)
          mu1(l0)=mu(i)
          rho1(l0)=rho(i)
          etk1(l0)=etk(i)
          etm1(l0)=etm(i)
          alf1(l0)=alf(i)
        endif
      enddo
      z1(l0)=h(l)
      la1(l0)=la(l)
      mu1(l0)=mu(l)
      rho1(l0)=rho(l)
      etk1(l0)=etk(l)
      etm1(l0)=etm(l)
      alf1(l0)=alf(l)
c
c     construction of sublayers
c
      write(*,*)'the multi-layered poroelastic model:'
c
      call psgsublay(ierr)
      if(ierr.eq.1)then
        stop 'the max. no of layers (lmax) too small defined!'
      endif
c
      zs=0.d0
      call psglayer(ierr)
      nlr=nno(lzrec)
c
      leninp=index(inputfile,' ')-1
c
      stype(1)='explosion (M11=M22=M33=1*kappa)'
      stype(2)='strike-slip (M12=M21=1*mue)'
      stype(3)='dip-slip (M13=M31=1*mue)'
      stype(4)='clvd (M33=1*mue, M11=M22=-M33/2)'
c
      iunit=10
      do istp=1,4
        do i=1,14
          if(select(i,istp))then
            iunit=iunit+1
            unit(i,istp)=iunit
            open(unit(i,istp),file=green(i,istp),status='unknown')
            write(unit(i,istp),'(a)')'################################'
            write(unit(i,istp),'(a)')'# The input file used: '
     &                        //inputfile(1:leninp)
            write(unit(i,istp),'(a)')'################################'
            write(unit(i,istp),'(a)')'# Greens function component: '
     &                        //comptxt(i)
            write(unit(i,istp),'(a)')'#(Okada solutions subtracted)'
            write(unit(i,istp),'(a)')'# Source type: '//stype(istp)
            write(unit(i,istp),'(a)')'# Observation distance sampling:'
            write(unit(i,istp),'(a)')'#    nr        r1[m]        r2[m]'
     &                             //'  samp_ratio'
            write(unit(i,istp),'(i7,2E14.6,f10.4)')nr,r1,r2,sampratio
            write(unit(i,istp),'(a)')'# Uniform obs. site parameters:'
            write(unit(i,istp),'(a)')'#    depth[m]       la[Pa]       '
     &       //'mu[Pa]  rho[kg/m^3]    etk[Pa*s]    etm[Pa*s]     alpha'
            write(unit(i,istp),'(7E13.6)')zrec,la(nlr),
     &           mu(nlr),rho(nlr),etk(nlr),etm(nlr),alf(nlr)
            write(unit(i,istp),'(a)')'# Source depth sampling:'
            write(unit(i,istp),'(a)')'#   nzs       zs1[m]       zs2[m]'
            write(unit(i,istp),'(i7,2d14.6)')nzs,zs1,zs2
            write(unit(i,istp),'(a)')'# Time sampling:'
            write(unit(i,istp),'(a)')'#    nt        t-window[s]'
            write(unit(i,istp),'(i7,E24.16)')nt,twindow
            write(unit(i,istp),'(a)')'# Data in each source depth block'
            write(unit(i,istp),'(a)')'# ==============================='
            write(unit(i,istp),'(a)')'# Line 1: source layer parameters'
            write(unit(i,istp),'(a)')'#  s_depth, la, mu, rho,'
     &                              //' etk, etm, alpha'
            write(unit(i,istp),'(a)')'# Line 2: coseismic responses '
     &                             //'(f(ir,it=1),ir=1,nr)'
            write(unit(i,istp),'(a)')'# Line 3: (idec(ir),ir=1,nr)'
     &                //'(decimal exponents for postseismic responses)'
            write(unit(i,istp),'(a)')'# Line 4: (f(ir,it=2),ir=1,nr)'
            write(unit(i,istp),'(a)')'# Line 5: (f(ir,it=3),ir=1,nr)'
            write(unit(i,istp),'(a)')'#  ...'
          endif
        enddo
      enddo
c
      call psgbsj(ierr)
      do izs=1,nzs
        zs=zs1+dble(izs-1)*dzs
        write(*,'(/,a,E13.4,a)')' Processing for the '
     &                 //'source at depth:',zs,' m.'
c
        call psglayer(ierr)
        do l=1,lp
          zp(l)=0.d0
          do i=1,l-1
            if(nno(i).eq.nno(l))zp(l)=zp(l)+hp(i)
          enddo
        enddo
        nls=nno(ls)
c
        do istp=1,4
          do i=1,14
            do ir=1,nr
              do it=1,nfmax
                du(it,ir,i,istp)=(0.d0,0.d0)
              enddo
            enddo
          enddo
        enddo
c
        zrs2=(zrec-zs)**2
        rsmin=0.5d0*dr
        do ir=1,nr
          rs(ir)=dmax1(rsmin,0.1d0*dsqrt(zrs2+r(ir)**2))
          geow(ir)=zrs2+(rs(ir)+r(ir))**2
        enddo
c
        swap=dsqrt(zrs2+(rs(nr)+r(nr))**2)/dsqrt(zrs2+(rs(1)+r(1))**2)
        nprf=1+idnint(dlog(swap)/dlog(2.5d0))
        if(nprf.gt.1)then
          dratio=swap**(1.d0/dble(nprf-1))
        else
          dratio=2.5d0
        endif
c
        isp=0
        nr2=0
200     isp=isp+1
        nr1=nr2+1
        nr2=nr1
        do ir=nr1+1,nr
          if(r(ir).le.dratio*dsqrt(zrs2+(rs(nr1)+r(nr1))**2))nr2=ir
        enddo
        call psgspec(isp,nr1,nr2,nt,dt,accuracy)
        if(nr2.lt.nr)goto 200
c
        do istp=1,4
          do i=1,14
            if(.not.select(i,istp))goto 400
            write(unit(i,istp),'(a)')'#################################'
            write(unit(i,istp),'(a,i2,a)')'# the ',izs,'. source depth:'
            write(unit(i,istp),'(a)')'#################################'
            write(unit(i,istp),'(7E13.6)')zs,la(nls),
     &        mu(nls),rho(nls),etk(nls),etm(nls),alf(nls)
            do ir=1,nr-1
              write(unit(i,istp),'(E14.6,$)')dreal(du(1,ir,i,istp))
            enddo
            write(unit(i,istp),'(E14.6)')dreal(du(1,nr,i,istp))
            do ir=1,nr
              du(1,ir,i,istp)=dcmplx(0.d0,dimag(du(1,ir,i,istp)))
              am=0.d0
              do it=1,(nt+1)/2
                am=dmax1(am,dabs(dreal(du(it,ir,i,istp))),
     &                      dabs(dimag(du(it,ir,i,istp))))
              enddo
              if(am.le.0.d0)then
                idec(ir)=0
              else
                idec(ir)=idint(dlog10(am))-4
                do it=1,(nt+1)/2
                  du(it,ir,i,istp)=du(it,ir,i,istp)
     &                  *dcmplx(10.d0**(-idec(ir)),0.d0)
                enddo
              endif
            enddo
            call outint(unit(i,istp),idec,nr)
            do it=1,(nt+1)/2
              if(it.gt.1)then
                do ir=1,nr
                  nout(ir)=idnint(dreal(du(it,ir,i,istp)))
                enddo
                call outint(unit(i,istp),nout,nr)
              endif
              if(it*2.le.nt)then
                do ir=1,nr
                  nout(ir)=idnint(dimag(du(it,ir,i,istp)))
                enddo
                call outint(unit(i,istp),nout,nr)
              endif
            enddo
400         continue
          enddo
        enddo
      enddo
c
c     end of izs loop
c
      do istp=1,4
        do i=1,14
          if(select(i,istp))close(unit(i,istp))
        enddo
      enddo
      runtime=time()-runtime
      if(nwarn.eq.0)then
        write(*,'(a)')'################################################'
        write(*,'(a)')'#                                              #'
        write(*,'(a)')'#        End of computations with PSGRN        #'
        write(*,'(a)')'#                                              #'
        write(*,'(a,i10,a)')'#        Run time: ',runtime,
     &                                             ' sec              #'
        write(*,'(a)')'################################################'
      else
        write(*,'(a)')'################################################'
        write(*,'(a,i10,a)')'#        Run time: ',runtime,
     &                                             ' sec              #'
        write(*,'(a)')'   Sorry, there have been',nwarn,' warnings.    '
        write(*,'(a)')'           Results may be inaccurate!           '
        write(*,'(a)')'################################################'
      endif
      stop
      end
