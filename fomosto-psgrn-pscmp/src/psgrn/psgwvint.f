      subroutine psgwvint(istate,lf,nr1,nr2,cs,
     &                    nkc,nkmax,dk0,accuracy,tty)
      implicit none
c
      integer istate,lf,nr1,nr2,nkc,nkmax
      double precision dk0,accuracy
      double complex cs
      logical tty
c
      include 'psgglob.h'
c
c     u: 1=uz, 2=ur, 3=ut,
c        4=ezz, 5=err, 6=ett, 7=ezr, 8=ert, 9=etz, 10=-dur/dz
c        11=-dut/dz, 12=rot(u)_z/2, 13=gd, 14=gr
c     NOTE: uz, ur, ezz, ezr, err, ett, dur/dz, gd, gr
c           have the same azimuth-factor as the poloidal mode (p-sv);
c           ut, ert, etz, dut/dz, rot(u)_z/2
c           have the same azimuth-factor as the
c           toroidal mode (sh);
c
      integer i,istp,ir,ik,nx,nkg,nnk,nlr
      double precision k,dk,kg,kgr,dkg,krt
      double precision pi,pi2,x,wl,wr,fac,y0abs,dyabs
      double precision bsj0(-1:3),y0max(8,4)
      double complex c0,c1,c2,c3,c4,c05,ck
      double complex clazr,cmuzr,clahs,cmuhs,evs
      double complex cbsj(3)
      double complex y0(8,4),cy(14,4)
      double complex urdr(4),utdr(4),urdr0(4)
      double complex obs(nrmax,16,4),obs0(nrmax,16,4)
      logical finish,tests
c
      integer iret,ipot
      double precision xokada,yokada,dip
      double complex alpha,pot1,pot2,pot3,pot4,cs45,ss45,potfac
      double complex ux,uy,uz,uxx,uyx,uzx,uxy,uyy,uzy,uxz,uyz,uzz
      double complex uza,ura,uta,esfa,urza,utza,etta,erta
      double complex szza,srra,stta,szra,srta,szta
c
      double precision eps
      data eps/1.0d-03/
c
      pi=4.d0*datan(1.d0)
      pi2=2.d0*pi
c
      c0=(0.d0,0.d0)
      c1=(1.d0,0.d0)
      c2=(2.d0,0.d0)
      c3=(3.d0,0.d0)
      c4=(4.d0,0.d0)
      c05=(0.5d0,0.d0)
      cs45=dcmplx(1.d0/dsqrt(2.d0),0.d0)
      ss45=dcmplx(1.d0/dsqrt(2.d0),0.d0)
c
      do istp=1,4
        urdr(istp)=c0
        utdr(istp)=c0
        urdr0(istp)=c0
        do i=1,15
          do ir=nr1,nr2
            obs(ir,i,istp)=c0
            obs0(ir,i,istp)=c0
          enddo
        enddo
      enddo
c
      call psgmoduli(cs,istate)
      call psgsource(1.d0)
      nlr=nno(lzrec)
      clazr=cla(nlr)
      cmuzr=cmu(nlr)
      call psghssrce(1.d0,clazr,cmuzr)
c
c     integration for contributions from extreme small wavenumber
c
      if(grfac.gt.0.d0)then
        dkg=dmin1(dk0,0.01d0*dmax1(dk0,kgmax*grfac))
        nkg=idint(dmin1(10.d0*kgmax*grfac,dk0*dble(nkc))/dkg)
        kg=dble(nkg)*dkg
        tests=.false.
        if(tests.and.tty)then
          open(10,file='kernel.dat',status='unknown')
          k=dkg
          do ik=1,1000
            call psgkern(y0,k,clazr,cmuzr)
            write(10,'(E15.7,64E12.4)')k/(kgmax*grfac),
     &       ((k*y0(i,istp),i=1,8),istp=1,4)
            k=1.01d0*k
          enddo
          close(10)
          pause
        endif
      else
        dkg=0.d0
        nkg=0
        kg=0.d0
      endif
c
      nnk=1
c
      do ik=1,nkmax+nkg
        if(ik.lt.nkg)then
          dk=dkg
          k=dble(ik)*dkg
        else if(ik.eq.nkg)then
          dk=0.5d0*(dkg+dk0)
          k=kg+0.25d0*(dk0-dkg)
        else
          dk=dk0
          k=kg+dble(ik-nkg)*dk0
        endif
        ck=dcmplx(k,0.d0)
        call psgkern(y0,k,clazr,cmuzr)
        do istp=1,4
c
c         for displacement components
c
          cy(1,istp)=y0(1,istp)
          cy(2,istp)=c05*(y0(3,istp)+cics(istp)*y0(5,istp))
          cy(3,istp)=c05*(y0(3,istp)-cics(istp)*y0(5,istp))
c
c         for strain components
c
          cy(4,istp)=y0(2,istp)
          cy(5,istp)=ck*y0(3,istp)
          cy(6,istp)=c05*(y0(4,istp)+cics(istp)*y0(6,istp))
          cy(7,istp)=c05*(y0(4,istp)-cics(istp)*y0(6,istp))
          cy(8,istp)=ck*y0(5,istp)
          cy(9,istp)=ck*y0(1,istp)
c
c         for tilt components
c
          cy(10,istp)=ck*y0(1,istp)
c
c         for potential
c
          cy(11,istp)=y0(7,istp)
c
c         for gravity components
c
          cy(12,istp)=y0(8,istp)
          cy(13,istp)=ck*y0(7,istp)
          cy(14,istp)=ck*y0(7,istp)
        enddo
c
c       obs1-3 are displacement components:
c       obs4 = normal stress: szz
c       obs5 = surface strain: err+ett
c       obs6 = ett for r > 0 will be derived later
c       obs7 = shear stress: szr
c       obs8 = rot(u)_z = dut/dr - (dur/dt)/r + ut/r
c       obs9 = shear stress: szt
c       obs10 = tilt-r: duz/dr
c       obs11 = duz/dt/r (reserved for tilt-t)
c       obs12 = potential
c       obs13 = gravity
c       obs14 = dp/dr
c       obs15 = dp/dt/r
c       obs16 = obs8
c
        if(r(nr1).eq.0.d0)then
c
c         compute ur/r and ut/r for r -> 0
c
          fac=k*k*dk*dexp(-0.5d0*(k*rs(nr1))**2)
          do istp=1,4
            do i=1,3,2
              if(ms(istp)+i-2.eq.1)then
                cbsj(i)=dcmplx(0.5d0*fac,0.d0)
              else if(ms(istp)+i-2.eq.-1)then
                cbsj(i)=-dcmplx(0.5d0*fac,0.d0)
              else
                cbsj(i)=(0.d0,0.d0)
              endif
            enddo
            urdr(istp)=urdr(istp)
     &          +cy(2,istp)*cbsj(1)-cy(3,istp)*cbsj(3)
            utdr(istp)=utdr(istp)
     &          -cics(istp)*(cy(2,istp)*cbsj(1)+cy(3,istp)*cbsj(3))
          enddo
        endif
c
        do ir=nr1,nr2
          fac=k*dk*dexp(-0.5d0*(k*rs(ir))**2)
          x=k*r(ir)
          nx=1+idint(x/dxbsj)
          if(nx.le.nbsjmax)then
            wl=(dble(nx)*dxbsj-x)/dxbsj
            wr=1.d0-wl
            do i=-1,3
              bsj0(i)=(wl*bsj(nx-1,i)+wr*bsj(nx,i))*fac
            enddo
          else
            do i=-1,3
              bsj0(i)=dcos(x-0.5d0*pi*(dble(i)+0.5d0))*fac
     &               /dsqrt(0.5d0*pi*x)
            enddo
          endif
          do istp=1,4
            do i=1,3
              cbsj(i)=dcmplx(bsj0(ms(istp)+i-2),0.d0)
            enddo
            obs(ir,1,istp)=obs(ir,1,istp)+cy(1,istp)*cbsj(2)
            obs(ir,2,istp)=obs(ir,2,istp)
     &         +cy(2,istp)*cbsj(1)-cy(3,istp)*cbsj(3)
            obs(ir,3,istp)=obs(ir,3,istp)
     &         -cics(istp)*(cy(2,istp)*cbsj(1)+cy(3,istp)*cbsj(3))
            obs(ir,4,istp)=obs(ir,4,istp)+cy(4,istp)*cbsj(2)
            obs(ir,5,istp)=obs(ir,5,istp)-cy(5,istp)*cbsj(2)
            obs(ir,7,istp)=obs(ir,7,istp)
     &          +cy(6,istp)*cbsj(1)-cy(7,istp)*cbsj(3)
            obs(ir,8,istp)=obs(ir,8,istp)+cy(8,istp)*cbsj(2)
            obs(ir,9,istp)=obs(ir,9,istp)
     &          -cics(istp)*(cy(6,istp)*cbsj(1)+cy(7,istp)*cbsj(3))
            obs(ir,10,istp)=obs(ir,10,istp)
     &          +c05*(cy(9,istp)*cbsj(1)-cy(10,istp)*cbsj(3))
            obs(ir,11,istp)=obs(ir,11,istp)-c05*cics(istp)
     &          *(cy(9,istp)*cbsj(1)+cy(10,istp)*cbsj(3))
            obs(ir,12,istp)=obs(ir,12,istp)+cy(11,istp)*cbsj(2)
            obs(ir,13,istp)=obs(ir,13,istp)+cy(12,istp)*cbsj(2)
            obs(ir,14,istp)=obs(ir,14,istp)
     &          +c05*(cy(13,istp)*cbsj(1)-cy(14,istp)*cbsj(3))
            obs(ir,15,istp)=obs(ir,15,istp)-c05*cics(istp)
     &          *(cy(13,istp)*cbsj(1)+cy(14,istp)*cbsj(3))         
          enddo
        enddo
        if(ik.eq.nnk*nkc+nkg)then
          nnk=2*nnk
          finish=.true.
          if(r(nr1).le.0.d0)then
            do istp=1,4
              finish=finish.and.cdabs(urdr(istp)-urdr0(istp))
     &               .le.accuracy*cdabs(urdr(istp))
              urdr0(istp)=urdr(istp)
            enddo
          endif
          do istp=1,4
            do i=1,15
              y0abs=0.d0
              dyabs=0.d0
              do ir=nr1,nr2
                y0abs=y0abs+geow(ir)*cdabs(obs(ir,i,istp))
                dyabs=dyabs
     &               +geow(ir)*cdabs(obs(ir,i,istp)-obs0(ir,i,istp))
                obs0(ir,i,istp)=obs(ir,i,istp)
              enddo
              finish=finish.and.dyabs.le.accuracy*y0abs
            enddo
          enddo
          if(finish)goto 100
        endif
      enddo
      ik=ik-1
100   continue
c
      do ir=nr1,nr2
        do istp=1,4
c
c         obs6 is ett = ur/r + (dut/dt)/r
c
          if(r(ir).le.0.d0)then
            obs(ir,6,istp)=urdr(istp)
     &                    +cics(istp)*cms(istp)*utdr(istp)
          else
            obs(ir,6,istp)=(obs(ir,2,istp)+cics(istp)*cms(istp)
     &                    *obs(ir,3,istp))/dcmplx(r(ir),0.d0)
          endif
c 
c         obs5 now is err = obs5(before) - ett
c
          obs(ir,5,istp)=obs(ir,5,istp)-obs(ir,6,istp)
c
c         obs4 now is ezz
c
          obs(ir,4,istp)=(obs(ir,4,istp)
     &        -clazr*(obs(ir,5,istp)+obs(ir,6,istp)))/(clazr+c2*cmuzr)
c
c         obs7 now is ezr = erz
c
          obs(ir,7,istp)=obs(ir,7,istp)/(c2*cmuzr)
          obs(ir,16,istp)=c05*obs(ir,8,istp)
c
c         obs8 now is ert = etr
c                         = u16 + (dur/dt)/r - ut/r
c                         = 0.5 * (dut/dr + (dur/dt)/r - ut/r)
c
          if(r(ir).le.0.d0)then
            obs(ir,8,istp)=obs(ir,16,istp)
     &              -(cics(istp)*cms(istp)*urdr(istp)+utdr(istp))
          else
            obs(ir,8,istp)=obs(ir,16,istp)
     &                    -(cics(istp)*cms(istp)*obs(ir,2,istp)
     &                    +obs(ir,3,istp))/dcmplx(r(ir),0.d0)
          endif
c
c         obs9 now is ezt = etz
c
          obs(ir,9,istp)=obs(ir,9,istp)/(c2*cmuzr)
c
c         obs10 now is -dur/dz (vertical tilt)
c                      =  obs10(before) - 2 * ezr
c                      =  duz/dr - 2 * ezr
c
          obs(ir,10,istp)= obs(ir,10,istp)-c2*obs(ir,7,istp)
        enddo
      enddo
c
c     end of total wavenumber integral
c
      if(tty)then
        if(istate.eq.-1)then
          write(*,'(a,i7,a,f10.4)')'     Coseismic response:'
     &       //' samples = ',ik,' x_max = ',k*r(nr2)
        else if(istate.eq.0)then
          write(*,'(a,i7,a,f10.4)')'  Steady-state response:'
     &       //' wavenumber samples = ',ik,' x_max = ',k*r(nr2)
        else
          write(*,'(i7,a,E12.5,a,i7,a,f10.4)')lf,'.',dimag(cs)/pi2,
     &       ' Hz: samples = ',ik,' x_max = ',k*r(nr2)
        endif
      endif
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     transform to outputs                                                   c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      do istp=1,4
        do ir=nr1,nr2
c
c         displacement components
c
          du(lf,ir,1,istp)=obs(ir,1,istp)
          du(lf,ir,2,istp)=obs(ir,2,istp)
          du(lf,ir,3,istp)=obs(ir,3,istp)
c
c         stress components
c
          evs=obs(ir,4,istp)+obs(ir,5,istp)+obs(ir,6,istp)
          du(lf,ir,4,istp)=clazr*evs+c2*cmuzr*obs(ir,4,istp)
          du(lf,ir,5,istp)=clazr*evs+c2*cmuzr*obs(ir,5,istp)
          du(lf,ir,6,istp)=clazr*evs+c2*cmuzr*obs(ir,6,istp)
          du(lf,ir,7,istp)=c2*cmuzr*obs(ir,7,istp)
          du(lf,ir,8,istp)=c2*cmuzr*obs(ir,8,istp)
          du(lf,ir,9,istp)=c2*cmuzr*obs(ir,9,istp)
c
c         tilt components and rotation
c
          du(lf,ir,10,istp)=obs(ir,10,istp)
     &                 +obs(ir,14,istp)/dcmplx(g0,0.d0)
          du(lf,ir,11,istp)=obs(ir,11,istp)-c2*obs(ir,9,istp)
     &                 +obs(ir,15,istp)/dcmplx(g0,0.d0)
          du(lf,ir,12,istp)=obs(ir,16,istp)
c
c         geoid
c
          du(lf,ir,13,istp)=obs(ir,12,istp)/dcmplx(g0,0.d0)
c
c         gravity change (space-fixed)
c
          du(lf,ir,14,istp)=obs(ir,13,istp)
        enddo
      enddo
c=============================================================================
c     end of wavenumber integration
c=============================================================================
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     adding/subtracting analytical half-space solutions                     c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
      if(.not.elastic(nlr).and.istate.ne.-1)then
        do ipot=1,2
          if(ipot.eq.1)then
            potfac=(1.d0,0.d0)
            clahs=clazr
            cmuhs=cmuzr
          else
            potfac=(-1.d0,0.d0)
            clahs=dcmplx(la(nlr),0.d0)
            cmuhs=dcmplx(mu(nlr),0.d0)
          endif
          alpha=(clahs+cmuhs)/(clahs+c2*cmuhs)
c
c         explosion 
c
          istp=1
c
          dip=0.d0
          pot1=c0
          pot2=c0
          pot3=c0
          pot4=potfac*(clahs+dcmplx(2.d0/3.d0,0.d0)*cmuhs)/cmuhs
          do ir=nr1,nr2
c
c           z,r(Aki) = -z,x(Okada)
c
            xokada=r(ir)
            yokada=0.d0
            call cdc3d0(alpha,xokada,yokada,-zrec,zs,dip,
     &                pot1,pot2,pot3,pot4,
     &                ux,uy,uz,uxx,uyx,uzx,uxy,uyy,uzy,uxz,uyz,uzz,iret)
            if(iret.eq.1)then
              stop ' Error in psgwvint: anormal return from cdc3d0!'
            endif
c-----------------------------------------------------------------------
            uza=-uz
            ura=ux
c-----------------------------------------------------------------------
            szza=clahs*(uzz+uxx+uyy)+c2*cmuhs*uzz
            srra=clahs*(uzz+uxx+uyy)+c2*cmuhs*uxx
            stta=clahs*(uzz+uxx+uyy)+c2*cmuhs*uyy
            szra=cmuhs*(-uxz-uzx)
c-----------------------------------------------------------------------
            urza=-uxz
c-----------------------------------------------------------------------
            du(lf,ir,1,istp)=du(lf,ir,1,istp)+uza
            du(lf,ir,2,istp)=du(lf,ir,2,istp)+ura
c
            du(lf,ir,4,istp)=du(lf,ir,4,istp)+szza
            du(lf,ir,5,istp)=du(lf,ir,5,istp)+srra
            du(lf,ir,6,istp)=du(lf,ir,6,istp)+stta
            du(lf,ir,7,istp)=du(lf,ir,7,istp)+szra

c
            du(lf,ir,10,istp)=du(lf,ir,10,istp)-urza
          enddo
c
c         strike-slip
c
          istp=2
c
          dip=90.d0
          pot1=potfac
          pot2=c0
          pot3=c0
          pot4=c0
          do ir=nr1,nr2
c
c           z,r,t(Aki) = -z,-45 deg, -135 deg(Okada)
c
            xokada=r(ir)/dsqrt(2.d0)
            yokada=-xokada
            call cdc3d0(alpha,xokada,yokada,-zrec,zs,dip,
     &                pot1,pot2,pot3,pot4,
     &                ux,uy,uz,uxx,uyx,uzx,uxy,uyy,uzy,uxz,uyz,uzz,iret)
            if(iret.eq.1)then
              stop ' Error in psgwvint: anormal return from cdc3d0!'
            endif
c-----------------------------------------------------------------------
            uza=-uz
            ura=ux*cs45-uy*ss45
c-----------------------------------------------------------------------
            szza=clahs*(uzz+uxx+uyy)+c2*cmuhs*uzz
            esfa=uxx+uyy
            etta=esfa-(uxx*cs45**2+uyy*ss45**2-(uxy+uyx)*cs45*ss45)
            srra=clahs*(uzz+uxx+uyy)+c2*cmuhs*(esfa-etta)
            stta=clahs*(uzz+uxx+uyy)+c2*cmuhs*etta
            szra=cmuhs*(-(uxz+uzx)*cs45+(uyz+uzy)*ss45)
c-----------------------------------------------------------------------
            urza=-uxz*cs45+uyz*ss45
c-----------------------------------------------------------------------
            du(lf,ir,1,istp)=du(lf,ir,1,istp)+uza
            du(lf,ir,2,istp)=du(lf,ir,2,istp)+ura
c
            du(lf,ir,4,istp)=du(lf,ir,4,istp)+szza
            du(lf,ir,5,istp)=du(lf,ir,5,istp)+srra
            du(lf,ir,6,istp)=du(lf,ir,6,istp)+stta
            du(lf,ir,7,istp)=du(lf,ir,7,istp)+szra
c
            du(lf,ir,10,istp)=du(lf,ir,10,istp)-urza
c
c           z,r,t (Aki) = -z,x,-y (Okada)
c
            xokada=r(ir)
            yokada=0.d0
            call cdc3d0(alpha,xokada,yokada,-zrec,zs,dip,
     &                pot1,pot2,pot3,pot4,
     &                ux,uy,uz,uxx,uyx,uzx,uxy,uyy,uzy,uxz,uyz,uzz,iret)
            if(iret.eq.1)then
              stop ' Error in psgwvint: anormal return from cdc3d0!'
            endif
c-----------------------------------------------------------------------
            uta=-uy
c-----------------------------------------------------------------------
            erta=-uxy
            srta=c2*cmuhs*erta
            szta=cmuhs*(uyz+uzy)
c-----------------------------------------------------------------------
            utza=uyz
c-----------------------------------------------------------------------
            du(lf,ir,3,istp)=du(lf,ir,3,istp)+uta
c
            du(lf,ir,8,istp)=du(lf,ir,8,istp)+srta
            du(lf,ir,9,istp)=du(lf,ir,9,istp)+szta
c
            du(lf,ir,11,istp)=du(lf,ir,11,istp)-utza
            du(lf,ir,12,istp)=du(lf,ir,12,istp)-c05*(uyx-uxy)
          enddo
c
c         dip-slip
c
          istp=3
c
          dip=90.d0
          pot1=c0
          pot2=potfac
          pot3=c0
          pot4=c0
          do ir=nr1,nr2
c
c           z,r,t(Aki) = -z,y,x(Okada)
c
            xokada=0.d0
            yokada=r(ir)
            call cdc3d0(alpha,xokada,yokada,-zrec,zs,dip,
     &                pot1,pot2,pot3,pot4,
     &                ux,uy,uz,uxx,uyx,uzx,uxy,uyy,uzy,uxz,uyz,uzz,iret)
            if(iret.eq.1)then
              stop ' Error in psgwvint: anormal return from cdc3d0!'
            endif
c-----------------------------------------------------------------------
            uza=-uz
            ura=uy
c-----------------------------------------------------------------------
            szza=clahs*(uzz+uxx+uyy)+c2*cmuhs*uzz
            srra=clahs*(uzz+uxx+uyy)+c2*cmuhs*uyy
            stta=clahs*(uzz+uxx+uyy)+c2*cmuhs*uxx
            szra=cmuhs*(-uyz-uzy)
c-----------------------------------------------------------------------
            urza=-uyz
c-----------------------------------------------------------------------
            du(lf,ir,1,istp)=du(lf,ir,1,istp)+uza
            du(lf,ir,2,istp)=du(lf,ir,2,istp)+ura
c
            du(lf,ir,4,istp)=du(lf,ir,4,istp)+szza
            du(lf,ir,5,istp)=du(lf,ir,5,istp)+srra
            du(lf,ir,6,istp)=du(lf,ir,6,istp)+stta
            du(lf,ir,7,istp)=du(lf,ir,7,istp)+szra
c
            du(lf,ir,10,istp)=du(lf,ir,10,istp)-urza
c
c            du(lf,ir,14,istp)=du(lf,ir,14,istp)
c     &                      +dcmplx(2.d0*g0/rearth,0.d0)*uza
c
c           z,r,t(Aki) = -z,x,-y(okada)
c
            xokada=r(ir)
            yokada=0.d0
            call cdc3d0(alpha,xokada,yokada,-zrec,zs,dip,
     &                pot1,pot2,pot3,pot4,
     &                ux,uy,uz,uxx,uyx,uzx,uxy,uyy,uzy,uxz,uyz,uzz,iret)
            if(iret.eq.1)then
              stop ' Error in psgwvint: anormal return from cdc3d0!'
            endif
c-----------------------------------------------------------------------
            uta=-uy
c-----------------------------------------------------------------------
            erta=-uxy
            srta=c2*cmuhs*erta
            szta=cmuhs*(uyz+uzy)
c-----------------------------------------------------------------------
            utza=uyz
c-----------------------------------------------------------------------
            du(lf,ir,3,istp)=du(lf,ir,3,istp)+uta
c
            du(lf,ir,8,istp)=du(lf,ir,8,istp)+srta
            du(lf,ir,9,istp)=du(lf,ir,9,istp)+szta
c
            du(lf,ir,11,istp)=du(lf,ir,11,istp)-utza
            du(lf,ir,12,istp)=du(lf,ir,12,istp)-c05*(uyx-uxy)
          enddo
c
c         clvd 
c
          istp=4
c
          dip=0.d0
          pot1=c0
          pot2=c0
          pot3=(0.75d0,0.d0)*potfac
          pot4=(-c05-(0.75d0,0.d0)*clahs/cmuhs)*potfac
          do ir=nr1,nr2
c
c           z,r(Aki) = -z,x(Okada)
c
            xokada=r(ir)
            yokada=0.d0
            call cdc3d0(alpha,xokada,yokada,-zrec,zs,dip,
     &                pot1,pot2,pot3,pot4,
     &                ux,uy,uz,uxx,uyx,uzx,uxy,uyy,uzy,uxz,uyz,uzz,iret)
            if(iret.eq.1)then
              stop ' Error in psgwvint: anormal return from cdc3d0!'
            endif
c-----------------------------------------------------------------------
            uza=-uz
            ura=ux
c-----------------------------------------------------------------------
            szza=clahs*(uzz+uxx+uyy)+c2*cmuhs*uzz
            srra=clahs*(uzz+uxx+uyy)+c2*cmuhs*uxx
            stta=clahs*(uzz+uxx+uyy)+c2*cmuhs*uyy
            szra=cmuhs*(-uxz-uzx)
c-----------------------------------------------------------------------
            urza=-uxz
c-----------------------------------------------------------------------
            du(lf,ir,1,istp)=du(lf,ir,1,istp)+uza
            du(lf,ir,2,istp)=du(lf,ir,2,istp)+ura
c
            du(lf,ir,4,istp)=du(lf,ir,4,istp)+szza
            du(lf,ir,5,istp)=du(lf,ir,5,istp)+srra
            du(lf,ir,6,istp)=du(lf,ir,6,istp)+stta
            du(lf,ir,7,istp)=du(lf,ir,7,istp)+szra
c
            du(lf,ir,10,istp)=du(lf,ir,10,istp)-urza
          enddo
        enddo
      endif
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c       end of adding half-space solutions                                   c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      if(lzrec.eq.1)then
        do ir=nr1,nr2
          do istp=1,4
            du(lf,ir,4,istp)=c0
            du(lf,ir,7,istp)=c0
            du(lf,ir,9,istp)=c0
          enddo
        enddo
      endif
c
      return
      end
