      subroutine cmbfix(sxx,syy,szz,sxy,syz,szx,
     &                  p,f,cmb,sig,st,di,ra)
      implicit none
c
c     calculate Coulomb stress
c
c     input:
c     stress tensor, pore pressure, friction coefficient
c     rupture orientation parameter (strike, dip and rake)
c
      double precision sxx,syy,szz,sxy,syz,szx,p,f,cmb,sig,st,di,ra
c
c     return:
c     Coulomb stress (cmb) and normal stress (sig)
c
c     local memories:
c
      integer i,j
      double precision pi,deg2rad,st0,di0,ra0,tau
      double precision s(3,3),ns(3),ts(3),rst(3),rdi(3)
c
      pi=4.d0*datan(1.d0)
      deg2rad=pi/180.d0
      st0=st*deg2rad
      di0=di*deg2rad
      ra0=ra*deg2rad
c
      s(1,1)=sxx
      s(1,2)=sxy
      s(1,3)=szx
      s(2,1)=sxy
      s(2,2)=syy
      s(2,3)=syz
      s(3,1)=szx
      s(3,2)=syz
      s(3,3)=szz
c
      ns(1)=dsin(di0)*dcos(st0+0.5d0*pi)
      ns(2)=dsin(di0)*dsin(st0+0.5d0*pi)
      ns(3)=-dcos(di0)
c
      rst(1)=dcos(st0)
      rst(2)=dsin(st0)
      rst(3)=0.d0
c
      rdi(1)=dcos(di0)*dcos(st0+0.5d0*pi)
      rdi(2)=dcos(di0)*dsin(st0+0.5d0*pi)
      rdi(3)=dsin(di0)
c
      do i=1,3
        ts(i)=rst(i)*dcos(ra0)-rdi(i)*dsin(ra0)
      enddo
c
      sig=0.d0
      tau=0.d0
      do j=1,3
        do i=1,3
          sig=sig+ns(i)*s(i,j)*ns(j)
          tau=tau+ts(i)*s(i,j)*ns(j)
        enddo
      enddo
c
      cmb=tau+f*(sig+p)
      return
      end