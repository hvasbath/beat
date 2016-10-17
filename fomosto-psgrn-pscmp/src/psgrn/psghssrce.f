      subroutine psghssrce(dislocation,clahs,cmuhs)
      implicit none
c
      double precision dislocation
      double complex clahs,cmuhs
c
      include 'psgglob.h'
c
      integer i,istp
      double complex cdam
c
      double precision pi
      double complex c2,c3,c4
c
      data pi/3.14159265358979d0/
      data c2,c3,c4/(2.d0,0.d0),(3.d0,0.d0),(4.d0,0.d0)/
c
      do istp=1,4
        do i=1,8
          sfcths0(i,istp)=(0.d0,0.d0)
          sfcths1(i,istp)=(0.d0,0.d0)
        enddo
      enddo
c
      cdam=dcmplx(dislocation/(2.d0*pi),0.d0)
c
c     explosion (m11=m22=m33=M0)
c
      sfcths0(1,1)=-cdam*(clahs+c2*cmuhs/c3)/(clahs+c2*cmuhs)
      sfcths1(4,1)=c2*cmuhs*sfcths0(1,1)
c
c     strike-slip (m12=m21=M0)
c
      sfcths1(4,2)=cdam*cmuhs
      sfcths1(6,2)=-sfcths1(4,2)
c
c     dip-slip (m13=m31=M0)
c
      sfcths0(3,3)=-cdam
      sfcths0(5,3)=sfcths0(3,3)
c
c     compensated linear vector dipole (CLVD) (m11=m22=-M0/2, M33=M0)
c
      sfcths0(1,4)=-cdam*cmuhs/(clahs+c2*cmuhs)
      sfcths1(4,4)=cdam*cmuhs*(c3-c4*cmuhs/(clahs+c2*cmuhs))/c2
c
      return
      end
