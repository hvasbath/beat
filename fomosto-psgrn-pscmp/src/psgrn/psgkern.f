      subroutine psgkern(y,k,clahs,cmuhs)
      implicit none
c
c     calculation of response function in Laplace domain
c     y(8,4): solution vector (complex)
c     k: wave number (input)
c
      double precision k
      double complex clahs,cmuhs,y(8,4)
c
      include 'psgglob.h'
c
      integer i,istp
      double complex yhs(8,4)
c
      double precision eps
      data eps/1.0d-06/
c
      do istp=1,4
        do i=1,8
          y(i,istp)=(0.d0,0.d0)
          yhs(i,istp)=(0.d0,0.d0)
        enddo
      enddo
c
      nongravity=kgmax*grfac.lt.eps*k
c
      call psgpsv(y,k)
      call psgsh(y,k)
c
c     subtract the halfspace solution
c
      call psghskern(yhs,k,clahs,cmuhs)
c
      do istp=1,4
        do i=1,8
          y(i,istp)=y(i,istp)-yhs(i,istp)
        enddo
      enddo
c
      return
      end	  
