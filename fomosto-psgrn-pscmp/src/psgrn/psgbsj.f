      subroutine psgbsj(ierr)
      implicit none
      integer ierr
c
      include 'psgglob.h'
c
      integer ix
      double precision x
      double precision bessj0,bessj1,bessj
c
      dxbsj=8.d0*datan(1.d0)/dble(dnx)
      do ix=0,nbsjmax
        x=dble(ix)*dxbsj
        bsj(ix,0)=bessj0(x)
        bsj(ix,1)=bessj1(x)
        if(x.gt.2.d0)then
          bsj(ix,2)=bsj(ix,1)*2.d0/x-bsj(ix,0)
        else
          bsj(ix,2)=bessj(2,x)
        endif
        if(x.gt.3.d0)then
          bsj(ix,3)=bsj(ix,2)*4.d0/x-bsj(ix,1)
        else
          bsj(ix,3)=bessj(3,x)
        endif
        bsj(ix,-1)=-bsj(ix,1)
      enddo
c
      return
      end