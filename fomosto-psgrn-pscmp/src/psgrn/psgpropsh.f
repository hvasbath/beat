      subroutine psgpropsh(hk,l1,l2,k,ysh,ysh0)
      implicit none
c
	include 'psgglob.h'
c
c	propagation of sh vectors
c
      integer l1,l2
      double precision k
      double complex ysh(2),ysh0(2)
      double complex hk(2,2,nzmax)
c
c     work space
c
      integer l
      double complex cnorm,yswab(2)
c
	if(l1.eq.l2)then
	  return
	else if(l1.lt.l2)then
        do l=l1+1,l2
c
c         determination of propagation matrix
c
	    call caxcb(hk(1,1,l-1),ysh,2,2,1,yswab)
	    call cmemcpy(yswab,ysh,2)
          if(l.gt.lzrec)then
            cnorm=dcmplx(dexp(-k*hp(l-1)),0.d0)
            ysh0(1)=ysh0(1)*cnorm
            ysh0(2)=ysh0(2)*cnorm
	    else if(l.eq.lzrec)then
            call cmemcpy(ysh,ysh0,2)
          endif
        enddo
	else
        do l=l1-1,l2,-1
c
c         determination of propagation matrix
c
	    call caxcb(hk(1,1,l),ysh,2,2,1,yswab)
	    call cmemcpy(yswab,ysh,2)
          if(l.lt.lzrec)then
            cnorm=dcmplx(dexp(-k*hp(l)),0.d0)
            ysh0(1)=ysh0(1)*cnorm
            ysh0(2)=ysh0(2)*cnorm
	    else if(l.eq.lzrec)then
            call cmemcpy(ysh,ysh0,2)
          endif
        enddo
	endif
	return
	end
