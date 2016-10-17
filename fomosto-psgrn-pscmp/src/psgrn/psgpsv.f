      subroutine psgpsv(y,k)
      implicit none
c
c     calculation of response to p-sv source
c     y(8,4): solution vector (complex)
c     k: wave number
c
      double precision k
      double complex y(8,4)
c
      include 'psgglob.h'
c
c     work space
c
      integer i,istp,j,l,n,ly,lup,lmd,llw,key
      double precision zup,zlw
      double complex ck
      double complex b(6,4),cy(6,4)
      double complex yup(6,3),ylw(6,3)
      double complex ma0(6,6),c0(6,3),coef(6,6)
      double complex b3(3,4),coef3(3,3)
      double complex yup0(6,3),ylw0(6,3)
c
c     psv layer matrics
c
      double complex maup(6,6,nzmax),maiup(6,6,nzmax)
      double complex malw(6,6,nzmax),mailw(6,6,nzmax)
c
      ck=dcmplx(k,0.d0)
c
c===============================================================================
c
      lup=1
      llw=lp
      lmd=ls
c
      do l=lup,ls-1
        ly=l
        n=nno(ly)
        zup=zp(ly)
        zlw=zp(ly)+hp(ly)
        call psgmatrix(maup(1,1,ly),k,zlw,n)
        call psgmatinv(maiup(1,1,ly),k,zup,n)
      enddo
      do l=ls,llw
        ly=l
        n=nno(ly)
        zup=zp(ly)
        zlw=zp(ly)+hp(ly)
        call psgmatrix(malw(1,1,ly),k,zup,n)
        call psgmatinv(mailw(1,1,ly),k,zlw,n)
      enddo
c
c     matrix propagation from surface to source
c
      do j=1,3
        do i=1,6
          yup(i,j)=(0.d0,0.d0)
        enddo
      enddo
      yup(1,1)=(1.d0,0.d0)
      if(ioc.eq.0)then
        yup(2,1)=-dcmplx(denswater*g0*grfac,0.d0)*yup(1,1)
        yup(6,1)=-dcmplx(gamma*denswater,0.d0)*yup(1,1)
      endif
      yup(3,2)=(1.d0,0.d0)
      yup(5,3)=(1.d0,0.d0)
      yup(6,3)=ck*yup(5,3)
      if(lup.eq.lzrec)call cmemcpy(yup,yup0,18)
c
      call psgproppsv(maup,maiup,lup,lmd,yup,yup0,k)
c
c===============================================================================
c
c     matrix propagation from half-space to source
c
c     coefficient vectors in the half-space
c
      do j=1,3
        do i=1,6
          c0(i,j)=(0.d0,0.d0)
        enddo
      enddo
      c0(2,1)=(1.d0,0.d0)
      c0(4,2)=(1.d0,0.d0)
      c0(6,3)=(1.d0,0.d0)
      call caxcb(malw(1,1,llw),c0,6,6,3,ylw)
c
      if(llw.eq.lzrec)call cmemcpy(ylw,ylw0,18)
c
      call psgproppsv(malw,mailw,llw,lmd,ylw,ylw0,k)
c
c===============================================================================
c
c     conditions on the source surface
c
c
c     source function
c
      do istp=1,4
        do i=1,4
          b(i,istp)=sfct0(i,istp)+sfct1(i,istp)*ck
        enddo
        do i=5,6
          b(i,istp)=sfct0(i+2,istp)+sfct1(i+2,istp)*ck
        enddo
      enddo
      do i=1,6
        do j=1,3
          coef(i,j)=yup(i,j)
          coef(i,j+3)=-ylw(i,j)
        enddo
      enddo
      key=0
      call cdsvd500(coef,b,6,4,0.d0,key)
      if(key.eq.0)then
        print *,'warning in pegpsv: anormal exit from cdgemp!'
        return
      endif
      if(lzrec.lt.ls)then
        do istp=1,4
          do i=1,6
            cy(i,istp)=(0.d0,0.d0)
            do j=1,3
              cy(i,istp)=cy(i,istp)+b(j,istp)*yup0(i,j)
            enddo
          enddo
        enddo
      else if(lzrec.gt.ls)then
        do istp=1,4
          do i=1,6
            cy(i,istp)=(0.d0,0.d0)
            do j=1,3
              cy(i,istp)=cy(i,istp)+b(j+3,istp)*ylw0(i,j)
            enddo
          enddo
        enddo
      else
        do istp=1,4
          do i=1,6
            cy(i,istp)=(0.d0,0.d0)
            do j=1,3
              cy(i,istp)=cy(i,istp)+(0.5d0,0.d0)
     &             *(b(j,istp)*yup0(i,j)+b(j+3,istp)*ylw0(i,j))
            enddo
          enddo
        enddo
      endif
c
      do istp=1,4
        do i=1,4
          y(i,istp)=cy(i,istp)
        enddo
      enddo
c
c     y7 <- y5
c     y8 = dy7/dz = y6 + 4*pi*G*rho*y1
c
      if(lzrec.eq.1)then
        do istp=1,4
          y(2,istp)=(0.d0,0.d0)
          y(7,istp)=cy(5,istp)
          y(8,istp)=ck*cy(5,istp)
        enddo
      else
        do istp=1,4
          y(7,istp)=cy(5,istp)
          y(8,istp)=cy(6,istp)
     &             +dcmplx(gamma*rho(nno(lzrec-1)),0.d0)*cy(1,istp)
        enddo
      endif
c
      return
      end
