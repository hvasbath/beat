      subroutine psgsh(y,k)
      implicit none
c
c     calculation of response to sh source
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
      integer i,istp,l,n,ly,lup,lmd,llw,key
      double precision hply
      double complex ck
      double complex b(2,4)
      double complex yup(2),ylw(2),yup0(2),ylw0(2),coef(2,2)
c
c     sh layer hask matrices
c
      double complex hkup(2,2,nzmax),hklw(2,2,nzmax)
c
      ck=dcmplx(k,0.d0)
c
c===============================================================================
      lup=1
      llw=lp
      lmd=ls
c
c     matrix propagation from surface to source
c
      do l=lup,ls-1
        ly=l
        n=nno(ly)
        hply=hp(ly)
        call psghksh(hkup(1,1,ly),2,k,hply,n)
      enddo
      do l=ls,llw-1
        ly=l
        n=nno(ly)
        hply=hp(ly)
        call psghksh(hklw(1,1,ly),2,k,-hply,n)
      enddo
c
      yup(1)=(1.d0,0.d0)
      yup(2)=(0.d0,0.d0)
      if(lup.eq.lzrec)call cmemcpy(yup,yup0,2)
c
      call psgpropsh(hkup,lup,lmd,k,yup,yup0)
c
c===============================================================================
c
c     matrix propagation from half-space to source
c
c     ylw: the starting solution vector
c
      ylw(1)=(1.d0,0.d0)
      ylw(2)=-cmu(nno(llw))*ck
      if(llw.eq.lzrec)call cmemcpy(ylw,ylw0,2)
c
      call psgpropsh(hklw,llw,lmd,k,ylw,ylw0)
c
c===============================================================================
c
c     conditions on the source surface
c
c
c     point source function
c
      do istp=1,4
        do i=1,2
          b(i,istp)=sfct0(i+4,istp)+sfct1(i+4,istp)*ck
        enddo
      enddo
      do i=1,2
        coef(i,1)=yup(i)
        coef(i,2)=-ylw(i)
      enddo
      key=0
      call cdsvd500(coef,b,2,4,0.d0,key)
      if(key.eq.0)then
        print *,'warning in psgsh: anormal exit from cdgemp!'
        return
      endif
      if(lzrec.lt.ls)then
        do istp=1,4
          do i=1,2
            y(i+4,istp)=b(1,istp)*yup0(i)
          enddo
        enddo
      else if(lzrec.gt.ls)then
        do istp=1,4
          do i=1,2
            y(i+4,istp)=b(2,istp)*ylw0(i)
          enddo
        enddo
      else
        do istp=1,4
          do i=1,2
            y(i+4,istp)=(0.5d0,0.d0)*(b(1,istp)*yup0(i)
     &                               +b(2,istp)*ylw0(i))
          enddo
        enddo
      endif
      return
      end
