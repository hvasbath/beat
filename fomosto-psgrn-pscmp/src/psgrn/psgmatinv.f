      subroutine psgmatinv(a,k,z,n)
      implicit none
c
      include 'psgglob.h'
c
      integer n
      double precision k,z
      double complex a(6,6)
c
      integer i,j,key
      double complex b(6,6)
      double complex ck,ck2,cx,cpx,cmx,cxi,cet,cgr
      double complex cmun,cetcx,c2cxi,c2ckcxi,c4ck2cxi,c4ckcxi,c4ckcmcxi
c
      double complex c1,c2,c4
      data c1,c2,c4/(1.d0,0.d0),(2.d0,0.d0),(4.d0,0.d0)/
c
      if(nongravity)then
        ck=dcmplx(k,0.d0)
        ck2=dcmplx(k*k,0.d0)
        cgr=dcmplx(gamma*rho(n),0.d0)
c
        cx=dcmplx(k*z,0.d0)
        cxi=cla(n)+c2*cmu(n)
        cet=cla(n)+cmu(n)
        cpx=c1+cx
        cmx=c1-cx
        cmun=cmu(n)
        cetcx=cet*cx
        c2cxi=c2*cxi
        c2ckcxi=c2cxi*ck
        c4ck2cxi=c4*ck2*cxi
        c4ckcxi=c4*ck*cxi
        c4ckcmcxi=c4ckcxi*cmun
c
        a(1,1)=cetcx/c2cxi
        a(1,2)=(cmun+cetcx)/c4ckcmcxi
        a(1,3)=cet*cmx/c2cxi
        a(1,4)=(cxi-cetcx)/c4ckcmcxi
        a(1,5)=(0.d0,0.d0)
        a(1,6)=(0.d0,0.d0)
c
        a(2,1)=-cet*cx/c2cxi
        a(2,2)=(-cmun+cetcx)/c4ckcmcxi
        a(2,3)=-cet*cpx/c2cxi
        a(2,4)=(cxi+cetcx)/c4ckcmcxi
        a(2,5)=(0.d0,0.d0)
        a(2,6)=(0.d0,0.d0)
c
        a(3,1)=cmun/c2cxi
        a(3,2)=c1/c4ckcxi
        a(3,3)=-cmun/c2cxi
        a(3,4)=-c1/c4ckcxi
        a(3,5)=(0.d0,0.d0)
        a(3,6)=(0.d0,0.d0)
c
        a(4,1)=cmun/c2cxi
        a(4,2)=-c1/c4ckcxi
        a(4,3)=cmun/c2cxi
        a(4,4)=-c1/c4ckcxi
        a(4,5)=(0.d0,0.d0)
        a(4,6)=(0.d0,0.d0)
c
        a(5,1)=cgr*(cet-cmun*cx)/c2ckcxi
        a(5,2)=-cgr*cx/c4ck2cxi
        a(5,3)=cgr*cmun*cx/c2ckcxi
        a(5,4)=cgr*cpx/c4ck2cxi
        a(5,5)=c1/c2
        a(5,6)=c1/(c2*ck)
c
        a(6,1)=-cgr*(cet+cmun*cx)/c2ckcxi
        a(6,2)=cgr*cx/c4ck2cxi
        a(6,3)=-cgr*cmun*cx/c2ckcxi
        a(6,4)=-cgr*cmx/c4ck2cxi
        a(6,5)=c1/c2
        a(6,6)=-c1/(c2*ck)
      else
        do i=1,6
          do j=1,6
            a(i,j)=0.d0
          enddo
          a(i,i)=(1.d0,0.d0)
        enddo
        call psgmatrix(b,k,z,n)
        key=1
        call cdsvd500(b,a,6,6,0.d0,key)
        if(key.eq.0)then
          print *,'warning in pegmativ: anormal exit from cdgemp!'
          return
        endif
      endif
c
      return
      end
