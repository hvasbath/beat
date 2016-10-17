      subroutine psgmatrix(a,k,z,n)
      implicit none
c
      include 'psgglob.h'
c
      integer n
      double precision k,z
      double complex a(6,6)
c
      integer i,j,j0
      double precision rho2grf
      double complex ck,ck2,cz,cx,cgar
      double complex cet,cpx,cmx,ca,cb,cka,ckgp,ckgk,ckg
      double complex c2cmck,cepj,cetcx,ckg2cka,clan,cmun
      double complex dep,dm,ddm2,ddm4,dmdm2
      double complex b(6,4),d(6),dm2(4),cm(4),cep(4)
c
      integer i0
      double precision eps
      double complex c0,c1,c2,c3
      data i0/8/
      data eps/1.0d-02/
      data c0,c1,c2,c3/(0.d0,0.d0),(1.d0,0.d0),(2.d0,0.d0),(3.d0,0.d0)/
c
      ck=dcmplx(k,0.d0)
      ck2=ck*ck
      cz=dcmplx(z,0.d0)
      cgar=dcmplx(gamma*rho(n),0.d0)
      clan=cla(n)
      cmun=cmu(n)
c
      if(nongravity)then
        cet=clan+cmun
        cx=dcmplx(k*z,0.d0)
        cpx=c1+cx
        cmx=c1-cx
        c2cmck=c2*cmun*ck
        cetcx=cet*cx
c
        a(1,1)=c1
        a(2,1)=c2cmck
        a(3,1)=c1
        a(4,1)=c2cmck
        a(5,1)=c0
        a(6,1)=-cgar
c
        a(1,2)=c1
        a(2,2)=-c2cmck
        a(3,2)=-c1
        a(4,2)=c2cmck
        a(5,2)=c0
        a(6,2)=-cgar
c
        a(1,3)=c1+cet*cmx/cmun
        a(2,3)=c2*cet*cmx*ck
        a(3,3)=-c1-cetcx/cmun
        a(4,3)=-c2*cetcx*ck
        a(5,3)=cgar*cz
        a(6,3)=cgar*(cx-cet*cmx/cmun)
c
        a(1,4)=c1+cet*cpx/cmun
        a(2,4)=-c2*cet*cpx*ck
        a(3,4)=c1-cetcx/cmun
        a(4,4)=c2*cetcx*ck
        a(5,4)=cgar*cz
        a(6,4)=cgar*(-cx-cet*cpx/cmun)
      else
        cb=cmun/(clan+c2*cmun)
        ca=c1-cb
        cka=dcmplx(dsqrt(2.d0/3.d0)*k,0.d0)
        rho2grf=rho(n)*g0*grfac
        ckgp=dcmplx(rho2grf,0.d0)/(clan+c2*cmun)
        ckgk=dcmplx(rho2grf,0.d0)/(clan+c2*cmun/c3)
        ckg=cdsqrt(ckgp*ckgk/c2)
        ckg2cka=cdsqrt((ckg+cka)*(ckg-cka))
c
        dm2(1)=ckg*(ckg+ckg2cka)
        dm2(2)=dm2(1)
        dm2(3)=ckg*(ckg-ckg2cka)
        dm2(4)=dm2(3)
c
        do j=1,3,2
          cm(j)=cdsqrt(ck2+dm2(j))
          cep(j)=cdexp(dm2(j)*cz/(cm(j)+ck))
        enddo
        do j=2,4,2
          cm(j)=-cm(j-1)
          cep(j)=cdexp(dm2(j)*cz/(cm(j)-ck))
        enddo
c
        do j=1,4
          b(1,j)=(ca-cb*dm2(j)/ck2)*dm2(j)
          b(3,j)=(ckgp+ca*cm(j))*dm2(j)/ck
          b(5,j)=-cgar*(ckgp+cb*cm(j)*dm2(j)/ck2)
          b(2,j)=(clan+c2*cmun)*cm(j)*b(1,j)-clan*ck*b(3,j)
          b(4,j)=cmun*(ck*b(1,j)+cm(j)*b(3,j))
          b(6,j)=cm(j)*b(5,j)-cgar*b(1,j)
        enddo
c
        if(k.le.cdabs(ckg))then
          do j=1,4
            cepj=cep(j)
            do i=1,6
              a(i,j)=b(i,j)*cepj
            enddo
          enddo
        else
          do j=1,2
            cepj=cep(j)
            do i=1,6
              a(i,j)=b(i,j)*cepj
            enddo
          enddo
c
c         y3 <- (y1-y3)*k/(m1-m3), y4 <- (y2-y4)*k/(m2-m4)
c         for numerical stability
c
          do j=3,4
            j0=j-2
            cepj=cep(j)
            dm=(dm2(j0)-dm2(j))/(cm(j0)+cm(j))
c
c           note that all other terms mutiplied by factor k/dm
c
            ddm2=ck*(cm(j0)+cm(j))
            ddm4=ddm2*(dm2(j0)+dm2(j))
            dmdm2=ck*dm2(j0)+cm(j)*ddm2
            cpx=dm*cz
            if(cdabs(cz).eq.0.d0)then
              dep=c0
            else if(cdabs(cpx).gt.eps)then
              dep=(cep(j0)-cepj)*ck/dm
            else
              dep=c1
              do i=i0,2,-1
                dep=c1+cpx*dep/dcmplx(dble(i),0.d0)
              enddo
              dep=ck*cz*dep*cepj
            endif
c
            d(1)=ca*ddm2-cb*ddm4/ck2
            d(3)=(ckgp*ddm2+ca*dmdm2)/ck
            d(5)=-cgar*cb*dmdm2/ck2
            d(2)=(clan+c2*cmun)*(ck*b(1,j0)+cm(j)*d(1))-clan*ck*d(3)
            d(4)=cmun*(ck*d(1)+ck*b(3,j0)+cm(j)*d(3))
            d(6)=ck*b(5,j0)+cm(j)*d(5)-cgar*d(1)
c
            do i=1,6
              a(i,j)=b(i,j0)*dep+d(i)*cepj
            enddo
          enddo
        endif
      endif
c
      a(1,5)=c0
      a(2,5)=c0
      a(3,5)=c0
      a(4,5)=c0
      a(5,5)=c1
      a(6,5)=ck
c
      a(1,6)=c0
      a(2,6)=c0
      a(3,6)=c0
      a(4,6)=c0
      a(5,6)=c1
      a(6,6)=-ck
c
      return
      end