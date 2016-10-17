      subroutine hsmatrix(a,k,z,clahs,cmuhs)
      implicit none
c
      double precision k,z
      double complex clahs,cmuhs,a(6,6)
c
      double complex ck,cx,cp,cm,et
c
      ck=dcmplx(k,0.d0)
      cx=dcmplx(k*z,0.d0)
      et=clahs+cmuhs
      cp=(1.d0,0.d0)+cx
      cm=(1.d0,0.d0)-cx
c
      a(1,1)=(1.d0,0.d0)
      a(2,1)=(2.d0,0.d0)*cmuhs*ck
      a(3,1)=(1.d0,0.d0)
      a(4,1)=(2.d0,0.d0)*cmuhs*ck
      a(5,1)=(0.d0,0.d0)
      a(6,1)=(0.d0,0.d0)
c
      a(1,2)=(1.d0,0.d0)
      a(2,2)=-(2.d0,0.d0)*cmuhs*ck
      a(3,2)=-(1.d0,0.d0)
      a(4,2)=(2.d0,0.d0)*cmuhs*ck
      a(5,2)=(0.d0,0.d0)
      a(6,2)=(0.d0,0.d0)
c
      a(1,3)=(1.d0,0.d0)+et*cm/cmuhs
      a(2,3)=(2.d0,0.d0)*et*cm*ck
      a(3,3)=-(1.d0,0.d0)-et*cx/cmuhs
      a(4,3)=-(2.d0,0.d0)*et*cx*ck
      a(5,3)=(0.d0,0.d0)
      a(6,3)=(0.d0,0.d0)
c
      a(1,4)=(1.d0,0.d0)+et*cp/cmuhs
      a(2,4)=-(2.d0,0.d0)*et*cp*ck
      a(3,4)=(1.d0,0.d0)-et*cx/cmuhs
      a(4,4)=(2.d0,0.d0)*et*cx*ck
      a(5,4)=(0.d0,0.d0)
      a(6,4)=(0.d0,0.d0)
c
      a(1,5)=(0.d0,0.d0)
      a(2,5)=(0.d0,0.d0)
      a(3,5)=(0.d0,0.d0)
      a(4,5)=(0.d0,0.d0)
      a(5,5)=(1.d0,0.d0)
      a(6,5)=ck*cmuhs
c
      a(1,6)=(0.d0,0.d0)
      a(2,6)=(0.d0,0.d0)
      a(3,6)=(0.d0,0.d0)
      a(4,6)=(0.d0,0.d0)
      a(5,6)=(1.d0,0.d0)
      a(6,6)=-ck*cmuhs
c
      return
      end
