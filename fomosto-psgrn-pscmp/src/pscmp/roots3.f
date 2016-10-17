      subroutine roots3(b,c,d,x)
      implicit none
c
c     finding 3 real roots of eq: x**3 + b*x**2 + c*x + d = 0
c
c     input:
c
      double precision b,c,d
c
c     output:
c
      double precision x(3)
c
c     local memories:
c
      double precision p,q,n,u,delta
c
      if(d.eq.0.d0)then
        x(1)=0.d0
        p=b*b-4.d0*c
        if(p.lt.0.d0)then
          print *,' Error in roots3: not all roots are real!'
        else
          x(2)=0.5d0*(-b+dsqrt(p))
          x(3)=0.5d0*(-b-dsqrt(p))
        endif
      else
        p=c-b*b/3.d0
        q=d-b*(c-2.d0*b*b/9.d0)/3.d0
        if(4.d0*p**3+27.d0*q**2.ge.0.d0)then
          print *,' Error in roots3: not all roots are real!'
        else
          n=dsqrt(-4.d0*p/3.d0)
          u=dacos(-0.5d0*q/dsqrt(-p**3/27.d0))/3.d0
          delta=8.d0*datan(1.d0)/3.d0
          x(1)=n*dcos(u)-b/3.d0
          x(2)=n*dcos(u+delta)-b/3.d0
          x(3)=n*dcos(u+2.d0*delta)-b/3.d0
        endif
      endif
      return
      end