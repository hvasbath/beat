      double precision function bessj1(x)
      implicit none
c
c     J_1(x)
c
      double precision x,ax,x1,x2,theta,fct
c
      double precision pi2
      double precision a0,a1,a2,a3,a4,a5,a6
      double precision b0,b1,b2,b3,b4,b5,b6
      double precision c0,c1,c2,c3,c4,c5,c6
c
      data pi2/6.28318530717959d0/
      data a0,a1,a2,a3,a4,a5,a6/  0.50000000d0,
     + -0.56249985d0, 0.21093573d0,-0.03954289d0,
     +  0.00443319d0,-0.00031761d0, 0.00001109d0/
      data b0,b1,b2,b3,b4,b5,b6/  0.79788456d0,
     +  0.00000156d0, 0.01659667d0, 0.00017105d0,
     + -0.00249511d0, 0.00113653d0,-0.00020033d0/
      data c0,c1,c2,c3,c4,c5,c6/ -2.35619449d0,
     +  0.12499612d0, 0.00005650d0,-0.00637879d0,
     +  0.00074348d0, 0.00079824d0,-0.00029166d0/
c
      ax=dabs(x)
      if(ax.le.3.d0)then
        x2=ax*ax/9.d0
        bessj1=x*(a0+x2*(a1+x2*(a2+x2*(a3+x2*(a4+x2*(a5+x2*a6))))))
      else
        x1=3.d0/ax
        fct=b0+x1*(b1+x1*(b2+x1*(b3+x1*(b4+x1*(b5+x1*b6)))))
        theta=ax+c0+x1*(c1+x1*(c2+x1*(c3+x1*(c4+x1*(c5+x1*c6)))))
        theta=dmod(theta,pi2)
        bessj1=dsign(1.d0,x)*fct*dcos(theta)/dsqrt(ax)
      endif
c
      return
      end
