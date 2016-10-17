      subroutine  cdc3d0(alpha,x,y,z,depth,dip,pot1,pot2,pot3,pot4,
     *               ux,uy,uz,uxx,uyx,uzx,uxy,uyy,uzy,uxz,uyz,uzz,iret)
      implicit none
      integer iret
      double precision x,y,z,depth,dip
      double complex pot1,pot2,pot3,pot4,alpha,ux,uy,uz
      double complex uxx,uyx,uzx,uxy,uyy,uzy,uxz,uyz,uzz
c
c********************************************************************
c*****                                                          *****
c*****    displacement and strain at depth                      *****
c*****    due to buried point source in a semiinfinite medium   *****
c*****                         coded by  y.okada ... sep.1991   *****
c*****                         revised   y.okada ... nov.1991   *****
c*****                         modified  r.wang  ... oct.2004   *****
c*****                                                          *****
c********************************************************************
c
c***** input
c*****   alpha : medium constant  (lambda+myu)/(lambda+2*myu)
c*****   x,y,z : coordinate of observing point
c*****   depth : source depth
c*****   dip   : dip-angle (degree)
c*****   pot1-pot4 : strike-, dip-, tensile- and inflate-potency
c*****       potency=(  moment of double-couple  )/myu     for pot1,2
c*****       potency=(intensity of isotropic part)/lambda  for pot3
c*****       potency=(intensity of linear dipole )/myu     for pot4
c
c***** output
c*****   ux, uy, uz  : displacement ( unit=(unit of potency) /
c*****               :                     (unit of x,y,z,depth)**2  )
c*****   uxx,uyx,uzx : x-derivative ( unit= unit of potency) /
c*****   uxy,uyy,uzy : y-derivative        (unit of x,y,z,depth)**3  )
c*****   uxz,uyz,uzz : z-derivative
c*****   iret        : return code  ( =0....normal,   =1....singular )
c
      integer i
      double precision ddip
      double complex xx,yy,zz,dd,pp1,pp2,pp3,pp4,du
      double complex aalpha
      double complex u(12),dua(12),dub(12),duc(12)
c
      double complex dummy(8),r,dumm(15)
      common /c1/dummy,r,dumm
c
      double complex f0
      data  f0/(0.d0,0.d0)/
c
      if(z.gt.0.d0)then
        stop ' Error in cdc3d0: positive z is given!'
      endif
      do 111 i=1,12
        u(i)=f0
        dua(i)=f0
        dub(i)=f0
        duc(i)=f0
  111 continue
      aalpha=alpha
      ddip=dip
      call dccon0(aalpha,ddip)
c======================================
c=====  real-source contribution  =====
c======================================
      xx=dcmplx(x,0.d0)
      yy=dcmplx(y,0.d0)
      zz=dcmplx(z,0.d0)
      dd=dcmplx(depth+z,0.d0)
      call dccon1(xx,yy,dd)
      if(r.eq.f0) go to 99
      pp1=pot1
      pp2=pot2
      pp3=pot3
      pp4=pot4
      call ua0(xx,yy,dd,pp1,pp2,pp3,pp4,dua)
c
      do 222 i=1,12
        if(i.lt.10) u(i)=u(i)-dua(i)
        if(i.ge.10) u(i)=u(i)+dua(i)
  222 continue
c=======================================
c=====  image-source contribution  =====
c=======================================
      dd=dcmplx(depth-z,0.d0)
      call dccon1(xx,yy,dd)
      call ua0(xx,yy,dd,pp1,pp2,pp3,pp4,dua)
      call ub0(xx,yy,dd,zz,pp1,pp2,pp3,pp4,dub)
      call uc0(xx,yy,dd,zz,pp1,pp2,pp3,pp4,duc)
c
      do 333 i=1,12
        du=dua(i)+dub(i)+zz*duc(i)
        if(i.ge.10) du=du+duc(i-9)
        u(i)=u(i)+du
  333 continue
c
      ux=u(1)
      uy=u(2)
      uz=u(3)
      uxx=u(4)
      uyx=u(5)
      uzx=u(6)
      uxy=u(7)
      uyy=u(8)
      uzy=u(9)
      uxz=u(10)
      uyz=u(11)
      uzz=u(12)
      iret=0
      return
c=======================================
c=====  in case of singular (r=0)  =====
c=======================================
   99 ux=f0
      uy=f0
      uz=f0
      uxx=f0
      uyx=f0
      uzx=f0
      uxy=f0
      uyy=f0
      uzy=f0
      uxz=f0
      uyz=f0
      uzz=f0
      iret=1
      return
      end
c===============================================================================
      subroutine  ua0(x,y,d,pp1,pp2,pp3,pp4,u)
      implicit none
      double complex x,y,d,pp1,pp2,pp3,pp4
      double complex u(12)
c
c********************************************************************
c*****    displacement and strain at depth (part-a)             *****
c*****    due to buried point source in a semiinfinite medium   *****
c********************************************************************
c
c***** input
c*****   x,y,d : station coordinates in fault system
c*****   pp1-pp4 : strike-, dip-, tensile- and inflate-potency
c***** output
c*****   u(12) : displacement and their derivatives
c
      integer i
      double complex du(12)
c
      double complex alp1,alp2,alp3,alp4,alp5
      double complex sd,cd,sdsd,cdcd,sdcd,s2d,c2d
      common /c0/alp1,alp2,alp3,alp4,alp5,sd,cd,sdsd,cdcd,sdcd,s2d,c2d
c
      double complex p,q,s,t,xy,x2,y2,d2,r,r2,r3,r5,qr,qrx,a3,a5,b3,c3
      double complex uy,vy,wy,uz,vz,wz
      common /c1/p,q,s,t,xy,x2,y2,d2,r,r2,r3,r5,qr,qrx,a3,a5,b3,c3,
     *           uy,vy,wy,uz,vz,wz
c
      double complex f0,f1,f3
      data f0,f1,f3/(0.d0,0.d0),(1.d0,0.d0),(3.d0,0.d0)/
c
      double complex cpi2
      data cpi2/(6.283185307179586d0,0.d0)/
c
      do 111  i=1,12
  111 u(i)=f0
c======================================
c=====  strike-slip contribution  =====
c======================================
      if(pp1.ne.f0) then
        du( 1)= alp1*q/r3    +alp2*x2*qr
        du( 2)= alp1*x/r3*sd +alp2*xy*qr
        du( 3)=-alp1*x/r3*cd +alp2*x*d*qr
        du( 4)= x*qr*(-alp1 +alp2*(f1+a5) )
        du( 5)= alp1*a3/r3*sd +alp2*y*qr*a5
        du( 6)=-alp1*a3/r3*cd +alp2*d*qr*a5
        du( 7)= alp1*(sd/r3-y*qr) +alp2*f3*x2/r5*uy
        du( 8)= f3*x/r5*(-alp1*y*sd +alp2*(y*uy+q) )
        du( 9)= f3*x/r5*( alp1*y*cd +alp2*d*uy )
        du(10)= alp1*(cd/r3+d*qr) +alp2*f3*x2/r5*uz
        du(11)= f3*x/r5*( alp1*d*sd +alp2*y*uz )
        du(12)= f3*x/r5*(-alp1*d*cd +alp2*(d*uz-q) )
        do 222 i=1,12
  222   u(i)=u(i)+pp1/cpi2*du(i)
      endif
c===================================
c=====  dip-slip contribution  =====
c===================================
      if(pp2.ne.f0) then
        du( 1)=            alp2*x*p*qr
        du( 2)= alp1*s/r3 +alp2*y*p*qr
        du( 3)=-alp1*t/r3 +alp2*d*p*qr
        du( 4)=                 alp2*p*qr*a5
        du( 5)=-alp1*f3*x*s/r5 -alp2*y*p*qrx
        du( 6)= alp1*f3*x*t/r5 -alp2*d*p*qrx
        du( 7)=                          alp2*f3*x/r5*vy
        du( 8)= alp1*(s2d/r3-f3*y*s/r5) +alp2*(f3*y/r5*vy+p*qr)
        du( 9)=-alp1*(c2d/r3-f3*y*t/r5) +alp2*f3*d/r5*vy
        du(10)=                          alp2*f3*x/r5*vz
        du(11)= alp1*(c2d/r3+f3*d*s/r5) +alp2*f3*y/r5*vz
        du(12)= alp1*(s2d/r3-f3*d*t/r5) +alp2*(f3*d/r5*vz-p*qr)
        do 333 i=1,12
  333   u(i)=u(i)+pp2/cpi2*du(i)
      endif
c========================================
c=====  tensile-fault contribution  =====
c========================================
      if(pp3.ne.f0) then
        du( 1)= alp1*x/r3 -alp2*x*q*qr
        du( 2)= alp1*t/r3 -alp2*y*q*qr
        du( 3)= alp1*s/r3 -alp2*d*q*qr
        du( 4)= alp1*a3/r3     -alp2*q*qr*a5
        du( 5)=-alp1*f3*x*t/r5 +alp2*y*q*qrx
        du( 6)=-alp1*f3*x*s/r5 +alp2*d*q*qrx
        du( 7)=-alp1*f3*xy/r5           -alp2*x*qr*wy
        du( 8)= alp1*(c2d/r3-f3*y*t/r5) -alp2*(y*wy+q)*qr
        du( 9)= alp1*(s2d/r3-f3*y*s/r5) -alp2*d*qr*wy
        du(10)= alp1*f3*x*d/r5          -alp2*x*qr*wz
        du(11)=-alp1*(s2d/r3-f3*d*t/r5) -alp2*y*qr*wz
        du(12)= alp1*(c2d/r3+f3*d*s/r5) -alp2*(d*wz-q)*qr
        do 444 i=1,12
  444   u(i)=u(i)+pp3/cpi2*du(i)
      endif
c=========================================
c=====  inflate source contribution  =====
c=========================================
      if(pp4.ne.f0) then
        du( 1)=-alp1*x/r3
        du( 2)=-alp1*y/r3
        du( 3)=-alp1*d/r3
        du( 4)=-alp1*a3/r3
        du( 5)= alp1*f3*xy/r5
        du( 6)= alp1*f3*x*d/r5
        du( 7)= du(5)
        du( 8)=-alp1*b3/r3
        du( 9)= alp1*f3*y*d/r5
        du(10)=-du(6)
        du(11)=-du(9)
        du(12)= alp1*c3/r3
        do 555 i=1,12
  555   u(i)=u(i)+pp4/cpi2*du(i)
      endif
      return
      end
c===============================================================================
      subroutine  ub0(x,y,d,z,pp1,pp2,pp3,pp4,u)
      implicit none
      double complex x,y,d,z,pp1,pp2,pp3,pp4
      double complex u(12)
c
c********************************************************************
c*****    displacement and strain at depth (part-b)             *****
c*****    due to buried point source in a semiinfinite medium   *****
c********************************************************************
c
c***** input
c*****   x,y,d,z : station coordinates in fault system
c*****   pp1-pp4 : strike-, dip-, tensile- and inflate-potency
c***** output
c*****   u(12) : displacement and their derivatives
c
      integer i
      double complex c,rd,d12,d32,d33,d53,d54,fi1,fi2,fi3,fi4,fi5
      double complex fj1,fj2,fj3,fj4,fk1,fk2,fk3
      double complex du(12)
c
      double complex alp1,alp2,alp3,alp4,alp5
      double complex sd,cd,sdsd,cdcd,sdcd,s2d,c2d
      common /c0/alp1,alp2,alp3,alp4,alp5,sd,cd,sdsd,cdcd,sdcd,s2d,c2d
c
      double complex p,q,s,t,xy,x2,y2,d2,r,r2,r3,r5,qr,qrx,a3,a5,b3,c3
      double complex uy,vy,wy,uz,vz,wz
      common /c1/p,q,s,t,xy,x2,y2,d2,r,r2,r3,r5,qr,qrx,a3,a5,b3,c3,
     *           uy,vy,wy,uz,vz,wz
c
      double complex f0,f1,f2,f3,f4,f5,f8,f9
      data f0,f1,f2,f3,f4,f5,f8,f9
     *        /(0.d0,0.d0),(1.d0,0.d0),(2.d0,0.d0),(3.d0,0.d0),
     *         (4.d0,0.d0),(5.d0,0.d0),(8.d0,0.d0),(9.d0,0.d0)/
c
      double complex cpi2
      data cpi2/(6.283185307179586d0,0.d0)/
c
      c=d+z
      rd=r+d
      d12=f1/(r*rd*rd)
      d32=d12*(f2*r+d)/r2
      d33=d12*(f3*r+d)/(r2*rd)
      d53=d12*(f8*r2+f9*r*d+f3*d2)/(r2*r2*rd)
      d54=d12*(f5*r2+f4*r*d+d2)/r3*d12
c
      fi1= y*(d12-x2*d33)
      fi2= x*(d12-y2*d33)
      fi3= x/r3-fi2
      fi4=-xy*d32
      fi5= f1/(r*rd)-x2*d32
      fj1=-f3*xy*(d33-x2*d54)
      fj2= f1/r3-f3*d12+f3*x2*y2*d54
      fj3= a3/r3-fj2
      fj4=-f3*xy/r5-fj1
      fk1=-y*(d32-x2*d53)
      fk2=-x*(d32-y2*d53)
      fk3=-f3*x*d/r5-fk2
c
      do 111  i=1,12
  111 u(i)=f0
c======================================
c=====  strike-slip contribution  =====
c======================================
      if(pp1.ne.f0) then
        du( 1)=-x2*qr  -alp3*fi1*sd
        du( 2)=-xy*qr  -alp3*fi2*sd
        du( 3)=-c*x*qr -alp3*fi4*sd
        du( 4)=-x*qr*(f1+a5) -alp3*fj1*sd
        du( 5)=-y*qr*a5      -alp3*fj2*sd
        du( 6)=-c*qr*a5      -alp3*fk1*sd
        du( 7)=-f3*x2/r5*uy      -alp3*fj2*sd
        du( 8)=-f3*xy/r5*uy-x*qr -alp3*fj4*sd
        du( 9)=-f3*c*x/r5*uy     -alp3*fk2*sd
        du(10)=-f3*x2/r5*uz  +alp3*fk1*sd
        du(11)=-f3*xy/r5*uz  +alp3*fk2*sd
        du(12)= f3*x/r5*(-c*uz +alp3*y*sd)
        do 222 i=1,12
  222   u(i)=u(i)+pp1/cpi2*du(i)
      endif
c===================================
c=====  dip-slip contribution  =====
c===================================
      if(pp2.ne.f0) then
        du( 1)=-x*p*qr +alp3*fi3*sdcd
        du( 2)=-y*p*qr +alp3*fi1*sdcd
        du( 3)=-c*p*qr +alp3*fi5*sdcd
        du( 4)=-p*qr*a5 +alp3*fj3*sdcd
        du( 5)= y*p*qrx +alp3*fj1*sdcd
        du( 6)= c*p*qrx +alp3*fk3*sdcd
        du( 7)=-f3*x/r5*vy      +alp3*fj1*sdcd
        du( 8)=-f3*y/r5*vy-p*qr +alp3*fj2*sdcd
        du( 9)=-f3*c/r5*vy      +alp3*fk1*sdcd
        du(10)=-f3*x/r5*vz -alp3*fk3*sdcd
        du(11)=-f3*y/r5*vz -alp3*fk1*sdcd
        du(12)=-f3*c/r5*vz +alp3*a3/r3*sdcd
        do 333 i=1,12
  333   u(i)=u(i)+pp2/cpi2*du(i)
      endif
c========================================
c=====  tensile-fault contribution  =====
c========================================
      if(pp3.ne.f0) then
        du( 1)= x*q*qr -alp3*fi3*sdsd
        du( 2)= y*q*qr -alp3*fi1*sdsd
        du( 3)= c*q*qr -alp3*fi5*sdsd
        du( 4)= q*qr*a5 -alp3*fj3*sdsd
        du( 5)=-y*q*qrx -alp3*fj1*sdsd
        du( 6)=-c*q*qrx -alp3*fk3*sdsd
        du( 7)= x*qr*wy     -alp3*fj1*sdsd
        du( 8)= qr*(y*wy+q) -alp3*fj2*sdsd
        du( 9)= c*qr*wy     -alp3*fk1*sdsd
        du(10)= x*qr*wz +alp3*fk3*sdsd
        du(11)= y*qr*wz +alp3*fk1*sdsd
        du(12)= c*qr*wz -alp3*a3/r3*sdsd
        do 444 i=1,12
  444   u(i)=u(i)+pp3/cpi2*du(i)
      endif
c=========================================
c=====  inflate source contribution  =====
c=========================================
      if(pp4.ne.f0) then
        du( 1)= alp3*x/r3
        du( 2)= alp3*y/r3
        du( 3)= alp3*d/r3
        du( 4)= alp3*a3/r3
        du( 5)=-alp3*f3*xy/r5
        du( 6)=-alp3*f3*x*d/r5
        du( 7)= du(5)
        du( 8)= alp3*b3/r3
        du( 9)=-alp3*f3*y*d/r5
        du(10)=-du(6)
        du(11)=-du(9)
        du(12)=-alp3*c3/r3
        do 555 i=1,12
  555   u(i)=u(i)+pp4/cpi2*du(i)
      endif
      return
      end
c===============================================================================
      subroutine  uc0(x,y,d,z,pp1,pp2,pp3,pp4,u)
      implicit none
      double complex x,y,d,z,pp1,pp2,pp3,pp4
      double complex u(12)
c
c********************************************************************
c*****    displacement and strain at depth (part-b)             *****
c*****    due to buried point source in a semiinfinite medium   *****
c********************************************************************
c
c***** input
c*****   x,y,d,z : station coordinates in fault system
c*****   pp1-pp4 : strike-, dip-, tensile- and inflate- potency
c***** output
c*****   u(12) : displacement and their derivatives
c
      integer i
      double complex c,q2,r7,a7,b5,b7,d7,c5,c7,qr5,qr7,dr5
      double complex du(12)
c
      double complex alp1,alp2,alp3,alp4,alp5
      double complex sd,cd,sdsd,cdcd,sdcd,s2d,c2d
      common /c0/alp1,alp2,alp3,alp4,alp5,sd,cd,sdsd,cdcd,sdcd,s2d,c2d
c
      double complex p,q,s,t,xy,x2,y2,d2,r,r2,r3,r5
      double complex qr,qrx,a3,a5,b3,c3,um(6)
      common /c1/p,q,s,t,xy,x2,y2,d2,r,r2,r3,r5,qr,qrx,a3,a5,b3,c3,um
c
      double complex f0,f1,f2,f3,f5,f7,f10,f15
      data f0,f1,f2,f3,f5,f7,f10,f15
     *        /(0.d0,0.d0),(1.d0,0.d0),(2.d0,0.d0),(3.d0,0.d0),
     *         (5.d0,0.d0),(7.d0,0.d0),(10.d0,0.d0),(15.d0,0.d0)/
c
      double complex cpi2
      data cpi2/(6.283185307179586d0,0.d0)/
c
      c=d+z
      q2=q*q
      r7=r5*r2
      a7=f1-f7*x2/r2
      b5=f1-f5*y2/r2
      b7=f1-f7*y2/r2
      c5=f1-f5*d2/r2
      c7=f1-f7*d2/r2
      d7=f2-f7*q2/r2
      qr5=f5*q/r2
      qr7=f7*q/r2
      dr5=f5*d/r2
c
      do 111  i=1,12
  111 u(i)=f0
c======================================
c=====  strike-slip contribution  =====
c======================================
      if( pp1.ne.f0) then
        du( 1)=-alp4*a3/r3*cd  +alp5*c*qr*a5
        du( 2)= f3*x/r5*( alp4*y*cd +alp5*c*(sd-y*qr5) )
        du( 3)= f3*x/r5*(-alp4*y*sd +alp5*c*(cd+d*qr5) )
        du( 4)= alp4*f3*x/r5*(f2+a5)*cd   -alp5*c*qrx*(f2+a7)
        du( 5)= f3/r5*( alp4*y*a5*cd +alp5*c*(a5*sd-y*qr5*a7) )
        du( 6)= f3/r5*(-alp4*y*a5*sd +alp5*c*(a5*cd+d*qr5*a7) )
        du( 7)= du(5)
        du( 8)= f3*x/r5*( alp4*b5*cd -alp5*f5*c/r2*(f2*y*sd+q*b7) )
        du( 9)= f3*x/r5*(-alp4*b5*sd +alp5*f5*c/r2*(d*b7*sd-y*c7*cd) )
        du(10)= f3/r5*   (-alp4*d*a5*cd +alp5*c*(a5*cd+d*qr5*a7) )
        du(11)= f15*x/r7*( alp4*y*d*cd  +alp5*c*(d*b7*sd-y*c7*cd) )
        du(12)= f15*x/r7*(-alp4*y*d*sd  +alp5*c*(f2*d*cd-q*c7) )
        do 222 i=1,12
  222   u(i)=u(i)+ pp1/cpi2*du(i)
      endif
c===================================
c=====  dip-slip contribution  =====
c===================================
      if( pp2.ne.f0) then
        du( 1)= alp4*f3*x*t/r5          -alp5*c*p*qrx
        du( 2)=-alp4/r3*(c2d-f3*y*t/r2) +alp5*f3*c/r5*(s-y*p*qr5)
        du( 3)=-alp4*a3/r3*sdcd         +alp5*f3*c/r5*(t+d*p*qr5)
        du( 4)= alp4*f3*t/r5*a5              -alp5*f5*c*p*qr/r2*a7
        du( 5)= f3*x/r5*(alp4*(c2d-f5*y*t/r2)
     *         -alp5*f5*c/r2*(s-y*p*qr7))
        du( 6)= f3*x/r5*(alp4*(f2+a5)*sdcd   -alp5*f5*c/r2*(t+d*p*qr7))
        du( 7)= du(5)
        du( 8)= f3/r5*(alp4*(f2*y*c2d+t*b5)
     *                               +alp5*c*(s2d-f10*y*s/r2-p*qr5*b7))
        du( 9)= f3/r5*(alp4*y*a5*sdcd-alp5*c*((f3+a5)*c2d
     *         +y*p*dr5*qr7))
        du(10)= f3*x/r5*(-alp4*(s2d-t*dr5) -alp5*f5*c/r2*(t+d*p*qr7))
        du(11)= f3/r5*(-alp4*(d*b5*c2d+y*c5*s2d)
     *        -alp5*c*((f3+a5)*c2d+y*p*dr5*qr7))
        du(12)= f3/r5*(-alp4*d*a5*sdcd
     *        -alp5*c*(s2d-f10*d*t/r2+p*qr5*c7))
        do 333 i=1,12
  333   u(i)=u(i)+ pp2/cpi2*du(i)
      endif
c========================================
c=====  tensile-fault contribution  =====
c========================================
      if( pp3.ne.f0) then
        du( 1)= f3*x/r5*(-alp4*s +alp5*(c*q*qr5-z))
        du( 2)= alp4/r3*(s2d-f3*y*s/r2)
     *         +alp5*f3/r5*(c*(t-y+y*q*qr5)-y*z)
        du( 3)=-alp4/r3*(f1-a3*sdsd)
     *         -alp5*f3/r5*(c*(s-d+d*q*qr5)-d*z)
        du( 4)=-alp4*f3*s/r5*a5 +alp5*(c*qr*qr5*a7-f3*z/r5*a5)
        du( 5)= f3*x/r5*(-alp4*(s2d-f5*y*s/r2)
     *         -alp5*f5/r2*(c*(t-y+y*q*qr7)-y*z))
        du( 6)= f3*x/r5*( alp4*(f1-(f2+a5)*sdsd)
     *         +alp5*f5/r2*(c*(s-d+d*q*qr7)-d*z))
        du( 7)= du(5)
        du( 8)= f3/r5*(-alp4*(f2*y*s2d+s*b5)
     *         -alp5*(c*(f2*sdsd+f10*y*(t-y)/r2-q*qr5*b7)+z*b5))
        du( 9)= f3/r5*( alp4*y*(f1-a5*sdsd)
     *         +alp5*(c*(f3+a5)*s2d-y*dr5*(c*d7+z)))
        du(10)= f3*x/r5*(-alp4*(c2d+s*dr5)
     *         +alp5*(f5*c/r2*(s-d+d*q*qr7)-f1-z*dr5))
        du(11)= f3/r5*( alp4*(d*b5*s2d-y*c5*c2d)
     *         +alp5*(c*((f3+a5)*s2d-y*dr5*d7)-y*(f1+z*dr5)))
        du(12)= f3/r5*(-alp4*d*(f1-a5*sdsd)
     *         -alp5*(c*(c2d+f10*d*(s-d)/r2-q*qr5*c7)+z*(f1+c5)))
        do 444 i=1,12
  444   u(i)=u(i)+ pp3/cpi2*du(i)
      endif
c=========================================
c=====  inflate source contribution  =====
c=========================================
      if( pp4.ne.f0) then
        du( 1)= alp4*f3*x*d/r5
        du( 2)= alp4*f3*y*d/r5
        du( 3)= alp4*c3/r3
        du( 4)= alp4*f3*d/r5*a5
        du( 5)=-alp4*f15*xy*d/r7
        du( 6)=-alp4*f3*x/r5*c5
        du( 7)= du(5)
        du( 8)= alp4*f3*d/r5*b5
        du( 9)=-alp4*f3*y/r5*c5
        du(10)= du(6)
        du(11)= du(9)
        du(12)= alp4*f3*d/r5*(f2+c5)
        do 555 i=1,12
  555   u(i)=u(i)+ pp4/cpi2*du(i)
      endif
      return
      end
c===============================================================================
      subroutine  dccon0(alpha,dip)
      implicit none
      double precision dip
      double complex alpha
c
c*******************************************************************
c*****   calculate medium constants and fault-dip constants    *****
c*******************************************************************
c
c***** input
c*****   alpha : medium constant  (lambda+myu)/(lambda+2*myu)
c*****   dip   : dip-angle (degree)
c### caution ### if cos(dip) is sufficiently small, it is set to zero
c
      double precision p18,sd0,cd0
c
      double complex alp1,alp2,alp3,alp4,alp5
      double complex sd,cd,sdsd,cdcd,sdcd,s2d,c2d
      common /c0/alp1,alp2,alp3,alp4,alp5,sd,cd,sdsd,cdcd,sdcd,s2d,c2d
c
      double precision eps,pi2
      double complex f0,f1,f2
      data eps,pi2/1.d-6,6.283185307179586d0/
      data f0,f1,f2/(0.d0,0.d0),(1.d0,0.d0),(2.d0,0.d0)/
c
      alp1=(f1-alpha)/f2
      alp2= alpha/f2
      alp3=(f1-alpha)/alpha
      alp4= f1-alpha
      alp5= alpha
c
      p18=pi2/360.d0
      sd0=dsin(dip*p18)
      sd=dcmplx(sd0,0.d0)
      cd0=dcos(dip*p18)
      cd=dcmplx(cd0,0.d0)
      if(dabs(cd0).lt.eps) then
        cd=f0
        if(sd0.gt.0.d0) sd= f1
        if(sd0.lt.0.d0) sd=-f1
      endif
      sdsd=sd*sd
      cdcd=cd*cd
      sdcd=sd*cd
      s2d=f2*sdcd
      c2d=cdcd-sdsd
      return
      end
c===============================================================================
      subroutine  dccon1(x,y,d)
      implicit none
      double complex x,y,d
c
c**********************************************************************
c*****   calculate station geometry constants for point source    *****
c**********************************************************************
c
c***** input
c*****   x,y,d : station coordinates in fault system
c### caution ### if x,y,d are sufficiently small, they are set to zero
c
      double complex r7
c
      double complex dummy(5),sd,cd,dumm(5)
      common /c0/dummy,sd,cd,dumm
c
      double complex p,q,s,t,xy,x2,y2,d2,r,r2,r3,r5,qr,qrx,a3,a5,b3,c3
      double complex uy,vy,wy,uz,vz,wz
      common /c1/p,q,s,t,xy,x2,y2,d2,r,r2,r3,r5,qr,qrx,a3,a5,b3,c3,
     *           uy,vy,wy,uz,vz,wz
c
      double precision eps
      double complex f0,f1,f3,f5
      data eps/1.d-6/
      data f0,f1,f3,f5/(0.d0,0.d0),(1.d0,0.d0),(3.d0,0.d0),(5.d0,0.d0)/
c
      if(cdabs(x).lt.eps) x=f0
      if(cdabs(y).lt.eps) y=f0
      if(cdabs(d).lt.eps) d=f0
      p=y*cd+d*sd
      q=y*sd-d*cd
      s=p*sd+q*cd
      t=p*cd-q*sd
      xy=x*y
      x2=x*x
      y2=y*y
      d2=d*d
      r2=x2+y2+d2
      r =cdsqrt(r2)
      if(r.eq.f0) return
      r3=r *r2
      r5=r3*r2
      r7=r5*r2
c
      a3=f1-f3*x2/r2
      a5=f1-f5*x2/r2
      b3=f1-f3*y2/r2
      c3=f1-f3*d2/r2
c
      qr=f3*q/r5
      qrx=f5*qr*x/r2
c
      uy=sd-f5*y*q/r2
      uz=cd+f5*d*q/r2
      vy=s -f5*y*p*q/r2
      vz=t +f5*d*p*q/r2
      wy=uy+sd
      wz=uz+cd
      return
      end
