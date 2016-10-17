      subroutine cmbopt(sxx,syy,szz,sxy,syz,szx,p,f,key,
     &                  st0,di0,ra0,
     &                  cmb,sig,st1,di1,ra1,st2,di2,ra2)
      implicit none
c
c     Coulomb stress with the optimal orientation
c
c     input:
c     stress tensor, pore pressure, friction coefficient
c     key = 0: determine optimal Coulomb stress only;
c           1: determine optimal Coulomb stress and orientations
c
      integer key
      double precision sxx,syy,szz,sxy,syz,szx,p,f
c
c     output
c     max. Coulomb stress at the two optimally oriented fault planes
c     sig = normal stress
c
      double precision st0,di0,ra0,cmb,sig,st1,di1,ra1,st2,di2,ra2
c
c     local memories:
c
      integer i,j,j0,j1,j2,jmin,jmax
      double precision pi,b,c,d,s1,s2,s3,snn,alpha,am,swap
      double precision cmb1,cmb2,cmb3,det1,det2,det3,detmax,rmax
      double precision s(3),r(3,3),ns(3,2),ts(3,2)
      double precision mscorr
c
      pi=4.d0*datan(1.d0)
c
      if(sxy.eq.0.d0.and.syz.eq.0.d0.and.szx.eq.0.d0)then
        s(1)=sxx
        s(2)=syy
        s(3)=szz
      else
        b=-(sxx+syy+szz)
        c=sxx*syy+syy*szz+szz*sxx-sxy**2-syz**2-szx**2
        d=sxx*syz**2+syy*szx**2+szz*sxy**2-2.d0*sxy*syz*szx-sxx*syy*szz
        call roots3(b,c,d,s)
      endif
c
      cmb1=0.5d0*dabs(s(2)-s(3))*dsqrt(1+f*f)+f*(0.5d0*(s(2)+s(3))+p)
      cmb2=0.5d0*dabs(s(3)-s(1))*dsqrt(1+f*f)+f*(0.5d0*(s(3)+s(1))+p)
      cmb3=0.5d0*dabs(s(1)-s(2))*dsqrt(1+f*f)+f*(0.5d0*(s(1)+s(2))+p)
c
      cmb=dmax1(cmb1,cmb2,cmb3)
      st1=0.d0
      di1=0.d0
      ra1=0.d0
      st2=0.d0
      di2=0.d0
      ra2=0.d0
      if(key.eq.0.or.s(1).eq.s(2).and.s(2).eq.s(3))return
c
      if(cmb.eq.cmb1)then
        s3=s(1)
        s1=dmax1(s(2),s(3))
        s2=dmin1(s(2),s(3))
      else if(cmb.eq.cmb2)then
        s1=dmax1(s(3),s(1))
        s2=dmin1(s(3),s(1))
        s3=s(2)
      else
        s1=dmax1(s(1),s(2))
        s2=dmin1(s(1),s(2))
        s3=s(3)
      endif
      sig=0.5d0*((s1-s2)*f/dsqrt(1+f*f)+s1+s2)
      s(1)=s1
      s(2)=s2
      s(3)=s3
c
c     determine eigenvectors (the principal stress directions)
c
      j0=0
      if(s(1).eq.s(2))then
        j0=3
        j1=1
        j2=2
      else if(s(2).eq.s(3))then
        j0=1
        j1=2
        j2=3
      else if(s(3).eq.s(1))then
        j0=2
        j1=1
        j2=3
      endif
c
      if(j0.eq.0)then
        jmin=1
        jmax=3
      else
        jmin=j0
        jmax=j0
        print *,' Warning: more than two optimal rupture orientations!'
      endif
c
      do j=jmin,jmax
        det1=syz*syz-(syy-s(j))*(szz-s(j))
        det2=szx*szx-(sxx-s(j))*(szz-s(j))
        det3=sxy*sxy-(sxx-s(j))*(syy-s(j))
        detmax=dmax1(dabs(det1),dabs(det2),dabs(det3))
        if(dabs(det1).eq.detmax)then
          r(1,j)=det1
          r(2,j)=(szz-s(j))*sxy-syz*szx
          r(3,j)=(syy-s(j))*szx-syz*sxy
        else if(dabs(det2).eq.detmax)then
          r(1,j)=(szz-s(j))*sxy-szx*syz
          r(2,j)=det2
          r(3,j)=(sxx-s(j))*syz-szx*sxy
        else
          r(1,j)=(syy-s(j))*szx-sxy*syz
          r(2,j)=(sxx-s(j))*syz-sxy*szx
          r(3,j)=det3
        endif
      enddo
c
c     if any two eigenvalues are identical, their corresponding
c     eigenvectors should be redetermined by orthogonalizing
c     them to the 3. eigenvector as well as to each other
c     
      if(j0.gt.0)then
        rmax=dmax1(dabs(r(1,j0)),dabs(r(2,j0)),dabs(r(3,j0)))
        if(dabs(r(1,j0)).eq.rmax)then
          r(1,j1)=-r(2,j0)
          r(2,j1)=r(1,j0)
          r(3,j1)=0.d0
c
          r(1,j2)=-r(3,j0)
          r(2,j2)=0.d0
          r(3,j2)=r(1,j0)
          am=r(1,j1)*r(1,j2)/(r(1,j1)**2+r(2,j1)**2)
          do i=1,3
            r(i,j2)=r(i,j2)-am*r(i,j1)
          enddo
        else if(dabs(r(2,j0)).eq.rmax)then
          r(1,j1)=r(2,j0)
          r(2,j1)=-r(1,j0)
          r(3,j1)=0.d0
c
          r(1,j2)=0.d0
          r(2,j2)=-r(3,j0)
          r(3,j2)=r(2,j0)
          am=r(2,j1)*r(2,j2)/(r(1,j1)**2+r(2,j1)**2)
          do i=1,3
            r(i,j2)=r(i,j2)-am*r(i,j1)
          enddo
        else if(dabs(r(3,j0)).eq.rmax)then
          r(1,j1)=r(3,j0)
          r(2,j1)=0.d0
          r(3,j1)=-r(1,j0)
c
          r(1,j2)=0.d0
          r(2,j2)=r(3,j0)
          r(3,j2)=-r(2,j0)
          am=r(3,j1)*r(3,j2)/(r(1,j1)**2+r(3,j1)**2)
          do i=1,3
            r(i,j2)=r(i,j2)-am*r(i,j1)
          enddo
        endif
      endif
c
      do j=1,3
        am=dsqrt(r(1,j)**2+r(2,j)**2+r(3,j)**2)
        do i=1,3
          r(i,j)=r(i,j)/am
        enddo
      enddo
c
      alpha=0.5d0*datan2(1.d0,f)
      snn=s(1)*dcos(alpha)**2+s(2)*dsin(alpha)**2
c
c     determine the two optimal fault-plane normals
c
      do i=1,3
        ns(i,1)=r(i,1)*dcos(alpha)+r(i,2)*dsin(alpha)          
        ns(i,2)=r(i,1)*dcos(alpha)-r(i,2)*dsin(alpha)
      enddo
c
c     determine the direction of max. shear stress
c
      do j=1,2
        am=dsqrt(ns(1,j)**2+ns(2,j)**2+ns(3,j)**2)
        if(ns(3,j).gt.0.d0)am=-am
        do i=1,3
          ns(i,j)=ns(i,j)/am
        enddo
        ts(1,j)=(sxx-snn)*ns(1,j)+sxy*ns(2,j)+szx*ns(3,j)
        ts(2,j)=sxy*ns(1,j)+(syy-snn)*ns(2,j)+syz*ns(3,j)
        ts(3,j)=szx*ns(1,j)+syz*ns(2,j)+(szz-snn)*ns(3,j)
        am=dsqrt(ts(1,j)**2+ts(2,j)**2+ts(3,j)**2)
        do i=1,3
          ts(i,j)=ts(i,j)/am
        enddo
      enddo
c
c     determine the two optimal focal mechanisms
c    
      st1=dmod(datan2(ns(2,1),ns(1,1))*180.d0/pi+270.d0,360.d0)
	di1=dacos(-ns(3,1))*180.d0/pi
      s1=dcos(st1*pi/180.d0)
      s2=dsin(st1*pi/180.d0)
      ra1=dacos(dmin1(dmax1(s1*ts(1,1)+s2*ts(2,1),-1.d0),1.d0))
     &   *180.d0/pi
      if(ts(3,1).gt.0.d0)ra1=-ra1
c
      st2=dmod(datan2(ns(2,2),ns(1,2))*180.d0/pi+270.d0,360.d0)
	di2=dacos(-ns(3,2))*180.d0/pi
      s1=dcos(st2*pi/180.d0)
      s2=dsin(st2*pi/180.d0)
      ra2=dacos(dmin1(dmax1(s1*ts(1,2)+s2*ts(2,2),-1.d0),1.d0))
     &   *180.d0/pi
      if(ts(3,2).gt.0.d0)ra2=-ra2
c
      if(mscorr(st0,di0,ra0,st1,di1,ra1).lt.
     &   mscorr(st0,di0,ra0,st2,di2,ra2))then
        swap=st1
        st1=st2
        st2=swap
        swap=di1
        di1=di2
        di2=swap
        swap=ra1
        ra1=ra2
        ra2=swap
      endif
      return
      end