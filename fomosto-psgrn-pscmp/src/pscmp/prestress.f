      subroutine prestress(s1,s2,s3,strike,dip,rake,p,f,
     &                     cmb,sxx,syy,szz,sxy,syz,szx)
      implicit none
c
c     determine regional stress tensor using the known principal
c     stresses and master fault mechanism, assuming that the master
c     fault follows the Coulomb failure criterion.
c
c     input:
c     principal stresses, master fault strike, dip and rake,
c     pore pressure, friction coefficient
c
      double precision s1,s2,s3,strike,dip,rake,p,f
c
c     output
c     max. preseismic Coulomb stress, prestress tensor
c
      double precision cmb,sxx,syy,szz,sxy,syz,szx
c
c     local memories:
c
      integer i,j,k
      double precision pi,alpha,st,di,ra,deg2rad,cmb1,cmb2,cmb3
      double precision ns(3),ts(3),rst(3),rdi(3)
      double precision sig(3),s(3,3),rot(3,3)
c
      cmb=0.d0
      sxx=0.d0
      syy=0.d0
      szz=0.d0
      sxy=0.d0
      syz=0.d0
      szx=0.d0
      if(s1.eq.0.d0.and.s2.eq.0.d0.and.s3.eq.0.d0)return
c
      cmb1=0.5d0*dabs(s2-s3)*dsqrt(1+f*f)+f*(0.5d0*(s2+s3)+p)
      cmb2=0.5d0*dabs(s3-s1)*dsqrt(1+f*f)+f*(0.5d0*(s3+s1)+p)
      cmb3=0.5d0*dabs(s1-s2)*dsqrt(1+f*f)+f*(0.5d0*(s1+s2)+p)
c
      cmb=dmax1(cmb1,cmb2,cmb3)
      if(s1.eq.s2.and.s2.eq.s3)return
c
      if(cmb.eq.cmb1)then
        sig(3)=s1
        sig(1)=dmax1(s2,s3)
        sig(2)=dmin1(s2,s3)
      else if(cmb.eq.cmb2)then
        sig(1)=dmax1(s3,s1)
        sig(2)=dmin1(s3,s1)
        sig(3)=s2
      else
        sig(1)=dmax1(s1,s2)
        sig(2)=dmin1(s1,s2)
        sig(3)=s3
      endif
c
c     determine principal stress orientation
c
      pi=4.d0*datan(1.d0)
      alpha=0.5d0*datan2(1.d0,f)
      deg2rad=pi/180.d0
      st=strike*deg2rad
      di=dip*deg2rad
      ra=rake*deg2rad
c
      ns(1)=dsin(di)*dcos(st+0.5d0*pi)
      ns(2)=dsin(di)*dsin(st+0.5d0*pi)
      ns(3)=-dcos(di)
c
      rst(1)=dcos(st)
      rst(2)=dsin(st)
      rst(3)=0.d0
c
      rdi(1)=dcos(di)*dcos(st+0.5d0*pi)
      rdi(2)=dcos(di)*dsin(st+0.5d0*pi)
      rdi(3)=dsin(di)
c
      do i=1,3
        ts(i)=rst(i)*dcos(ra)-rdi(i)*dsin(ra)
      enddo
c
      do i=1,3
        rot(i,1)=ns(i)*dcos(alpha)+ts(i)*dsin(alpha)
        rot(i,2)=ns(i)*dsin(alpha)-ts(i)*dcos(alpha)
      enddo
      rot(1,3)=ts(2)*ns(3)-ts(3)*ns(2)
      rot(2,3)=ts(3)*ns(1)-ts(1)*ns(3)
      rot(3,3)=ts(1)*ns(2)-ts(2)*ns(1)
c
      do j=1,3
        do i=1,j
          s(i,j)=0.d0
          do k=1,3
            s(i,j)=s(i,j)+sig(k)*rot(i,k)*rot(j,k)
          enddo
        enddo
      enddo
      sxx=s(1,1)
      syy=s(2,2)
      szz=s(3,3)
      sxy=s(1,2)
      syz=s(2,3)
      szx=s(1,3)
      return
      end