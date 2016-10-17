      double precision function mscorr(st1,di1,ra1,st2,di2,ra2)
      implicit none
c
c     corralation between two source mechanism (st,di and ra in degree)
c
c     input:
c
      double precision st1,di1,ra1,st2,di2,ra2
c
c     local memories:
c
      integer i,j
      double precision ncorr,tcorr
      double precision st(2),di(2),ra(2),ns(3,2),ts(3,2),rst(3),rdi(3)
c
      double precision PI,DEG2RAD
      data PI,DEG2RAD/3.14159265358979d0,1.745329252d-02/
c
      st(1)=st1*DEG2RAD
      di(1)=di1*DEG2RAD
      ra(1)=ra1*DEG2RAD
      st(2)=st2*DEG2RAD
      di(2)=di2*DEG2RAD
      ra(2)=ra2*DEG2RAD
c
      do j=1,2
        ns(1,j)=dsin(di(j))*dcos(st(j)+0.5d0*PI)
        ns(2,j)=dsin(di(j))*dsin(st(j)+0.5d0*PI)
        ns(3,j)=-dcos(di(j))
c
        rst(1)=dcos(st(j))
        rst(2)=dsin(st(j))
        rst(3)=0.d0
c
        rdi(1)=dcos(di(j))*dcos(st(j)+0.5d0*PI)
        rdi(2)=dcos(di(j))*dsin(st(j)+0.5d0*PI)
        rdi(3)=dsin(di(j))
c
        do i=1,3
          ts(i,j)=rst(i)*dcos(ra(j))-rdi(i)*dsin(ra(j))
        enddo
      enddo
      ncorr=0.d0
      tcorr=0.d0
      do i=1,3
        ncorr=ncorr+ns(i,1)*ns(i,2)
        tcorr=tcorr+ts(i,1)*ts(i,2)
      enddo
      mscorr=ncorr*tcorr
      return
      end