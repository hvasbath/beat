	subroutine psgsublay(ierr)
	implicit none
c
	integer ierr
c
	include 'psgglob.h'
c
c	work space
c
	integer i,i0,l
	double precision dh,dla,dmu,drho,detk,detm,dalf,z,dz
c
	n0=0
c
	do l=1,l0-1
	  dz=z2(l)-z1(l)
	  dla=2.d0*dabs(la2(l)-la1(l))/(la2(l)+la1(l))
	  dmu=2.d0*dabs(mu2(l)-mu1(l))/(mu2(l)+mu1(l))
        if(rho2(l)+rho1(l).gt.0.d0)then
	    drho=2.d0*dabs(rho2(l)-rho1(l))/(rho2(l)+rho1(l))
        else
          drho=0.d0
        endif
	  if(etk2(l)+etk1(l).gt.0.d0)then
          detk=2.d0*dabs(etk2(l)-etk1(l))/(etk2(l)+etk1(l))
        else
          detk=0.d0
        endif
	  if(etm2(l)+etm1(l).gt.0.d0)then
          detm=2.d0*dabs(etm2(l)-etm1(l))/(etm2(l)+etm1(l))
        else
          detm=0.d0
        endif
        if(alf2(l)+alf1(l).gt.0.d0)then
	    dalf=2.d0*dabs(alf2(l)-alf1(l))
     &            /(alf2(l)+alf1(l))
        else
          dalf=0.d0
        endif
	  i0=idnint(dmax1(1.d0,dla/reslm,dmu/reslm,drho/resld,
     &                  detk/reslv,detm/reslv,dalf/reslv))
	  dla=(la2(l)-la1(l))/dz
	  dmu=(mu2(l)-mu1(l))/dz
	  drho=(rho2(l)-rho1(l))/dz
	  detk=(etk2(l)-etk1(l))/dz
	  detm=(etm2(l)-etm1(l))/dz
	  dalf=(alf2(l)-alf1(l))/dz
	  dh=dz/dble(i0)
	  do i=1,i0
	    n0=n0+1
	    if(n0.ge.lmax)then
	      ierr=1
	      return
	    endif
	    h(n0)=dh
	    z=(dble(i)-0.5d0)*dh
	    la(n0)=la1(l)+dla*z
	    mu(n0)=mu1(l)+dmu*z
	    rho(n0)=rho1(l)+drho*z
	    etk(n0)=etk1(l)+detk*z
	    etm(n0)=etm1(l)+detm*z
	    alf(n0)=alf1(l)+dalf*z
	  enddo
	enddo
c
c	last layer is half-space
c
	n0=n0+1
	h(n0)=0.d0
	la(n0)=la1(l0)
	mu(n0)=mu1(l0)
	rho(n0)=rho1(l0)
	etk(n0)=etk1(l0)
	etm(n0)=etm1(l0)
	alf(n0)=alf1(l0)
c
	write(*,'(8a)')'  no',' thick(m)    ','  la(Pa)    ',
     &    '  mu(Pa)    ','rho(kg/m^3) ','  etk(Pa*s) ',
     &    '  etm(Pa*s) ','   alpha'
	do i=1,n0
	  write(*,1001)i,h(i),la(i),mu(i),
     &               rho(i),etk(i),etm(i),alf(i)
        elastic(i)=(etk(i).le.0.d0.or.alf(i).ge.1.d0).and.
     &             etm(i).le.0.d0
	enddo
1001	format(i4,f11.4,6E12.4)
	ierr=0
	return
	end
