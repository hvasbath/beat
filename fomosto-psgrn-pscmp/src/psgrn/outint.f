	subroutine outint(unit,ivar,n)
	implicit none
c
	integer unit,n
	integer ivar(n)
c
	integer i,j,k,m,me
c
	do i=1,n
	  if(ivar(i).ge.0)then
	    write(unit,'(a,$)')' '
	  else
	    write(unit,'(a,$)')' -'
	  endif
	  m=iabs(ivar(i))
	  if(m.eq.0)then
	    me=1
	  else
	    me=idint(dlog10(0.1d0+dble(m)))+1
	  endif
	  do j=1,me
	    k=m/10**(me-j)
	    if(i.eq.n.and.j.eq.me)then
	      write(unit,'(i1)')k
	    else
	      write(unit,'(i1,$)')k
	    endif
	    m=m-k*10**(me-j)
	  enddo
	enddo
c
	return
	end
	  
