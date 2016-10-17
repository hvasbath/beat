      subroutine cdsvd500(ca,cb,n,m,eps,key)
      implicit none
c-------------------------------------------------------------------------------
c     Solve complex linear equation system by single-value decomposiztion      i
c     method (modified from ludcmp and lubksb in the <Numerical Recipies>      i
c     ca: coefficient matrix(n,n);                                             i
c     cb: right-hand matrix(n,m) by input,                                     i
c         solution matrix(n,m) by return;                                      i
c     cunit: unit of the culomn vectors                                        i
c     eps: control constant;                                                   i
c     key: if the main term of a column is                                     i
c          smaller than eps, key=0: anormal return,                            i
c          else key=1: normal return.                                          i
c                                                                              i
c     Note: n <= 500 will be NOT CHECKED!                                      i
c-------------------------------------------------------------------------------
      integer n,m,key
      double precision eps
      double complex ca(n,n),cb(n,m)
c
      integer NMAX
      parameter (NMAX=500)
      integer i,ii,imax,j,k,ll
      integer indx(NMAX)
      double precision aamax,dum
      double precision vv(NMAX)
      double complex cdum,csum
c
      do i=1,n
        aamax=0.d0
        do j=1,n
          aamax=dmax1(aamax,cdabs(ca(i,j)))
        enddo
        if(aamax.le.eps)then
          key=0
          return
        endif
        vv(i)=1.d0/aamax
      enddo
      do j=1,n
        do i=1,j-1
          csum=ca(i,j)
          do k=1,i-1
            csum=csum-ca(i,k)*ca(k,j)
          enddo
          ca(i,j)=csum
        enddo
        aamax=0.d0
        do i=j,n
          csum=ca(i,j)
          do k=1,j-1
            csum=csum-ca(i,k)*ca(k,j)
          enddo
          ca(i,j)=csum
          dum=vv(i)*cdabs(csum)
          if(dum.ge.aamax) then
            imax=i
            aamax=dum
          endif
        enddo
        if(j.ne.imax) then
          do k=1,n
            cdum=ca(imax,k)
            ca(imax,k)=ca(j,k)
            ca(j,k)=cdum
          enddo
          vv(imax)=vv(j)
        endif
        indx(j)=imax
        if(cdabs(ca(j,j)).le.eps)then
          key=0
          return
        endif
        if(j.ne.n) then
          cdum=(1.d0,0.d0)/ca(j,j)
          do i=j+1,n
            ca(i,j)=ca(i,j)*cdum
          enddo
        endif
      enddo
c
      do k=1,m
        ii=0
        do i=1,n
          ll=indx(i)
          csum=cb(ll,k)
          cb(ll,k)=cb(i,k)
          if(ii.ne.0) then
            do j=ii,i-1
              csum=csum-ca(i,j)*cb(j,k)
            enddo
          else if(cdabs(csum).ne.0.d0)then
            ii=i
          endif
          cb(i,k)=csum
        enddo
        do i=n,1,-1
          csum=cb(i,k)
          do j=i+1,n
            csum=csum-ca(i,j)*cb(j,k)
          enddo
          cb(i,k)=csum/ca(i,i)
        enddo
      enddo
      key=1
      return
      end
