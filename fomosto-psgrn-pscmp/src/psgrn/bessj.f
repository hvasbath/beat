      double precision function bessj(n,x)
      implicit none
      integer n
      double precision x
c
      integer iacc
      parameter(iacc=40)
      double precision bigno,bigni
      parameter(bigno=1.d+10,bigni=1.d-10)
c
      integer j,jsum,m
      double precision ax,bj,bjm,bjp,sum,tox,bessj0,bessj1
      if(n.lt.2)pause 'bad argument n in bessj'
      ax=dabs(x)
      if(ax.eq.0.d0)then
        bessj=0.d0
      else if(ax.gt.dble(n))then
        tox=2.d0/ax
        bjm=bessj0(ax)
        bj=bessj1(ax)
        do 11 j=1,n-1
          bjp=dble(j)*tox*bj-bjm
          bjm=bj
          bj=bjp
11      continue
        bessj=bj
      else
        tox=2.d0/ax
        m=2*((n+idint(dsqrt(dble(iacc*n))))/2)
        bessj=0.d0
        jsum=0
        sum=0.d0
        bjp=0.d0
        bj=1.d0
        do 12 j=m,1,-1
          bjm=dble(j)*tox*bj-bjp
          bjp=bj
          bj=bjm
          if(dabs(bj).gt.bigno)then
            bj=bj*bigni
            bjp=bjp*bigni
            bessj=bessj*bigni
            sum=sum*bigni
          endif
          if(jsum.ne.0)sum=sum+bj
          jsum=1-jsum
          if(j.eq.n)bessj=bjp
12      continue
        sum=2.d0*sum-bj
        bessj=bessj/sum
      endif
      if(x.lt.0.d0.and.mod(n,2).eq.1)bessj=-bessj
      return
      end
