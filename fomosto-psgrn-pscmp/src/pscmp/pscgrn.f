      subroutine pscgrn(neq,ns,nrec,ntr,onlysc,nsc)
      implicit none
c
c     Last modified: Potsdam, Feb, 2002, by R. Wang
c
      integer neq,ns,nrec,ntr,nsc
      logical onlysc
c
      include 'pscglob.h'
c~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
c     GREEN'S FUNNCTIONN PARAMETERS
c     =============================
c
c     Green's function source types:
c       1 = explosion (m11=m22=m33=kappa)
c       2 = strike-slip (m12=m21=mue)
c       3 = dip-slip (m13=m31=mue)
c       4 = compensated linear vector dipole (m33=mue, m11=m22=-m33/2)
c     Green's function coordinate system:
c       (z,r,t) = cylindrical with z being downward(!)
c~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      integer unit(14,4)
      integer idec(NRMAX),igrns(NTMAX,NRMAX)
      double precision cogrns(NRMAX,14,4),grns(NTMAX,NRMAX,14,4)
      double precision r(NRMAX)
      character*163 greens(14,4)
      logical select(14,4)
c~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
c     LOCAL WORK SPACES
c     =================
c~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      integer i,j,n,irec,ips,isc,ir,ieq,is,izs,nzs1,nzs2,nsmall,nlarge
      integer it,itr,itstart,lend,lenf,id1,id2,istp,npsum
      double precision si,co,si2,co2,expo,tmax,wei
      double precision dis,dise,disn,azi,ur,ut,uz
      double precision szz,srr,stt,szr,srt,stz
      double precision tr,tt,rot,gd,gr
      double precision dr,dract,dt
      double precision zs,las,mus,rhos,etks,etms,alfs
      double precision psss,shss,psds,shds,pscl,psep
      double precision d1,d2,d3,d4,d5,d6,d7
      character*180 dataline
c
c     OPEN GREEN'S FUNCTIONS FILES
c     ============================
c
      do lend=80,1,-1
        if(grndir(lend:lend).ne.' ')goto 100
      enddo
100   continue
      do i=1,14
        do lenf=80,1,-1
          if(green(i)(lenf:lenf).ne.' ')goto 110
        enddo
110     continue
        greens(i,1)=grndir(1:lend)//green(i)(1:lenf)//'.ep'
        greens(i,2)=grndir(1:lend)//green(i)(1:lenf)//'.ss'
        greens(i,3)=grndir(1:lend)//green(i)(1:lenf)//'.ds'
        greens(i,4)=grndir(1:lend)//green(i)(1:lenf)//'.cl'
      enddo
c
      do istp=1,4
        do i=1,14
          select(i,istp)=.true.
        enddo
      enddo
      select(3,1)=.false.
      select(8,1)=.false.
      select(9,1)=.false.
      select(11,1)=.false.
      select(12,1)=.false.
      select(3,4)=.false.
      select(8,4)=.false.
      select(9,4)=.false.
      select(11,4)=.false.
      select(12,4)=.false.
c
      do istp=1,4
        do i=1,14
          if(.not.select(i,istp))goto 120
          unit(i,istp)=10+14*(istp-1)+i
          open(unit(i,istp),file=greens(i,istp),status='old')
          if(i*istp.eq.1)then
            call getdata(unit(i,istp),dataline)
            read(dataline,*)nr,r1,r2,sampratio
            if(nr.gt.NRMAX)then
              stop 'srror: NRMAX too small defined!'
            endif
            call getdata(unit(i,istp),dataline)
            read(dataline,*)zrec,larec,murec,rhorec,etkrec,
     &                      etmrec,alfrec
            call getdata(unit(i,istp),dataline)
            read(dataline,*)nzs,zs1,zs2
            if(nzs.gt.NZSMAX)then
              stop 'srror: NZSMAX too small defined!'
            endif
            call getdata(unit(i,istp),dataline)
            read(dataline,*)nt,twindow
            if(nt.gt.NTMAX)then
              stop 'srror: NTMAX too small defined!'
            endif
          else
            call getdata(unit(i,istp),dataline)
            read(dataline,*)n,d1,d2,d3
            if(n.ne.nr.or.d1.ne.r1.or.d2.ne.r2.or.d3.ne.sampratio)then
              stop 'srror: different observation sampling in Greens!'
            endif
            call getdata(unit(i,istp),dataline)
            read(dataline,*)d1,d2,d3,d4,d5,d6,d7
            if(d1.ne.zrec.or.d2.ne.larec.or.
     &         d3.ne.murec.or.d4.ne.rhorec.or.
     &         d5.ne.etkrec.or.d6.ne.etmrec.or.
     &         d7.ne.alfrec)then
              stop 'srror: diff. observation site parameters in Greens!'
            endif
            call getdata(unit(i,istp),dataline)
            read(dataline,*)n,d1,d2
            if(n.ne.nzs.or.d1.ne.zs1.or.d2.ne.zs2)then
              stop 'srror: different source sampling in Greens!'
            endif
            call getdata(unit(i,istp),dataline)
            read(dataline,*)n,d1
            if(n.ne.nt.or.d1.ne.twindow)then
              stop 'srror: different time sampling in Greens!'
            endif
          endif
120       continue
        enddo
      enddo
c
c     all Green's function files have been opened
c     ===========================================
c
      if(nt.gt.1)then
        dt=twindow/dble(nt-1)
      else
        dt=twindow
      endif
      if(nr.eq.2)then
        r(1)=r1
        r(2)=r2
        dr=r2-1
      else
        dr=2.d0*(r2-r1)/dble(nr-1)/(1.d0+sampratio)
        r(1)=r1
        do ir=2,nr
          dract=dr*(1.d0+(sampratio-1.d0)*dble(ir-2)/dble(nr-2))
          r(ir)=r(ir-1)+dract
        enddo
      endif
c
c     INITIALIZATION
c     ==============
c
      if(onlysc)then
        tmax=0.d0
        do isc=1,nsc
          tmax=dmax1(tmax,tsc(isc))
        enddo
      else
        tmax=twindow
      endif
c
      if(onlysc)then
        ntr=0
        do isc=1,nsc
          it=min0(1+idint(tsc(isc)/dt),nt)
          if(ntr.eq.0)then
            ntr=ntr+1
            itsc(ntr)=it
          else
            do itr=1,ntr
              if(it.eq.itsc(itr))goto 130
            enddo
            ntr=ntr+1
            itsc(ntr)=it
          endif
130       it=min0(it+1,nt)
          do itr=1,ntr
            if(it.eq.itsc(itr))goto 140
          enddo
          ntr=ntr+1
          itsc(ntr)=it
140       continue
        enddo
      else
        ntr=nt
      endif
      if(ntr.gt.NTRMAX)then
        stop ' srror: NTRMAX defined too small!'
      endif
c
c     DISCRETISATION OF RECTANGULAR PLANE SOURCES
c     ===========================================
c
      call pscdisc(ns,npsum,tmax)
c
c     SUPERPOSITION OF ALL DISCRETE POINT SOURCES
c     ===========================================
c
      print *,'... superposition of all discrete point sources ...'
      nzs1=1
      nzs2=0
      do izs=1,nzs
        if(nps(izs).gt.0)then
          nzs2=izs
        endif
      enddo
c
      do i=1,14
        do irec=1,nrec
          do ieq=1,neq
            coobs(ieq,irec,i)=0.d0
          enddo
          do itr=1,ntr
            obs(itr,irec,i)=0.d0
          enddo
        enddo
      enddo
c
      call pscokada(ns,nrec,tmax)
c
      do izs=nzs1,nzs2
        nsmall=0
        nlarge=0
c
c       read in Green's functions
c
        do istp=1,4
          do i=1,14
            if(select(i,istp))then
              if(i.eq.1)then
                call getdata(unit(i,istp),dataline)
                read(dataline,*)zs,las,mus,rhos,etks,etms,alfs
              else
                call getdata(unit(i,istp),dataline)
                read(dataline,*)d1,d2,d3,d4,d5,d6,d7
                if(d1.ne.zs.or.d2.ne.las.or.
     &             d3.ne.mus.or.d4.ne.rhos.or.
     &             d5.ne.etks.or.d6.ne.etms.or.
     &             d7.ne.alfs)then
                  stop 'srror: different s. layer parameters in greens!'
                endif
              endif
              read(unit(i,istp),*)(cogrns(ir,i,istp),ir=1,nr)
              read(unit(i,istp),*)(idec(ir),ir=1,nr)
              if(nt.gt.1)then
                read(unit(i,istp),*)((igrns(it,ir),ir=1,nr),it=2,nt)
              endif
              do ir=1,nr
                grns(1,ir,i,istp)=0.d0
                expo=10.d0**idec(ir)
                do it=2,nt
                  grns(it,ir,i,istp)=dble(igrns(it,ir))*expo
                enddo
              enddo
            endif
          enddo
        enddo
        if(nps(izs).ge.1)then
c
          write(*,'(a,i4,a,F12.4,a)')' processing ',nps(izs),
     &           ' source patches at depth ',zs/KM2M,' km'
        endif
c
        do ips=1,nps(izs)
          is=isno(ips,izs)
          itstart=idnint(tstart(is)/dt)
          do irec=1,nrec
            call disazi(REARTH,plat(ips,izs),plon(ips,izs),
     &                  latrec(irec),lonrec(irec),disn,dise)
            dis=dsqrt(disn**2+dise**2)
            if(dis.gt.r(nr))then
c             print *,' Warning: too large distances ignored!'
              nlarge=nlarge+1
            else
              id1=1
              do ir=1,nr
                if(dis.ge.r(ir))id1=ir
              enddo
              id2=min0(id1+1,nr)
              if(dis.gt.0.d0)then
                azi=datan2(dise,disn)
              else
                azi=0.d0
              endif
              if(id1.le.0)then
                id1=1
                id2=1
                d1=1.d0
                d2=0.d0
              else if(id1.lt.nr)then
                dract=r(id2)-r(id1)
                d2=(dis-r(id1))/dract
                d1=1.d0-d2
                id2=id1+1
              else
                d1=1.d0
                d2=0.d0
                id2=id1
              endif
c
              co=dcos(azi)
              si=dsin(azi)
              co2=dcos(2.d0*azi)
              si2=dsin(2.d0*azi)
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c             pmwei(1-6):
c             1 = weight for strike-slip: m12=m21=1;
c             poloidal*sin(2 * theta), toroidal*cos(2 * theta)
c
c             2 = weight for dip-slip: m13=m31=1
c             poloidal * cos(theta), toroidal * sin(theta)
c
c             3 = weight for clvd: m33=-m11=-m22=1
c             axisymmetric
c
c             4 = weight for 45 deg strike-slip: m11=-m22=1
c             greenfct4(theta) = green1(theta + 45 deg)
c
c             5 = weight for 45 deg dip-slip: m23=m32=1
c             greenfct5(theta) = green2(theta - 90 deg)
c
c             6 = weight for explosion
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
              psep=pmwei(6,ips,izs)
              psss=pmwei(1,ips,izs)*si2+pmwei(4,ips,izs)*co2
              shss=pmwei(1,ips,izs)*co2-pmwei(4,ips,izs)*si2
              psds=pmwei(2,ips,izs)*co+pmwei(5,ips,izs)*si
              shds=pmwei(2,ips,izs)*si-pmwei(5,ips,izs)*co
              pscl=pmwei(3,ips,izs)
c
c             coseismic responses
c
              uz=0.d0
              ur=0.d0
              ut=0.d0
              rot=0.d0
              szz=0.d0
              srr=0.d0
              stt=0.d0
              szr=0.d0
              srt=0.d0
              stz=0.d0
              tr=0.d0
              tt=0.d0
              gd=0.d0
              gr=0.d0
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c             contributions from the explosion components
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
              uz=uz+psep*(d1*cogrns(id1,1,1)+d2*cogrns(id2,1,1))
              ur=ur+psep*(d1*cogrns(id1,2,1)+d2*cogrns(id2,2,1))
c
              szz=szz+psep*(d1*cogrns(id1,4,1)+d2*cogrns(id2,4,1))
              srr=srr+psep*(d1*cogrns(id1,5,1)+d2*cogrns(id2,5,1))
              stt=stt+psep*(d1*cogrns(id1,6,1)+d2*cogrns(id2,6,1))
              szr=szr+psep*(d1*cogrns(id1,7,1)+d2*cogrns(id2,7,1))
c
              tr=tr+psep*(d1*cogrns(id1,10,1)+d2*cogrns(id2,10,1))
c
              gd=gd+psep*(d1*cogrns(id1,13,1)+d2*cogrns(id2,13,1))
              gr=gr+psep*(d1*cogrns(id1,14,1)+d2*cogrns(id2,14,1))
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c             contributions from the strike-slip components
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
              uz=uz+psss*(d1*cogrns(id1,1,2)+d2*cogrns(id2,1,2))
              ur=ur+psss*(d1*cogrns(id1,2,2)+d2*cogrns(id2,2,2))
              ut=ut+shss*(d1*cogrns(id1,3,2)+d2*cogrns(id2,3,2))
c
              szz=szz+psss*(d1*cogrns(id1,4,2)+d2*cogrns(id2,4,2))
              srr=srr+psss*(d1*cogrns(id1,5,2)+d2*cogrns(id2,5,2))
              stt=stt+psss*(d1*cogrns(id1,6,2)+d2*cogrns(id2,6,2))
              szr=szr+psss*(d1*cogrns(id1,7,2)+d2*cogrns(id2,7,2))
              srt=srt+shss*(d1*cogrns(id1,8,2)+d2*cogrns(id2,8,2))
              stz=stz+shss*(d1*cogrns(id1,9,2)+d2*cogrns(id2,9,2))
c
              tr=tr+psss*(d1*cogrns(id1,10,2)+d2*cogrns(id2,10,2))
              tt=tt+shss*(d1*cogrns(id1,11,2)+d2*cogrns(id2,11,2))
c
              rot=rot
     &           +shss*(d1*cogrns(id1,12,2)+d2*cogrns(id2,12,2))
c
              gd=gd+psss*(d1*cogrns(id1,13,2)+d2*cogrns(id2,13,2))
              gr=gr+psss*(d1*cogrns(id1,14,2)+d2*cogrns(id2,14,2))
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c             contributions from the dip-slip components
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
              uz=uz+psds*(d1*cogrns(id1,1,3)+d2*cogrns(id2,1,3))
              ur=ur+psds*(d1*cogrns(id1,2,3)+d2*cogrns(id2,2,3))
              ut=ut+shds*(d1*cogrns(id1,3,3)+d2*cogrns(id2,3,3))
c
              szz=szz+psds*(d1*cogrns(id1,4,3)+d2*cogrns(id2,4,3))
              srr=srr+psds*(d1*cogrns(id1,5,3)+d2*cogrns(id2,5,3))
              stt=stt+psds*(d1*cogrns(id1,6,3)+d2*cogrns(id2,6,3))
              szr=szr+psds*(d1*cogrns(id1,7,3)+d2*cogrns(id2,7,3))
              srt=srt+shds*(d1*cogrns(id1,8,3)+d2*cogrns(id2,8,3))
              stz=stz+shds*(d1*cogrns(id1,9,3)+d2*cogrns(id2,9,3))
c
              tr=tr+psds*(d1*cogrns(id1,10,3)+d2*cogrns(id2,10,3))
              tt=tt+shds*(d1*cogrns(id1,11,3)+d2*cogrns(id2,11,3))
c
              rot=rot
     &           +shds*(d1*cogrns(id1,12,3)+d2*cogrns(id2,12,3))
c
              gd=gd+psds*(d1*cogrns(id1,13,3)+d2*cogrns(id2,13,3))
              gr=gr+psds*(d1*cogrns(id1,14,3)+d2*cogrns(id2,14,3))
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c             contributions from the clvd components
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
              uz=uz+pscl*(d1*cogrns(id1,1,4)+d2*cogrns(id2,1,4))
              ur=ur+pscl*(d1*cogrns(id1,2,4)+d2*cogrns(id2,2,4))
c
              szz=szz+pscl*(d1*cogrns(id1,4,4)+d2*cogrns(id2,4,4))
              srr=srr+pscl*(d1*cogrns(id1,5,4)+d2*cogrns(id2,5,4))
              stt=stt+pscl*(d1*cogrns(id1,6,4)+d2*cogrns(id2,6,4))
              szr=szr+pscl*(d1*cogrns(id1,7,4)+d2*cogrns(id2,7,4))
c
              tr=tr+pscl*(d1*cogrns(id1,10,4)+d2*cogrns(id2,10,4))
c
              gd=gd+pscl*(d1*cogrns(id1,13,4)+d2*cogrns(id2,13,4))
              gr=gr+pscl*(d1*cogrns(id1,14,4)+d2*cogrns(id2,14,4))
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
              ieq=ieqno(is)
              coobs(ieq,irec,1)=coobs(ieq,irec,1)+ur*co-ut*si
              coobs(ieq,irec,2)=coobs(ieq,irec,2)+ur*si+ut*co
              coobs(ieq,irec,3)=coobs(ieq,irec,3)+uz
c
              coobs(ieq,irec,4)=coobs(ieq,irec,4)+srr*co*co
     &                                       +stt*si*si-srt*si2
              coobs(ieq,irec,5)=coobs(ieq,irec,5)+srr*si*si
     &                                       +stt*co*co+srt*si2
              coobs(ieq,irec,6)=coobs(ieq,irec,6)+szz
              coobs(ieq,irec,7)=coobs(ieq,irec,7)+0.5d0*(srr-stt)*si2
     &                                       +srt*co2
              coobs(ieq,irec,8)=coobs(ieq,irec,8)+szr*si+stz*co
              coobs(ieq,irec,9)=coobs(ieq,irec,9)+szr*co-stz*si
c
              coobs(ieq,irec,10)=coobs(ieq,irec,10)+tr*co-tt*si
              coobs(ieq,irec,11)=coobs(ieq,irec,11)+tr*si+tt*co
c
              coobs(ieq,irec,12)=coobs(ieq,irec,12)+rot
c
              coobs(ieq,irec,13)=coobs(ieq,irec,13)+gd
              coobs(ieq,irec,14)=coobs(ieq,irec,14)+gr
c
c             postseismic responses just before each earthquake
c
              do ieq=1,neq
                if(tstart(is).ge.eqtime(ieq))goto 200
                do itr=1,2
                  if(itr.eq.1)then
                    it=min0(1+idint((eqtime(ieq)-tstart(is))/dt),nt)
                    wei=1.d0-dmod((eqtime(ieq)-tstart(is))/dt,1.d0)
                  else
                    it=min0(2+idint((eqtime(ieq)-tstart(is))/dt),nt)
                    wei=dmod((eqtime(ieq)-tstart(is))/dt,1.d0)
                  endif
                  uz=0.d0
                  ur=0.d0
                  ut=0.d0
                  rot=0.d0
                  szz=0.d0
                  srr=0.d0
                  stt=0.d0
                  szr=0.d0
                  srt=0.d0
                  stz=0.d0
                  tr=0.d0
                  tt=0.d0
                  gd=0.d0
                  gr=0.d0
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c                 contributions from the explosion components
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
                  uz=uz+psep*(d1*grns(it,id1,1,1)+d2*grns(it,id2,1,1))
                  ur=ur+psep*(d1*grns(it,id1,2,1)+d2*grns(it,id2,2,1))
c
                  szz=szz+psep*(d1*grns(it,id1,4,1)+d2*grns(it,id2,4,1))
                  srr=srr+psep*(d1*grns(it,id1,5,1)+d2*grns(it,id2,5,1))
                  stt=stt+psep*(d1*grns(it,id1,6,1)+d2*grns(it,id2,6,1))
                  szr=szr+psep*(d1*grns(it,id1,7,1)+d2*grns(it,id2,7,1))
c
                  tr=tr+psep*(d1*grns(it,id1,10,1)+d2*grns(it,id2,10,1))
c
                  gd=gd+psep*(d1*grns(it,id1,13,1)+d2*grns(it,id2,13,1))
                  gr=gr+psep*(d1*grns(it,id1,14,1)+d2*grns(it,id2,14,1))
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c                 contributions from the strike-slip components
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
                  uz=uz+psss*(d1*grns(it,id1,1,2)+d2*grns(it,id2,1,2))
                  ur=ur+psss*(d1*grns(it,id1,2,2)+d2*grns(it,id2,2,2))
                  ut=ut+shss*(d1*grns(it,id1,3,2)+d2*grns(it,id2,3,2))
c
                  szz=szz+psss*(d1*grns(it,id1,4,2)+d2*grns(it,id2,4,2))
                  srr=srr+psss*(d1*grns(it,id1,5,2)+d2*grns(it,id2,5,2))
                  stt=stt+psss*(d1*grns(it,id1,6,2)+d2*grns(it,id2,6,2))
                  szr=szr+psss*(d1*grns(it,id1,7,2)+d2*grns(it,id2,7,2))
                  srt=srt+shss*(d1*grns(it,id1,8,2)+d2*grns(it,id2,8,2))
                  stz=stz+shss*(d1*grns(it,id1,9,2)+d2*grns(it,id2,9,2))
c
                  tr=tr+psss*(d1*grns(it,id1,10,2)+d2*grns(it,id2,10,2))
                  tt=tt+shss*(d1*grns(it,id1,11,2)+d2*grns(it,id2,11,2))
c
                  rot=rot
     &               +shss*(d1*grns(it,id1,12,2)+d2*grns(it,id2,12,2))
c
                  gd=gd+psss*(d1*grns(it,id1,13,2)+d2*grns(it,id2,13,2))
                  gr=gr+psss*(d1*grns(it,id1,14,2)+d2*grns(it,id2,14,2))
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c                 contributions from the dip-slip components
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
                  uz=uz+psds*(d1*grns(it,id1,1,3)+d2*grns(it,id2,1,3))
                  ur=ur+psds*(d1*grns(it,id1,2,3)+d2*grns(it,id2,2,3))
                  ut=ut+shds*(d1*grns(it,id1,3,3)+d2*grns(it,id2,3,3))
c
                  szz=szz+psds*(d1*grns(it,id1,4,3)+d2*grns(it,id2,4,3))
                  srr=srr+psds*(d1*grns(it,id1,5,3)+d2*grns(it,id2,5,3))
                  stt=stt+psds*(d1*grns(it,id1,6,3)+d2*grns(it,id2,6,3))
                  szr=szr+psds*(d1*grns(it,id1,7,3)+d2*grns(it,id2,7,3))
                  srt=srt+shds*(d1*grns(it,id1,8,3)+d2*grns(it,id2,8,3))
                  stz=stz+shds*(d1*grns(it,id1,9,3)+d2*grns(it,id2,9,3))
c
                  tr=tr+psds*(d1*grns(it,id1,10,3)+d2*grns(it,id2,10,3))
                  tt=tt+shds*(d1*grns(it,id1,11,3)+d2*grns(it,id2,11,3))
c
                  rot=rot
     &               +shds*(d1*grns(it,id1,12,3)+d2*grns(it,id2,12,3))
c
                  gd=gd+psds*(d1*grns(it,id1,13,3)+d2*grns(it,id2,13,3))
                  gr=gr+psds*(d1*grns(it,id1,14,3)+d2*grns(it,id2,14,3))
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c               contributions from the clvd components
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
                  uz=uz+pscl*(d1*grns(it,id1,1,4)+d2*grns(it,id2,1,4))
                  ur=ur+pscl*(d1*grns(it,id1,2,4)+d2*grns(it,id2,2,4))
c
                  szz=szz+pscl*(d1*grns(it,id1,4,4)+d2*grns(it,id2,4,4))
                  srr=srr+pscl*(d1*grns(it,id1,5,4)+d2*grns(it,id2,5,4))
                  stt=stt+pscl*(d1*grns(it,id1,6,4)+d2*grns(it,id2,6,4))
                  szr=szr+pscl*(d1*grns(it,id1,7,4)+d2*grns(it,id2,7,4))
c
                  tr=tr+pscl*(d1*grns(it,id1,10,4)+d2*grns(it,id2,10,4))
c
                  gd=gd+pscl*(d1*grns(it,id1,13,4)+d2*grns(it,id2,13,4))
                  gr=gr+pscl*(d1*grns(it,id1,14,4)+d2*grns(it,id2,14,4))
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
                  poobs(ieq,irec,1)=poobs(ieq,irec,1)+wei*(ur*co-ut*si)
                  poobs(ieq,irec,2)=poobs(ieq,irec,2)+wei*(ur*si+ut*co)
                  poobs(ieq,irec,3)=poobs(ieq,irec,3)+wei*uz
c
                  poobs(ieq,irec,4)=poobs(ieq,irec,4)
     &                             +wei*(srr*co*co+stt*si*si-srt*si2)
                  poobs(ieq,irec,5)=poobs(ieq,irec,5)
     &                             +wei*(srr*si*si+stt*co*co+srt*si2)
                  poobs(ieq,irec,6)=poobs(ieq,irec,6)+wei*szz
                  poobs(ieq,irec,7)=poobs(ieq,irec,7)
     &                             +wei*(0.5d0*(srr-stt)*si2+srt*co2)
                  poobs(ieq,irec,8)=poobs(ieq,irec,8)
     &                             +wei*(szr*si+stz*co)
                  poobs(ieq,irec,9)=poobs(ieq,irec,9)
     &                             +wei*(szr*co-stz*si)
c
                  poobs(ieq,irec,10)=poobs(ieq,irec,10)
     &                              +wei*(tr*co-tt*si)
                  poobs(ieq,irec,11)=poobs(ieq,irec,11)
     &                              +wei*(tr*si+tt*co)
c
                  poobs(ieq,irec,12)=poobs(ieq,irec,12)+wei*rot
c
                  poobs(ieq,irec,13)=poobs(ieq,irec,13)+wei*gd
                  poobs(ieq,irec,14)=poobs(ieq,irec,14)+wei*gr
                enddo
200             continue
              enddo
c
c             postseismic responses
c
              do itr=1,ntr
                if(onlysc)then
                  it=itsc(itr)-itstart
                else
                  it=itr-itstart
                endif
                if(it.le.0)goto 300
                uz=0.d0
                ur=0.d0
                ut=0.d0
                rot=0.d0
                szz=0.d0
                srr=0.d0
                stt=0.d0
                szr=0.d0
                srt=0.d0
                stz=0.d0
                tr=0.d0
                tt=0.d0
                gd=0.d0
                gr=0.d0
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c                 contributions from the explosion components
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
                uz=uz+psep*(d1*grns(it,id1,1,1)+d2*grns(it,id2,1,1))
                ur=ur+psep*(d1*grns(it,id1,2,1)+d2*grns(it,id2,2,1))
c
                szz=szz+psep*(d1*grns(it,id1,4,1)+d2*grns(it,id2,4,1))
                srr=srr+psep*(d1*grns(it,id1,5,1)+d2*grns(it,id2,5,1))
                stt=stt+psep*(d1*grns(it,id1,6,1)+d2*grns(it,id2,6,1))
                szr=szr+psep*(d1*grns(it,id1,7,1)+d2*grns(it,id2,7,1))
c
                tr=tr+psep*(d1*grns(it,id1,10,1)+d2*grns(it,id2,10,1))
c
                gd=gd+psep*(d1*grns(it,id1,13,1)+d2*grns(it,id2,13,1))
                gr=gr+psep*(d1*grns(it,id1,14,1)+d2*grns(it,id2,14,1))
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c                 contributions from the strike-slip components
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
                uz=uz+psss*(d1*grns(it,id1,1,2)+d2*grns(it,id2,1,2))
                ur=ur+psss*(d1*grns(it,id1,2,2)+d2*grns(it,id2,2,2))
                ut=ut+shss*(d1*grns(it,id1,3,2)+d2*grns(it,id2,3,2))
c
                szz=szz+psss*(d1*grns(it,id1,4,2)+d2*grns(it,id2,4,2))
                srr=srr+psss*(d1*grns(it,id1,5,2)+d2*grns(it,id2,5,2))
                stt=stt+psss*(d1*grns(it,id1,6,2)+d2*grns(it,id2,6,2))
                szr=szr+psss*(d1*grns(it,id1,7,2)+d2*grns(it,id2,7,2))
                srt=srt+shss*(d1*grns(it,id1,8,2)+d2*grns(it,id2,8,2))
                stz=stz+shss*(d1*grns(it,id1,9,2)+d2*grns(it,id2,9,2))
c
                tr=tr+psss*(d1*grns(it,id1,10,2)+d2*grns(it,id2,10,2))
                tt=tt+shss*(d1*grns(it,id1,11,2)+d2*grns(it,id2,11,2))
c
                rot=rot
     &             +shss*(d1*grns(it,id1,12,2)+d2*grns(it,id2,12,2))
c
                gd=gd+psss*(d1*grns(it,id1,13,2)+d2*grns(it,id2,13,2))
                gr=gr+psss*(d1*grns(it,id1,14,2)+d2*grns(it,id2,14,2))
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c               contributions from the dip-slip components
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
                uz=uz+psds*(d1*grns(it,id1,1,3)+d2*grns(it,id2,1,3))
                ur=ur+psds*(d1*grns(it,id1,2,3)+d2*grns(it,id2,2,3))
                ut=ut+shds*(d1*grns(it,id1,3,3)+d2*grns(it,id2,3,3))
c
                szz=szz+psds*(d1*grns(it,id1,4,3)+d2*grns(it,id2,4,3))
                srr=srr+psds*(d1*grns(it,id1,5,3)+d2*grns(it,id2,5,3))
                stt=stt+psds*(d1*grns(it,id1,6,3)+d2*grns(it,id2,6,3))
                szr=szr+psds*(d1*grns(it,id1,7,3)+d2*grns(it,id2,7,3))
                srt=srt+shds*(d1*grns(it,id1,8,3)+d2*grns(it,id2,8,3))
                stz=stz+shds*(d1*grns(it,id1,9,3)+d2*grns(it,id2,9,3))
c
                tr=tr+psds*(d1*grns(it,id1,10,3)+d2*grns(it,id2,10,3))
                tt=tt+shds*(d1*grns(it,id1,11,3)+d2*grns(it,id2,11,3))
c
                rot=rot
     &             +shds*(d1*grns(it,id1,12,3)+d2*grns(it,id2,12,3))
c
                gd=gd+psds*(d1*grns(it,id1,13,3)+d2*grns(it,id2,13,3))
                gr=gr+psds*(d1*grns(it,id1,14,3)+d2*grns(it,id2,14,3))
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c               contributions from the clvd components
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
                uz=uz+pscl*(d1*grns(it,id1,1,4)+d2*grns(it,id2,1,4))
                ur=ur+pscl*(d1*grns(it,id1,2,4)+d2*grns(it,id2,2,4))
c
                szz=szz+pscl*(d1*grns(it,id1,4,4)+d2*grns(it,id2,4,4))
                srr=srr+pscl*(d1*grns(it,id1,5,4)+d2*grns(it,id2,5,4))
                stt=stt+pscl*(d1*grns(it,id1,6,4)+d2*grns(it,id2,6,4))
                szr=szr+pscl*(d1*grns(it,id1,7,4)+d2*grns(it,id2,7,4))
c
                tr=tr+pscl*(d1*grns(it,id1,10,4)+d2*grns(it,id2,10,4))
c
                gd=gd+pscl*(d1*grns(it,id1,13,4)+d2*grns(it,id2,13,4))
                gr=gr+pscl*(d1*grns(it,id1,14,4)+d2*grns(it,id2,14,4))
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
                obs(itr,irec,1)=obs(itr,irec,1)+ur*co-ut*si
                obs(itr,irec,2)=obs(itr,irec,2)+ur*si+ut*co
                obs(itr,irec,3)=obs(itr,irec,3)+uz
c
                obs(itr,irec,4)=obs(itr,irec,4)+srr*co*co
     &                                           +stt*si*si-srt*si2
                obs(itr,irec,5)=obs(itr,irec,5)+srr*si*si
     &                                           +stt*co*co+srt*si2
                obs(itr,irec,6)=obs(itr,irec,6)+szz
                obs(itr,irec,7)=obs(itr,irec,7)+0.5d0*(srr-stt)*si2
     &                                           +srt*co2
                obs(itr,irec,8)=obs(itr,irec,8)+szr*si+stz*co
                obs(itr,irec,9)=obs(itr,irec,9)+szr*co-stz*si
c
                obs(itr,irec,10)=obs(itr,irec,10)+tr*co-tt*si
                obs(itr,irec,11)=obs(itr,irec,11)+tr*si+tt*co
c
                obs(itr,irec,12)=obs(itr,irec,12)+rot
c
                obs(itr,irec,13)=obs(itr,irec,13)+gd
                obs(itr,irec,14)=obs(itr,irec,14)+gr
300             continue
              enddo
            endif
          enddo
        enddo
        if(nsmall.gt.0)then
          nwarn=nwarn+nsmall
          write(*,'(a,i5,a)')' Warning: ',nsmall,'too small distances'
     &                     //' exceed the Green-function coverage!'
        endif
        if(nlarge.gt.0)then
          nwarn=nwarn+nlarge
          write(*,'(a,i5,a)')' Warning: ',nlarge,' too large distances'
     &                     //' exceed the Green-function coverage!'
        endif
      enddo
c
      do istp=1,4
        do i=1,13
          if(select(i,istp))close(unit(i,istp))
        enddo
      enddo
      return
      end
