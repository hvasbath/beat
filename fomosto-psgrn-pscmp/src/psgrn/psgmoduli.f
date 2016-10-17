      subroutine psgmoduli(cs,istate)
      implicit none
c
      integer istate
      double complex cs
c
      include 'psgglob.h'
c
      integer n
      double precision kappa,r2d3
      double complex c2d3,cmuu,cmur,cetk,cetm
c
      r2d3=2.d0/3.d0
      c2d3=dcmplx(2.d0/3.d0,0.d0)
      do n=1,n0
        if(elastic(n).or.istate.eq.-1)then
c         Elastic body
          cmu(n)=dcmplx(mu(n),0.d0)
          cla(n)=dcmplx(la(n),0.d0)
        else
          kappa=la(n)+mu(n)*r2d3
          cmuu=dcmplx(mu(n),0.d0)
          cetm=dcmplx(etm(n),0.d0)
          if(etk(n).le.0.d0.or.alf(n).ge.1.d0)then
c           Maxwell body
            cmur=dcmplx(mu(n)*relaxmin,0.d0)
            cmuu=cmuu-cmur
            cmu(n)=cmur+cmuu*cetm*cs/(cmuu+cetm*cs)
          else
            cetk=dcmplx(etk(n),0.d0)
            cmur=dcmplx(mu(n)*alf(n)/(1.d0-alf(n)),0.d0)
            if(etm(n).le.0.d0)then
c             Standard-Linear-Solid
              cmu(n)=cmuu*(cmur+cetk*cs)/(cmuu+cmur+cetk*cs)
            else
c             Burgers body
              cmu(n)=dcmplx(mu(n)*relaxmin,0.d0)
     &              +cmuu*cetm*cs*(cmur+cetk*cs)
     &              /(cmuu*cetm*cs+(cmur+cetk*cs)*(cmuu+cetm*cs))
            endif
          endif
c         no bulk relaxation!
          cla(n)=dcmplx(kappa,0.d0)-cmu(n)*c2d3
        endif
      enddo
      return
      end