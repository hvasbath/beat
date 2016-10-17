        subroutine cmemcpy(a,b,n)
        implicit none
c
c       copy complex array a to b
c
        integer n
        double complex a(n),b(n)
c
        integer i
c
        do i=1,n
          b(i)=a(i)
        enddo
        return
        end
