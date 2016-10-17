c*******************************************************************************
c*******************************************************************************
        subroutine memcpy(a,b,n)
        implicit none
c
c       copy real array a to b
c
        integer n
        double precision a(n),b(n)
c
        integer i
c
        do i=1,n
          b(i)=a(i)
        enddo
        return
        end
