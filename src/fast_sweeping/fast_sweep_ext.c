#define NPY_NO_DEPRECATED_API 7
#include "Python.h"
#include "numpy/arrayobject.h"
#include <numpy/npy_math.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

typedef npy_float64 float64_t;

static PyObject *FastSweepExtError;

int good_array(PyObject* o, int typenum, npy_intp size_want, int ndim_want, npy_intp* shape_want) {
    int i;

    if (!PyArray_Check(o)) {
        PyErr_SetString(FastSweepExtError, "not a NumPy array" );
	return 0;
    }

    if (PyArray_TYPE((PyArrayObject*)o) != typenum) {
        PyErr_SetString(FastSweepExtError, "array of unexpected type");
	return 0;     }

    if (!PyArray_ISCARRAY((PyArrayObject*)o)) {
        PyErr_SetString(FastSweepExtError, "array is not contiguous or not well behaved");
	return 0;     }

    if (size_want != -1 && size_want != PyArray_SIZE((PyArrayObject*)o)) {
        PyErr_SetString(FastSweepExtError, "array is of unexpected size");
	return 0;     }

    if (ndim_want != -1 && ndim_want != PyArray_NDIM((PyArrayObject*)o)) {
        PyErr_SetString(FastSweepExtError, "array is of unexpected ndim");
	return 0;     }

    if (ndim_want != -1 && shape_want != NULL) {
        for (i=0; i<ndim_want; i++) {
	    if (shape_want[i] != -1 && shape_want[i] != PyArray_DIMS((PyArrayObject*)o)[i]) {
	        PyErr_SetString(FastSweepExtError, "array is of unexpected shape");
		return 0;
	    }
	}
    }
    return 1;
} 

void Vect_from_Mat(npy_intp VectPos[1], npy_intp sel_row, npy_intp sel_col, npy_intp colNum)
{  //goes row by row 
   VectPos[0] = sel_row*colNum + sel_col;
   //the vectors are given, starting at zero, the colNum is given in actual numbers of columns (starting with 1!)
   return;
}

void fast_sweep(float64_t *Slowness, float64_t *StartTime, float64_t PatchSize, npy_intp HypoInStk, npy_intp HypoInDip, npy_intp NumInStk, npy_intp NumInDip)
{
    int num_iter;
    npy_intp i, j, ii;
    npy_intp PatchNum;
    npy_intp VectPos[1];
    
    
    float64_t epsilon  = 0.1;
    float64_t err      = 1.0E+6; //high dummy value;
    float64_t NewVal[1];
    
    float64_t *Time_old;
    
    num_iter = 0;
    PatchNum = NumInStk*NumInDip;

    Time_old = (float64_t *) malloc((size_t) ((PatchNum)*sizeof(float32_t)));   

    int cnt = 0;
    for (i = 0; i < NumInStk; i++)
    {   for (j = 0; j < NumInDip; j++)
        {   Vect_from_Mat(VectPos, i, j, NumInDip);
            StartTime[ VectPos[0] ] = +INFINITY;     
            cnt++;
    }   }
    Vect_from_Mat(VectPos, HypoInStk, HypoInDip, NumInDip);
    StartTime[ VectPos[0] ] = 0.0;
    
    //----------------------
    while (err > epsilon)
    {   for (i = 0; i < NumInStk; i++)
        {   for (j = 0; j < NumInDip; j++)
            {   Vect_from_Mat(VectPos, i, j, NumInDip);
                Time_old[ VectPos[0] ] = StartTime[ VectPos[0] ];   
        }   }
        //----------------------
        for (ii = 0; ii < 4; ii++) 
        {   if      (ii == 0)
            {   for (i = 0; i < NumInStk; i++)
                {   for (j = 0; j < NumInDip; j++)
                    {   
                        upwind(NewVal, StartTime, i,j, Slowness, PatchSize, NumInStk, NumInDip);
                        Vect_from_Mat(VectPos, i, j, NumInDip);
                        StartTime[VectPos[0]] = NewVal[0];        
            }   }   } 
            else if (ii == 1)  
            {   for (i = (NumInStk-1); i >= 0; i--)
                {   for (j = 0; j < NumInDip; j++)
                    {   
                        upwind(NewVal, StartTime, i,j, Slowness, PatchSize, NumInStk, NumInDip);
                        Vect_from_Mat(VectPos, i, j, NumInDip);
                        StartTime[VectPos[0]] = NewVal[0];  
            }   }   }
            else if (ii == 2)  
            {   for (i = (NumInStk-1); i >= 0; i--)
                {   for (j = (NumInDip-1); j >= 0; j--)
                    {   
                        upwind(NewVal, StartTime, i,j, Slowness, PatchSize, NumInStk, NumInDip);
                        Vect_from_Mat(VectPos, i, j, NumInDip);
                        StartTime[VectPos[0]] = NewVal[0];  
            }   }   }
            else if (ii == 3)  
            {   for (i = 0; i < NumInStk; i++)
                {   for (j = (NumInDip-1); j >= 0; j--)
                    {   upwind(NewVal, StartTime, i,j, Slowness, PatchSize, NumInStk, NumInDip);
                        Vect_from_Mat(VectPos, i, j, NumInDip);
                        StartTime[VectPos[0]] = NewVal[0];  
            }   }   } 
        }
        //----------------------
        err = 0.0;
        for (i = 0; i < PatchNum; i++)
        {   
            err += pow((StartTime[i]-Time_old[i]),2.0);   
        }
        num_iter++;
    }
    
    return;
}

void upwind(float64_t NewVal[1], float64_t *StartTime, npy_intp i, npy_intp j, float64_t *Slowness, float64_t PatchSize, npy_intp NumInStk, npy_intp NumInDip)
{   
    npy_intp     i1, i2, j1, j2;
    npy_intp     VectPos0[1];
    npy_intp     VectPos1[1], VectPos2[1];
    npy_intp     VectPos3[1], VectPos4[1];
    float64_t    u_xmin, u_ymin;
    
    i1 = i - 1;      i2 = i + 1;    
    j1 = j - 1;      j2 = j + 1;    
    
    if (i1 < 0)     {       i1 = 0;     }           if (i2 >= NumInStk)     {       i2 = NumInStk-1;     }
    if (j1 < 0)     {       j1 = 0;     }           if (j2 >= NumInDip)     {       j2 = NumInDip-1;     }

    Vect_from_Mat(VectPos0, i,  j, NumInDip);
    Vect_from_Mat(VectPos1, i1, j, NumInDip);
    Vect_from_Mat(VectPos2, i2, j, NumInDip);
    Vect_from_Mat(VectPos3, i, j1, NumInDip);
    Vect_from_Mat(VectPos4, i, j2, NumInDip);

    u_xmin = (StartTime[VectPos1[0]] < StartTime[VectPos2[0]]) ?  StartTime[VectPos1[0]] : StartTime[VectPos2[0]];
    u_ymin = (StartTime[VectPos3[0]] < StartTime[VectPos4[0]]) ?  StartTime[VectPos3[0]] : StartTime[VectPos4[0]];

    eq_solve(NewVal, StartTime[VectPos0[0]], u_xmin, u_ymin, Slowness[VectPos0[0]],PatchSize);

    NewVal[0] = (NewVal[0] <  StartTime[VectPos0[0]]) ? NewVal[0] :  StartTime[VectPos0[0]];
    return;    
}

void eq_solve(float32_t NewVal[1], float64_t x, float64_t a, float64_t b, float64_t f, float64_t h)
{   
    if (fabs(a-b) >= f*h)       
    {   NewVal[0]  = (a < b) ? a : b;        
        NewVal[0] += f*h;
    }
    else
    {   NewVal[0]  = a +b +pow( (2.0*f*f*h*h - pow((a-b),2.0)  ),0.5);
        NewVal[0] /= 2.0;
    }
    return;
}

void PyObject* w_fast_sweep(PyObject *dummy, PyObject *args){
    PyObject *slowness_arr, *tzero_arr;
    float_t64 patch_size;
    npy_intp hyp_in_strike, hyp_in_dip, num_in_strike, num_in_dip;
    PyArrayObject *c_slowness_arr, *c_tzero_arr;

    if (!PyArg_ParseTuple(args, 'OOdIIII', &slowness_arr, &tzero_arr&, &patch_size, &h_strk, &h_dip, &num_strk, &num_dip)){
        PyErrString(FastSweepExtError, 'Invalid call to fast_sweep! \n usage: fast_sweep(slowness_arr, tzero_arr, patch_size, h_strk, h_dip, num_strk, num_dip)');
	return NULL;

	if (! good_array(slowness_arr, NPY_FLOAT64))
    fast_sweep(Slowness, StrtTime, PatchSize, HypInStk, HypInDip, NumInStk, NumInDip)
}
