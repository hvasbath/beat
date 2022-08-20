#define NPY_NO_DEPRECATED_API 7
#include "Python.h"
#include "numpy/arrayobject.h"
#include <numpy/npy_math.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

typedef npy_float64 float64_t;

struct module_state {
    PyObject *error;
};

#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))

int good_array(PyObject* o, int typenum, npy_intp size_want, int ndim_want, npy_intp* shape_want){
    int i;

    if (!PyArray_Check(o)) {
        PyErr_SetString(PyExc_AttributeError, "not a NumPy array" );
        return 0;
    }

    if (PyArray_TYPE((PyArrayObject*)o) != typenum) {
        PyErr_SetString(PyExc_AttributeError, "array of unexpected type");
        return 0;
    }

    if (!PyArray_ISCARRAY((PyArrayObject*)o)) {
        PyErr_SetString(PyExc_AttributeError, "array is not contiguous or not well behaved");
        return 0;
    }

    if (size_want != -1 && size_want != PyArray_SIZE((PyArrayObject*)o)) {
        PyErr_SetString(PyExc_AttributeError, "array is of unexpected size");
        return 0;
    }

    if (ndim_want != -1 && ndim_want != PyArray_NDIM((PyArrayObject*)o)) {
        PyErr_SetString(PyExc_AttributeError, "array is of unexpected ndim");
        return 0;
    }

    if (ndim_want != -1 && shape_want != NULL) {
        for (i=0; i<ndim_want; i++) {
            if (shape_want[i] != -1 && shape_want[i] != PyArray_DIMS((PyArrayObject*)o)[i]) {
                PyErr_SetString(PyExc_AttributeError, "array is of unexpected shape");
            return 0;
            }
        }
    }

    return 1;
}

void Vect_from_Mat(npy_intp VectPos[1], npy_intp sel_row, npy_intp sel_col, npy_intp colNum){
    //goes row by row
    VectPos[0] = sel_row*colNum + sel_col;
    //the vectors are given, starting at zero, the colNum is given in actual numbers of columns (starting with 1!)
    return;
}

void eq_solve(float64_t NewVal[1], float64_t x, float64_t a, float64_t b, float64_t f, float64_t h){
    if (fabs(a-b) >= f*h){
        NewVal[0]  = (a < b) ? a : b;
        NewVal[0] += f*h;
    }
    else{
        NewVal[0]  = a +b +pow( (2.0*f*f*h*h - pow((a-b),2.0)  ),0.5);
        NewVal[0] /= 2.0;
    }
    return;
}

void upwind(float64_t NewVal[1], float64_t *StartTime, npy_intp i, npy_intp j, float64_t *Slowness, float64_t PatchSize, npy_intp NumInStk, npy_intp NumInDip){
    npy_intp     i1, i2, j1, j2;
    npy_intp     VectPos0[1];
    npy_intp     VectPos1[1], VectPos2[1];
    npy_intp     VectPos3[1], VectPos4[1];
    float64_t    u_xmin, u_ymin;

    i1 = i - 1;
    i2 = i + 1;
    j1 = j - 1;
    j2 = j + 1;

    if (i1 < 0){
        i1 = 0;
    }

    if (i2 >= NumInStk){
        i2 = NumInStk-1;
    }

    if (j1 < 0){
        j1 = 0;
    }

    if (j2 >= NumInDip){
        j2 = NumInDip-1;
    }

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

void fast_sweep(float64_t *Slowness, float64_t *StartTime, float64_t PatchSize, npy_intp HypoInStk, npy_intp HypoInDip, npy_intp NumInStk, npy_intp NumInDip){
    /* convention for the fault orientation here is dip-direction along columns and strike-direction along rows of the start-times*/
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

    Time_old = (float64_t *) malloc((size_t) ((PatchNum)*sizeof(float64_t)));

    int cnt = 0;

    for (i = 0; i < NumInStk; i++){
        for (j = 0; j < NumInDip; j++){
            Vect_from_Mat(VectPos, i, j, NumInDip);
            StartTime[ VectPos[0] ] = +INFINITY;
            cnt++;
        }
    }

    Vect_from_Mat(VectPos, HypoInStk, HypoInDip, NumInDip);
    StartTime[ VectPos[0] ] = 0.0;

    while (err > epsilon){
        for (i = 0; i < NumInStk; i++){
            for (j = 0; j < NumInDip; j++){
                Vect_from_Mat(VectPos, i, j, NumInDip);
                Time_old[ VectPos[0] ] = StartTime[ VectPos[0] ];
            }
        }

        for (ii = 0; ii < 4; ii++){
            if (ii == 0){
                for (i = 0; i < NumInStk; i++){
                    for (j = 0; j < NumInDip; j++){
                        upwind(NewVal, StartTime, i,j, Slowness, PatchSize, NumInStk, NumInDip);
                        Vect_from_Mat(VectPos, i, j, NumInDip);
                        StartTime[VectPos[0]] = NewVal[0];
                    }
                }
            }
            else if (ii == 1){
                for (i = (NumInStk-1); i >= 0; i--){
                    for (j = 0; j < NumInDip; j++){
                        upwind(NewVal, StartTime, i,j, Slowness, PatchSize, NumInStk, NumInDip);
                        Vect_from_Mat(VectPos, i, j, NumInDip);
                        StartTime[VectPos[0]] = NewVal[0];
                    }
                }
            }
            else if (ii == 2){
                for (i = (NumInStk-1); i >= 0; i--){
                    for (j = (NumInDip-1); j >= 0; j--){
                        upwind(NewVal, StartTime, i,j, Slowness, PatchSize, NumInStk, NumInDip);
                        Vect_from_Mat(VectPos, i, j, NumInDip);
                        StartTime[VectPos[0]] = NewVal[0];
                    }
                }
            }
            else if (ii == 3){
                for (i = 0; i < NumInStk; i++){
                    for (j = (NumInDip-1); j >= 0; j--){
                        upwind(NewVal, StartTime, i,j, Slowness, PatchSize, NumInStk, NumInDip);
                        Vect_from_Mat(VectPos, i, j, NumInDip);
                        StartTime[VectPos[0]] = NewVal[0];
                    }
                }
            }
        }

        err = 0.0;
        for (i = 0; i < PatchNum; i++){
            err += pow((StartTime[i]-Time_old[i]),2.0);
        }
        num_iter++;
    }
    free(Time_old);
    return;
}

static PyObject* w_fast_sweep(PyObject *m, PyObject *args){
    PyObject *slowness_arr;
    PyArrayObject *c_slowness_arr, *tzero_arr;

    float64_t patch_size, *slowness, *tzero;
    npy_intp h_strk, h_dip, num_strk, num_dip, arr_size[1];

    struct module_state *st = GETSTATE(m);

    if (!PyArg_ParseTuple(args, "Odkkkk", &slowness_arr, &patch_size, &h_strk, &h_dip, &num_strk, &num_dip)){
        PyErr_SetString(st->error, "Invalid call to fast_sweep! \n usage: fast_sweep(slowness_arr, patch_size, h_strk, h_dip, num_strk, num_dip)");
        return NULL;
    }

    arr_size[0] = PyArray_SIZE((PyArrayObject*) slowness_arr);
//    printf("size matrix: %lu\n", PyArray_SIZE((PyArrayObject*) slowness_arr));
//    printf("ndim matrix: %i\n", PyArray_NDIM((PyArrayObject*) slowness_arr));
    if (!good_array(slowness_arr, NPY_FLOAT64, arr_size[0], -1, NULL)){
        return NULL;
    }

    tzero_arr = (PyArrayObject*) PyArray_EMPTY(1, arr_size, NPY_FLOAT64, 0);
    if (tzero_arr==NULL){
        PyErr_SetString(st->error, "Failed to allocate tzero!");
        return NULL;
    }

    c_slowness_arr = PyArray_GETCONTIGUOUS((PyArrayObject*) slowness_arr);

    slowness = PyArray_DATA(c_slowness_arr);
    tzero = PyArray_DATA(tzero_arr);

    fast_sweep(slowness, tzero, patch_size, h_strk, h_dip, num_strk, num_dip);

    Py_DECREF(c_slowness_arr);

    return (PyObject*) tzero_arr;
}

static PyMethodDef FastSweepExtMethods[] = {
    {"fast_sweep", w_fast_sweep, METH_VARARGS,
     "Fast Sweeping Algorithm to calculate rupture onset-times on patches of a plane given slowness of the rupturing patches.\n"},

    {NULL, NULL, 0, NULL}  /* Sentinel */
};

static int fast_sweep_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int fast_sweep_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "fast_sweep_ext",
        NULL,
        sizeof(struct module_state),
        FastSweepExtMethods,
        NULL,
        fast_sweep_traverse,
        fast_sweep_clear,
        NULL
};

#define INITERROR return NULL

PyMODINIT_FUNC
PyInit_fast_sweep_ext(void)
{
    PyObject* module = PyModule_Create(&moduledef);
    if (module == NULL)
        INITERROR;
    import_array();

    struct module_state *st = GETSTATE(module);
    st->error = PyErr_NewException("beat.fast_sweep_ext.error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        INITERROR;
    }
    Py_INCREF(st->error);
    PyModule_AddObject(module, "error", st->error);

    return module;
}
