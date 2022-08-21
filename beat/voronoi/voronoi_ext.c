#define NPY_NO_DEPRECATED_API 7
#include "Python.h"
#include "numpy/arrayobject.h"
#include <numpy/npy_math.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

struct module_state {
    PyObject *error;
};

#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))

typedef npy_float64 float64_t;

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

void GetMinDistances(npy_intp *VOindx4GFPnt, float64_t *GFPoints_stk, float64_t *GFPoints_dip, float64_t *VOPoints_stk, float64_t *VOPoints_dip, npy_intp GFPntNum, npy_intp VOPntNum)
{
    // for now this is done brute force... => get distance from each to each and keep the smallest
    npy_intp     i,          j;
    float64_t  CurrDist,    CurrClosest, dist_dip, dist_strike;
    //----------------------------
    for (i = 0; i < GFPntNum; i++)
    {   CurrClosest = 1.0E+99;
        for (j = 0; j < VOPntNum; j++)
        {
            dist_strike = GFPoints_stk[i] - VOPoints_stk[j];
            dist_dip = GFPoints_dip[i] - VOPoints_dip[j];
            CurrDist = pow( ((dist_strike*dist_strike) + (dist_dip*dist_dip)), 0.5);
            if (CurrDist < CurrClosest)
            {   CurrClosest     = CurrDist;
                VOindx4GFPnt[i] = j;
            }
    }   }
    //----------------------------
    return;
}


static PyObject* w_voronoi(PyObject *m, PyObject *args){
    PyObject *gf_points_dip, *gf_points_strike;
    PyObject *voronoi_points_dip, *voronoi_points_strike;

    PyArrayObject *c_gf_points_dip, *c_gf_points_strike, *c_voronoi_points_dip, *c_voronoi_points_strike, *gf2voro_idxs_arr;

    float64_t *gf_dips, *gf_strikes, *voro_dips, *voro_strikes;
    npy_intp n_gfs[1], n_voros[1], *gf2voro_idxs;

    struct module_state *st = GETSTATE(m);

    if (!PyArg_ParseTuple(args, "OOOO", &gf_points_dip, &gf_points_strike, &voronoi_points_dip, &voronoi_points_strike)){
        PyErr_SetString(st->error, "Invalid call to voronoi! \n usage: voronoi(gf_points_dip, gf_points_strike, voronoi_points_dip, voronoi_points_strike)");
        return NULL;
    }

    n_gfs[0] = PyArray_SIZE((PyArrayObject*) gf_points_dip);
    // printf("size matrix: %lu\n", PyArray_SIZE((PyArrayObject*) gf_points_dip));
    // printf("ndim matrix: %i\n", PyArray_NDIM((PyArrayObject*) gf_points_dip));
    if (!good_array(gf_points_dip, NPY_FLOAT64, n_gfs[0], -1, NULL)){
        return NULL;
    }

    n_gfs[0] = PyArray_SIZE((PyArrayObject*) gf_points_strike);
    if (!good_array(gf_points_strike, NPY_FLOAT64, n_gfs[0], -1, NULL)){
        return NULL;
    }

    gf2voro_idxs_arr = (PyArrayObject*) PyArray_EMPTY(1, n_gfs, NPY_INT64, 0);
    if (gf2voro_idxs_arr==NULL){
        PyErr_SetString(st->error, "Failed to allocate index_matrix!");
        return NULL;
    }

    n_voros[0] = PyArray_SIZE((PyArrayObject*) voronoi_points_dip);
    if (!good_array(voronoi_points_dip, NPY_FLOAT64, n_voros[0], -1, NULL)){
        return NULL;
    }

    n_voros[0] = PyArray_SIZE((PyArrayObject*) voronoi_points_strike);
    if (!good_array(voronoi_points_strike, NPY_FLOAT64, n_voros[0], -1, NULL)){
        return NULL;
    }


    c_gf_points_dip = PyArray_GETCONTIGUOUS((PyArrayObject*) gf_points_dip);
    c_gf_points_strike = PyArray_GETCONTIGUOUS((PyArrayObject*) gf_points_strike);
    c_voronoi_points_dip = PyArray_GETCONTIGUOUS((PyArrayObject*) voronoi_points_dip);
    c_voronoi_points_strike = PyArray_GETCONTIGUOUS((PyArrayObject*) voronoi_points_strike);

    gf_dips = PyArray_DATA(c_gf_points_dip);
    gf_strikes = PyArray_DATA(c_gf_points_strike);
    voro_dips = PyArray_DATA(c_voronoi_points_dip);
    voro_strikes = PyArray_DATA(c_voronoi_points_strike);
    gf2voro_idxs = PyArray_DATA(gf2voro_idxs_arr);

    GetMinDistances(gf2voro_idxs, gf_strikes, gf_dips, voro_strikes, voro_dips, n_gfs[0], n_voros[0]);

    Py_DECREF(c_gf_points_dip);
    Py_DECREF(c_gf_points_strike);
    Py_DECREF(c_voronoi_points_dip);
    Py_DECREF(c_voronoi_points_strike);

    return (PyObject*) gf2voro_idxs_arr;
}

static PyMethodDef VoronoiExtMethods[] = {
    {"voronoi", w_voronoi, METH_VARARGS,
"Voronoi cell discretization based on shortest euklidean distance.\n"},

    {NULL, NULL, 0, NULL}  /* Sentinel */
};

static int voronoi_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int voronoi_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "voronoi_ext",
        NULL,
        sizeof(struct module_state),
        VoronoiExtMethods,
        NULL,
        voronoi_traverse,
        voronoi_clear,
        NULL
};

#define INITERROR return NULL

PyMODINIT_FUNC
PyInit_voronoi_ext(void)
{
    PyObject *module = PyModule_Create(&moduledef);
    if (module == NULL)
        INITERROR;
    import_array();

    struct module_state *st = GETSTATE(module);
    st->error = PyErr_NewException("beat.voronoi.voronoi_ext.error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        INITERROR;
    }

    Py_INCREF(st->error);
    PyModule_AddObject(module, "error", st->error);

    return module;
}
