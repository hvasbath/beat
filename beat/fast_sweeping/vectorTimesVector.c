#section support_code

// Support code function
bool vector_same_shape(PyArrayObject* arr1, PyArrayObject* arr2)
{
    return (PyArray_DIMS(arr1)[0] == PyArray_DIMS(arr2)[0]);
}


#section support_code_apply

// Apply-specific support function
void APPLY_SPECIFIC(vector_elemwise_mult)(
    DTYPE_INPUT_0* x_ptr, int x_str,
    DTYPE_INPUT_1* y_ptr, int y_str,
    DTYPE_OUTPUT_0* z_ptr, int z_str, int nbElements)
{
    for (int i=0; i < nbElements; i++){
        z_ptr[i * z_str] = x_ptr[i * x_str] * y_ptr[i * y_str];
    }
}

// Apply-specific main function
int APPLY_SPECIFIC(vector_times_vector)(PyArrayObject* input0,
                                        PyArrayObject* input1,
                                        PyArrayObject** output0)
{
    // Validate that the inputs have the same shape
    if ( !vector_same_shape(input0, input1))
    {
        PyErr_Format(PyExc_ValueError, "Shape mismatch : "
                    "input0.shape[0] and input1.shape[0] should "
                    "match but x.shape[0] == %i and "
                    "y.shape[0] == %i",
                    PyArray_DIMS(input0)[0], PyArray_DIMS(input1)[0]);
        return 1;
    }

    // Validate that the output storage exists and has the same
    // dimension as x.
    if (NULL == *output0 || !(vector_same_shape(input0, *output0)))
    {
        /* Reference received to invalid output variable.
        Decrease received reference's ref count and allocate new
        output variable */
        Py_XDECREF(*output0);
        *output0 = (PyArrayObject*)PyArray_EMPTY(1,
                                                PyArray_DIMS(input0),
                                                TYPENUM_OUTPUT_0,
                                                0);

        if (!*output0) {
            PyErr_Format(PyExc_ValueError,
                        "Could not allocate output storage");
            return 1;
        }
    }

    // Perform the actual vector-vector multiplication
    APPLY_SPECIFIC(vector_elemwise_mult)(
                            (DTYPE_INPUT_0*)PyArray_DATA(input0),
                            PyArray_STRIDES(input0)[0] / ITEMSIZE_INPUT_0,
                            (DTYPE_INPUT_1*)PyArray_DATA(input1),
                            PyArray_STRIDES(input1)[0] / ITEMSIZE_INPUT_1,
                            (DTYPE_OUTPUT_0*)PyArray_DATA(*output0),
                            PyArray_STRIDES(*output0)[0] / ITEMSIZE_OUTPUT_0,
                            PyArray_DIMS(input0)[0]);

    return 0;
}