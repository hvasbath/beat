import theano
import theano.tensor as tt
from theano import gof


class TheanoSweeper(gof.COp):

    __props__ = ()

    func_file = "./fast_sweep_ext_test.c"
    func_name = "APPLY_SPECIFIC(fast_sweep)"

    def __init__(self):
        super(TheanoSweeper, self).__init__(self.func_file,
                                            self.func_name)

    def make_node(self, x):

        # Validate the inputs' type
        if x.type.ndim != 1:
            raise TypeError('x must be a 1-d vector')
        # Create an output variable
        output_var = theano.tensor.TensorType(
                        dtype=x.dtype,
                        broadcastable=[False])()
        # output_var = tt.as_tensor_variable(num.zeros((2, 2))).type()
        return gof.Apply(self, [x], [output_var])

    def R_op(self, inputs, eval_points):
        pass


class VectorTimesVector(gof.COp):

    __props__ = ()

    func_file = "./vectorTimesVector.c"
    func_name = "APPLY_SPECIFIC(vector_times_vector)"

    def __init__(self):
        super(VectorTimesVector, self).__init__(self.func_file, self.func_name)

    def make_node(self, x, y):
        # Validate the inputs' type
        if x.type.ndim != 1:
            raise TypeError('x must be a 1-d vector')
        if y.type.ndim != 1:
            raise TypeError('y must be a 1-d vector')

        # Create an output variable of the same type as x
        output_var = theano.tensor.TensorType(
                        dtype=x.dtype,
                        broadcastable=[False])()

        return gof.Apply(self, [x, y], [output_var])

    def R_op(self, inputs, eval_points):
        pass
