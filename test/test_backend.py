import os
from unittest import TestCase

import numpy as num
import pymc3 as pm
import theano.tensor as tt

from beat.backend import NumpyChain, TextChain, check_multitrace, load_multitrace


class TestBackend(TestCase):
    def setUp(self):
        self.data_keys = ["Data", "A", "F", "D", "B", "Hood", "like"]
        number_of_parameters = 5
        self.like_index = self.data_keys.index("like")

        mu1 = num.ones(number_of_parameters) * (1.0 / 2)
        mu2 = -mu1

        stdev = 0.1
        sigma = num.power(stdev, 2) * num.eye(number_of_parameters)
        isigma = num.linalg.inv(sigma)
        dsigma = num.linalg.det(sigma)

        w1 = stdev
        w2 = 1 - stdev

        def two_gaussians(x):
            log_like1 = (
                -0.5 * number_of_parameters * tt.log(2 * num.pi)
                - 0.5 * tt.log(dsigma)
                - 0.5 * (x - mu1).T.dot(isigma).dot(x - mu1)
            )
            log_like2 = (
                -0.5 * number_of_parameters * tt.log(2 * num.pi)
                - 0.5 * tt.log(dsigma)
                - 0.5 * (x - mu2).T.dot(isigma).dot(x - mu2)
            )
            return tt.log(w1 * tt.exp(log_like1) + w2 * tt.exp(log_like2))

        with pm.Model() as self.PT_test:
            for data_key in self.data_keys:
                if data_key != "like":
                    uniform = pm.Uniform(
                        data_key,
                        shape=number_of_parameters,
                        lower=-2.0 * num.ones_like(mu1),
                        upper=2.0 * num.ones_like(mu1),
                        testval=-1.0 * num.ones_like(mu1),
                        transform=None,
                    )
                else:
                    like = pm.Deterministic("tmp", two_gaussians(uniform))
            pm.Potential(self.data_keys[self.like_index], like)

        # create or get test folder to write files.
        self.test_dir_path = os.path.join(os.path.dirname(__file__), "PT_TEST")
        if not os.path.exists(self.test_dir_path):
            try:
                os.mkdir(self.test_dir_path)
            except IOError as e:
                print(e)

        # create data.
        data_increment = 1
        self.lpoint = []
        for data_key in self.data_keys:
            if data_key != "like":
                chain_data = (
                    num.arange(number_of_parameters).astype(num.float) * data_increment
                )
            else:
                chain_data = num.array([10.0]).astype(num.float)
            data_increment += 1
            self.lpoint.append(chain_data)

        self.sample_size = 5
        self.data = []
        self.expected_chain_data = {}
        for i in range(self.sample_size):
            self.data.append(self.lpoint)
        for data_key, chain_data in zip(self.data_keys, self.lpoint):
            data = []
            for i in range(self.sample_size):
                data.append(chain_data)
            self.expected_chain_data[data_key] = num.array(data)

    def test_text_chain(self):

        textchain = TextChain(dir_path=self.test_dir_path, model=self.PT_test)
        textchain.setup(10, 0, overwrite=True)

        # write data to buffer
        draw = 0
        for lpoint in self.data:
            draw += 1
            textchain.write(lpoint, draw)

        textchain.record_buffer()

        for data_key in self.data_keys:
            chain_data = textchain.get_values(data_key)
            data_index = 1
            chain_at = textchain.point(data_index)
            self.assertEqual(
                chain_data.all(), self.expected_chain_data.get(data_key).all()
            )
            self.assertEqual(
                chain_at[data_key].all(),
                self.expected_chain_data.get(data_key)[data_index].all(),
            )

    def test_chain_bin(self):

        numpy_chain = NumpyChain(dir_path=self.test_dir_path, model=self.PT_test)
        numpy_chain.setup(10, 0, overwrite=True)
        print(numpy_chain)
        # write data to buffer
        draw = 0
        for lpoint in self.data:
            draw += 1
            numpy_chain.write(lpoint, draw)

        numpy_chain.record_buffer()

        # print("Var shapes: ", numpy_chain.var_shapes)
        # print("flat names: ", numpy_chain.flat_names)
        # print("Var names: ", numpy_chain.varnames)

        # print("Var shapes: ", numpy_chain.var_shapes)
        # print("flat names: ", numpy_chain.flat_names)
        # print("Var names: ", numpy_chain.varnames)

        for data_key in self.data_keys:
            chain_data = numpy_chain.get_values(data_key)
            data_index = 1
            chain_at = numpy_chain.point(data_index)
            # print(data_key + ": ", chain_data)
            self.assertEqual(
                chain_data.all(), self.expected_chain_data.get(data_key).all()
            )
            self.assertEqual(
                chain_at[data_key].all(),
                self.expected_chain_data.get(data_key)[data_index].all(),
            )

    def test_load_bin_chain(self):
        numpy_chain = NumpyChain(dir_path=self.test_dir_path, model=self.PT_test)
        numpy_chain.setup(5, 0, overwrite=False)
        # print(len(numpy_chain), numpy_chain.data_file())
        data_index = 1
        chain_at = numpy_chain.point(data_index)
        # print(chain_at)
        for data_key in self.data_keys:
            self.assertEqual(
                chain_at[data_key].all(),
                self.expected_chain_data.get(data_key)[data_index].all(),
            )

    def test_load_check_multitrace(self):
        mtrace = load_multitrace(
            self.test_dir_path, varnames=self.PT_test.vars, backend="bin"
        )
        mtrace.point(1)

        corrupted = check_multitrace(mtrace, self.sample_size, 1)
        self.assertEqual(len(corrupted), 0)


#    def tearDown(self):
#        import shutil
#        shutil.rmtree(self.test_dir_path)
