import os
from unittest import TestCase

import numpy as num
import pymc3 as pm
import theano.tensor as tt

from beat.backend import TextChain, TextChainBin


class TestBackend(TestCase):

    def setUp(self):
        self.data_keys = ["Data", "like"]
        number_of_parameters = 100

        mu1 = num.ones(number_of_parameters) * (1. / 2)
        mu2 = -mu1

        stdev = 0.1
        sigma = num.power(stdev, 2) * num.eye(number_of_parameters)
        isigma = num.linalg.inv(sigma)
        dsigma = num.linalg.det(sigma)

        w1 = stdev
        w2 = (1 - stdev)

        def two_gaussians(x):
            log_like1 = - 0.5 * number_of_parameters * tt.log(2 * num.pi) \
                        - 0.5 * tt.log(dsigma) \
                        - 0.5 * (x - mu1).T.dot(isigma).dot(x - mu1)
            log_like2 = - 0.5 * number_of_parameters * tt.log(2 * num.pi) \
                        - 0.5 * tt.log(dsigma) \
                        - 0.5 * (x - mu2).T.dot(isigma).dot(x - mu2)
            return tt.log(w1 * tt.exp(log_like1) + w2 * tt.exp(log_like2))

        with pm.Model() as self.PT_test:
            uniform = pm.Uniform(self.data_keys[0], shape=number_of_parameters, lower=-2. * num.ones_like(mu1),
                                 upper=2. * num.ones_like(mu1), testval=-1. * num.ones_like(mu1), transform=None)
            like = pm.Deterministic('tmp', two_gaussians(uniform))
            pm.Potential(self.data_keys[1], like)

        # create or get test folder to write files.
        self.test_dir_path = os.path.join(os.path.dirname(__file__), 'PT_TEST')
        if not os.path.exists(self.test_dir_path):
            try:
                os.mkdir(self.test_dir_path)
            except IOError as e:
                print(e)

        # create data.
        chain_data = num.arange(number_of_parameters).astype(num.float)
        chain_like = num.array([10])
        self.lpoint = [chain_data, chain_like]
        self.data_size = 100
        self.data = []
        self.expected_chain_data = []
        self.expected_chain_like = []
        for i in range(self.data_size):
            self.data.append(self.lpoint)
            self.expected_chain_data.append(chain_data)
            self.expected_chain_like.append(chain_like)

        self.expected_chain_data = num.array(self.expected_chain_data)
        self.expected_chain_like = num.array(self.expected_chain_like)

    def test_text_chain(self):

        textchain = TextChain(name=self.test_dir_path, model=self.PT_test)
        textchain.setup(10, 1, overwrite=True)

        # write data to buffer
        draw = 0
        for lpoint in self.data:
            draw += 1
            textchain.write(lpoint, draw)

        textchain.record_buffer()

        chain_data = textchain.get_values(self.data_keys[0])
        chain_like = textchain.get_values(self.data_keys[1])
        data_index = 1
        chain_at = textchain.point(data_index)

        self.assertEqual(chain_data.all(), self.expected_chain_data.all())
        self.assertEqual(chain_like.all(), self.expected_chain_like.all())
        self.assertEqual(chain_at[self.data_keys[0]].all(), self.expected_chain_data[data_index].all())
        self.assertEqual(chain_at[self.data_keys[1]].all(), self.expected_chain_like[data_index].all())

    def test_text_chain_bin(self):

        textchainbin = TextChainBin(name=self.test_dir_path, model=self.PT_test)
        textchainbin.setup(10, 1, overwrite=True)

        # write data to buffer
        draw = 0
        for lpoint in self.data:
            draw += 1
            textchainbin.write(lpoint, draw)

        textchainbin.record_buffer()

        chain_data = textchainbin.get_values(self.data_keys[0])
        chain_like = textchainbin.get_values(self.data_keys[1])
        data_index = 1
        chain_at = textchainbin.point(data_index)

        self.assertEqual(chain_data.all(), self.expected_chain_data.all())
        self.assertEqual(chain_like.all(), self.expected_chain_like.all())
        self.assertEqual(chain_at[self.data_keys[0]].all(), self.expected_chain_data[data_index].all())
        self.assertEqual(chain_at[self.data_keys[1]].all(), self.expected_chain_like[data_index].all())
