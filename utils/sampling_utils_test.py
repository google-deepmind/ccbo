# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for utils.utilities."""

import unittest
import numpy as np

from ccbo.experiments import data
from ccbo.utils import sampling_utils
from ccbo.utils import scm_utils


class SamplingUtilsTest(unittest.TestCase):

  def test_sample_scm(self):
    example = data.EXAMPLES_DICT["synthetic1"]()
    scm = example.structural_causal_model(
        variables=("X", "Z"), lambdas=(1., 2.))
    graph = scm.graph
    precision = 5

    # Test sampling from true observational distribution
    d_o_expected_values = {
        "X":
            np.array([[1.62435], [-1.07297], [1.74481], [-0.24937],
                      [-0.32241]]),
        "Z":
            np.array([[-0.41472], [3.78945], [-0.58653], [2.74533], [0.99641]]),
        "Y":
            np.array([[-0.63389], [-3.92631], [0.12215], [-3.85439], [0.72569]])
    }

    # Sample from the true observational distribution
    d_o = sampling_utils.sample_scm(
        scm_funcs=scm.scm_funcs,
        graph=None,
        n_samples=5,
        compute_moments=False,
        seed=1)

    # Test
    for key, value in d_o.items():
      assert isinstance(value, np.ndarray)
      np.testing.assert_array_almost_equal(value, d_o_expected_values[key],
                                           precision)

    # Test sampling from true intervetional distribution
    d_i_expected_values = {
        "X":
            np.array([[1.], [1.], [1.], [1.], [1.]]),
        "Y":
            np.array([[-0.57003], [-2.91060], [0.22282], [-3.22900],
                      [1.13283]]),
        "Z":
            np.array([[-0.24388], [1.23329], [-0.39333], [1.82999], [-0.01617]])
    }

    intervention = {v: None for v in graph.nodes}
    intervention_level = np.array(1.)
    intervention_var = "X"
    intervention[intervention_var] = intervention_level

    # Sample from the true interventional distribution
    d_i = sampling_utils.sample_scm(
        scm_funcs=scm.scm_funcs,
        graph=None,
        interventions=intervention,
        n_samples=5,
        compute_moments=False,
        seed=1)

    # Test
    for val in d_i[intervention_var]:
      self.assertEqual(val, intervention_level)

    for var in ["Z", "Y"]:
      np.testing.assert_array_almost_equal(d_i[var], d_i_expected_values[var],
                                           precision)

    # Test sampling from estimated interventional distribution
    d_i_estimated_expected_values = {
        "X":
            np.array([[1.], [1.], [1.], [1.], [1.]]),
        "Y":
            np.array([[-0.45850], [0.03227], [-0.33184], [-0.02329],
                      [-1.01595]]),
        "Z":
            np.array([[-0.18379], [0.37207], [-0.52341], [0.71496], [-0.52929]])
    }

    # Sample from the estimated interventional distribution given the fitted
    # SCM functions
    fitted_scm_fncs = scm_utils.fit_scm_fncs(graph, d_o,
                                             scm.scm_funcs,
                                             1)
    fitted_scm = scm_utils.build_fitted_scm(graph, fitted_scm_fncs,
                                            scm.scm_funcs)
    d_i_estimated = sampling_utils.sample_scm(
        scm_funcs=fitted_scm().functions(),
        graph=graph,
        interventions=intervention,
        n_samples=5,
        compute_moments=False,
        seed=1)

    # Test
    for var in ["Z", "Y"]:
      np.testing.assert_array_almost_equal(d_i_estimated[var],
                                           d_i_estimated_expected_values[var],
                                           precision)

  def test_select_sample(self):
    values = {
        "X": np.array([[1.], [2], [3.]]),
        "Y": np.array([[4.], [5], [6.]]),
        "Z": np.array([[7.], [8], [9.]]),
    }
    input_variables_list = ["X", ["Z", "Y"]]
    expected_values = [
        np.array([[1.], [2], [3.]]),
        np.array([[7., 4.], [8., 5.], [9., 6.]])
    ]

    for var, value in zip(input_variables_list, expected_values):
      res = sampling_utils.select_sample(values, var)
      np.testing.assert_array_equal(res, value)


if __name__ == "__main__":
  unittest.main()
