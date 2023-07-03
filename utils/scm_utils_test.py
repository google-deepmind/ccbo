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

"""Tests for utils.scm_utils."""

import unittest
import numpy as np

from ccbo.experiments import data
from ccbo.utils import scm_utils


def setup():
  example = data.EXAMPLES_DICT["synthetic1"]()
  scm = example.structural_causal_model(
      variables=("X", "Z"), lambdas=(1., 2.))
  graph = scm.graph
  scm_funcs = scm.scm_funcs

  d_o = {
      "X":
          np.array([[1.62435], [-1.07297], [1.74481], [-0.24937],
                    [-0.32241]]),
      "Z":
          np.array([[-0.41472], [3.78945], [-0.58653], [2.74533], [0.99641]]),
      "Y":
          np.array([[-0.63389], [-3.92631], [0.12215], [-3.85439], [0.72569]])
  }
  return graph, scm_funcs, d_o


class ScmUtilsTest(unittest.TestCase):

  def test_build_fitted_scm(self):
    graph, scm_funcs, d_o = setup()
    fitted_scm_fncs = scm_utils.fit_scm_fncs(graph, d_o, scm_funcs, 1)
    fitted_scm = scm_utils.build_fitted_scm(graph, fitted_scm_fncs, scm_funcs)

    # Check that keys of dictionary are correct
    self.assertEqual(list(fitted_scm().functions().keys()), list(graph.nodes))

    # Check that the correct sampling functions are used by looking at the
    # number of args taken by each function
    for k, v in fitted_scm().functions().items():
      if not list(graph.predecessors(k)):
        # When variable is exogenous number of args = 1
        self.assertEqual(v.__code__.co_argcount, 1)
      else:
        # When variable is endogenous number of args = 3
        self.assertEqual(v.__code__.co_argcount, 3)

  def test_fit_scm_fncs(self):
    graph, scm_funcs, d_o = setup()
    fitted_scm_fncs = scm_utils.fit_scm_fncs(graph, d_o, scm_funcs, 1)

    # Check that keys of dictionary are correct
    for fn_key, k in zip(
        list(fitted_scm_fncs.keys()),
        [(None, "X"), (("X",), "Z"), (("Z",), "Y")],
    ):
      self.assertEqual(fn_key, k)

    # Check that KernelDensity is used for exogenous variables (with number of
    # input variables equal to None) and GPRegression is used for variables that
    # have parents (whose number gives the number of input variables).
    for k, v in fitted_scm_fncs.items():
      print(k)
      if not list(graph.predecessors(k[1])):
        # When variable is exogenous we use KernelDensity
        self.assertIsNone(k[0])
        self.assertEqual(type(v).__name__, "KernelDensity")
      else:
        # When variable is endogenous we use GPRegression
        self.assertEqual(type(v).__name__, "GPRegression")


if __name__ == "__main__":
  unittest.main()
