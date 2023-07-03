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

from ccbo.utils import utilities


class UtilitiesTest(unittest.TestCase):

  def test_make_column_shape_2d(self):
    num_rows = 6
    row_data = np.arange(num_rows)
    result = utilities.make_column_shape_2d(row_data)
    self.assertEqual((num_rows, 1), result.shape)

  def test_check_reshape_add_data(self):
    intervention_data_x = {
        ('X',): np.array([[1.], [3.], [5.], [7.], [9.]]),
        ('Y',): None,
        ('X', 'Z'): np.ones((5, 2))
    }
    intervention_data_y = {
        ('X',): np.array([[2.], [4.], [6.], [8.], [10.]]),
        ('Y',): None,
        ('X', 'Z'): np.zeros((5, 1))
    }
    new_interventional_data_x = np.array([[5., 3.]])
    y_new = 11.

    # Test appending interventional data to existing data
    best_es = ('X', 'Z')
    result_x, result_y = utilities.check_reshape_add_data(
        intervention_data_x, intervention_data_y, new_interventional_data_x,
        y_new, best_es)
    expected_x = np.array([[1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.],
                           [5., 3.]])
    expected_y = np.array([[0.], [0.], [0.], [0.], [0.], [11.]])

    self.assertTrue(np.equal(expected_x, result_x).all())
    self.assertTrue(np.equal(expected_y, result_y).all())

    # Test adding new interventional data
    best_es = ('Y',)
    new_interventional_data_x = np.array([5.])
    result_new_x, result_new_y = utilities.check_reshape_add_data(
        intervention_data_x, intervention_data_y, new_interventional_data_x,
        y_new, best_es)
    expected_new_x = np.array([[5.]])
    expected_new_y = np.array([[11.]])

    self.assertTrue(np.equal(expected_new_x, result_new_x).all())
    self.assertTrue(np.equal(expected_new_y, result_new_y).all())

  def test_monte_carlo_expectation(self):
    intervention_samples = {
        'X': np.array([1., 3., 5., 7., 9.]),
        'Y': np.array([0., 1., 0., 1.]),
        'Z': np.ones((5, 5))
    }

    expected_dict = {'X': 5.,
                     'Y': 0.5,
                     'Z': np.ones(5)}
    result_dict = utilities.get_monte_carlo_expectation(intervention_samples)

    self.assertEqual(expected_dict.keys(), result_dict.keys())
    for var, mean in result_dict.items():
      if isinstance(mean, np.ndarray):
        self.assertTrue(np.equal(expected_dict[var], mean).all())
      else:
        self.assertEqual(expected_dict[var], mean)

if __name__ == '__main__':
  unittest.main()
