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

"""Implement the causal kernel as an instance of a stationary kernel."""

from typing import Callable, Optional

import GPy
from GPy.kern.src import psi_comp
from GPy.kern.src import stationary
import numpy as np
from paramz import transformations


class CausalRBF(stationary.Stationary):
  """Implement the causal RBF kernel.

    For details about the kernel see:
    Aglietti, V., Lu, X., Paleyes, A., & González, J. (2020, June). Causal
    bayesian optimization. In International Conference on Artificial
    Intelligence and Statistics (pp. 3155-3164). PMLR.
  """

  def __init__(
      self,
      input_dim: int,
      variance_adjustment: Callable[[np.ndarray], np.ndarray],
      variance: float = 1.0,
      lengthscale: Optional[float] = None,
      rescale_variance: Optional[float] = 1.0,
      ard: Optional[bool] = False,
      active_dims: Optional[int] = None,
      name: str = "rbf",
      usegpu: Optional[bool] = False,
      inv_l: Optional[bool] = False,
  ):
    super().__init__(input_dim, variance, lengthscale, ard, active_dims, name,
                     useGPU=usegpu)
    self.usegpu = usegpu
    if self.usegpu:
      self.psicomp = psi_comp.PSICOMP_RBF_GPU()
    else:
      self.psicomp = psi_comp.PSICOMP_RBF()
    self.use_invengthscale = inv_l
    if inv_l:
      self.unlink_parameter(self.lengthscale)
      self.inv_l = GPy.core.Param("inv_lengthscale", 1.0 / self.lengthscale**2,
                                  transformations.Logexp())
      self.link_parameter(self.inv_l)
    self.variance_adjustment = variance_adjustment
    self.rescale_variance = GPy.core.Param("rescale_variance", rescale_variance,
                                           transformations.Logexp())

  def K(self, x: np.ndarray, x2: Optional[np.ndarray] = None) -> np.ndarray:
    """Kernel function applied on inputs x and x2."""
    if x2 is None:
      x2 = x
    r = self._scaled_dist(x, x2)
    values = self.variance * np.exp(-0.5 * r ** 2)

    value_diagonal_x = self.variance_adjustment(x)
    value_diagonal_x2 = self.variance_adjustment(x2)

    additional_matrix = np.dot(
        np.sqrt(value_diagonal_x), np.sqrt(np.transpose(value_diagonal_x2)))

    assert additional_matrix.shape == values.shape, (
        additional_matrix.shape,
        values.shape,
    )
    return values + additional_matrix

  def Kdiag(self, x: np.ndarray) -> np.ndarray:
    # ret = np.empty(x.shape[0])
    # ret[:] = np.repeat(0.1, x.shape[0])

    # diagonal_terms = ret

    value = self.variance_adjustment(x)

    if x.shape[0] == 1 and x.shape[1] == 1:
      diagonal_terms = value
    else:
      if np.isscalar(value):
        diagonal_terms = value
      else:
        diagonal_terms = value[:, 0]
    return self.variance + diagonal_terms

  def K_of_r(self, r: float) -> float:
    return self.variance * np.exp(-0.5 * r ** 2)

  def dK_dr(self, r: float) -> float:
    return -r * self.K_of_r(r)

  def dK2_drdr(self, r: float) -> float:
    return (r ** 2 - 1) * self.K_of_r(r)

  def dK2_drdr_diag(self) -> float:
    return -self.variance  # as the diagonal of r is always filled with zeros
