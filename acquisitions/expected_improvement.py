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

"""Implementation of the constrained causal acquisition functions."""

from typing import Callable, Tuple, Optional, Any

from emukit.core import acquisition as acq
from emukit.model_wrappers import gpy_model_wrappers
import numpy as np
from ccbo.utils import utilities


class CausalExpectedImprovement(acq.Acquisition):
  """Causal expected improvement acquisition function."""

  def __init__(
      self,
      current_global_opt: Optional[float],
      task: utilities.Task,
      mean_function: Callable[[np.ndarray], np.ndarray],
      variance_function: Callable[[np.ndarray], np.ndarray],
      model: gpy_model_wrappers.GPyModelWrapper,
      previous_variance: float = 1.0,
      jitter: float = 0.0
  ) -> None:

    self.model = model
    self.mean_function = mean_function
    self.previous_variance = previous_variance
    self.variance_function = variance_function
    self.jitter = jitter
    self.current_global_opt = current_global_opt
    self.task = task
    self.gradients = False

  def get_improvement(self, x: np.ndarray) -> np.ndarray:
    """Compute the expected improvement."""
    mean, standard_deviation = self.get_mean_std_for_improvement(x)
    improvement = self.get_improvement_for_current_opt(
        self.current_global_opt, mean, standard_deviation)

    if self.task.value == utilities.Task.MAX.value:
      improvement *= -1
    return improvement

  def get_mean_std_for_improvement(
      self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get mean and std at x to be used to compute the expected improvement."""
    if self.model:
      mean, variance = self.model.predict(x)
    else:
      mean = self.mean_function(x)
      variance = self.previous_variance * np.ones(
          (x.shape[0], 1)) + self.variance_function(x)

    # Variance computed with MonteCarlo leads to numerical instability
    # This is ensuring that negative values or nan values are not generated
    if np.any(np.isnan(variance)):
      variance[np.isnan(variance)] = 0
    elif np.any(variance < 0):
      variance = variance.clip(0.0001)

    standard_deviation = np.sqrt(variance)
    mean += self.jitter
    return mean, standard_deviation

  def get_improvement_for_current_opt(
      self,
      current_global_opt: float,
      mean: np.ndarray,
      standard_deviation: np.ndarray,
  ) -> np.ndarray:
    """Get improvement wrt the given global minimum and with given params."""
    u, pdf, cdf = utilities.get_standard_normal_pdf_cdf(
        current_global_opt, mean, standard_deviation
    )
    return standard_deviation * (u * cdf + pdf)

  def evaluate(self, x: np.ndarray) -> np.ndarray:
    """Evaluate the causal EI."""
    return self.get_improvement(x)

  def evaluate_with_gradients(
      self, x: np.ndarray
  ) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the causal EI and its derivative."""
    raise NotImplementedError("Not implemented for this class.")

  def evaluate_to_store(self, x: np.ndarray) -> Any:
    raise NotImplementedError("Not implemented for this class.")

  @property
  def has_gradients(self) -> bool:
    """Returns that this acquisition does not have gradients."""
    # This function is needed to comply with emukit requirements
    return self.gradients
