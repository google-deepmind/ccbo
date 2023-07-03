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

from typing import Callable, Any

from emukit.model_wrappers import gpy_model_wrappers
import numpy as np

from ccbo.acquisitions import expected_improvement
from ccbo.utils import utilities


class NoisyCausalExpectedImprovement(
    expected_improvement.CausalExpectedImprovement
):
  """Causal expected improvement acquisition function."""

  def __init__(
      self,
      task: utilities.Task,
      mean_function: Callable[[np.ndarray], np.ndarray],
      variance_function: Callable[[np.ndarray], np.ndarray],
      model: gpy_model_wrappers.GPyModelWrapper,
      previous_variance: float = 1.0,
      jitter: float = 0.0,
      n_samples: int = 10
  ) -> None:

    base_args = {
        "current_global_opt": None,
        "task": task,
        "mean_function": mean_function,
        "variance_function": variance_function,
        "previous_variance": previous_variance,
        "jitter": jitter,
        "model": model}

    expected_improvement.CausalExpectedImprovement.__init__(self, **base_args)

    # How many samples to get from the given model to compute the current global
    # optimum and then evaluate the improvement
    self.n_samples = n_samples

  def get_global_opt(self,
                     mean: np.ndarray,
                     standard_deviation: np.ndarray,
                     task: Any,
                     *unused_args) -> float:
    """Get one value of global optimum by sampling."""
    return task(np.random.normal(mean, standard_deviation))

  def get_improvement(self, x: np.ndarray) -> np.ndarray:
    """Compute the expected improvement."""
    out = []
    for _ in range(self.n_samples):
      mean, standard_deviation = self.get_mean_std_for_improvement(x)

      current_global_opt = self.get_global_opt(mean, standard_deviation,
                                               utilities.EVAL_FN[self.task], x)
      improvement = self.get_improvement_for_current_opt(
          current_global_opt, mean, standard_deviation)

      if self.task.value == utilities.Task.MAX.value:
        improvement *= -1
      out.append(improvement)

    return np.mean(np.hstack(out), axis=1)
