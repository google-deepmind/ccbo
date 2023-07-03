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

"""Implementation of the causal constrained acquisition functions."""

from typing import Any, Callable, Dict, Tuple, Optional, List

from emukit.model_wrappers import gpy_model_wrappers
import numpy as np
import scipy.stats

from ccbo.acquisitions import expected_improvement
from ccbo.utils import utilities


class ConstrainedCausalExpectedImprovement(
    expected_improvement.CausalExpectedImprovement
):
  """Implementation of the causal constrained EI acquisition function."""

  def __init__(
      self,
      current_global_opt: Optional[float],
      task: utilities.Task,
      mean_function: Callable[[np.ndarray], np.ndarray],
      variance_function: Callable[[np.ndarray], np.ndarray],
      model: Optional[gpy_model_wrappers.GPyModelWrapper],
      model_constraints: Optional[gpy_model_wrappers.GPyModelWrapper],
      constraints: Dict[str, List[Any]],
      constraints_dict: Dict[Tuple[str, ...], Dict[str, Any]],
      mean_function_constraints: Optional[
          Dict[str, Callable[[np.ndarray], np.ndarray]]
      ],
      variance_function_constraints: Optional[
          Dict[str, Callable[[np.ndarray], np.ndarray]]
      ],
      exploration_set: Tuple[str, ...],
  ) -> None:
    base_args = {
        "current_global_opt": current_global_opt,
        "task": task,
        "mean_function": mean_function,
        "variance_function": variance_function,
        "model": model,
    }
    super().__init__(**base_args)

    self.model_constraints = model_constraints
    self.constraints = constraints
    self.constraints_dict = constraints_dict
    self.mean_function_constraints = mean_function_constraints
    self.variance_function_constraints = variance_function_constraints
    self.exploration_set = exploration_set

  def get_probability_feasibility(self, x: np.ndarray) -> Any:
    """Compute the probability of feasibility."""
    probability_feasibility = np.ones((x.shape[0], 1))

    # Check if constraints exist for the given exploration set
    # self.mean_function_constraints is an empty dict if the exploration
    # set does not have constraints. With any we return False in this case.
    # if any(self.mean_function_constraints):
    if self.model_constraints:
      for p in self.model_constraints.keys():
        direction, value = self.constraints[p][0], self.constraints[p][1]
        if self.model_constraints[p]:
          mean, variance = self.model_constraints[p].predict(x)
          if len(variance.shape) == 3:
            # The predict function returns a 3-d tensor that we want to reduce
            # to the same shape of the inputs
            variance = np.squeeze(variance, axis=2)
          elif len(variance.shape) > 3:
            raise ValueError("Prediction returns a high dimensional tensor!")
        else:
          assert self.mean_function_constraints
          assert self.variance_function_constraints
          mean = self.mean_function_constraints[p](x)
          variance = self.variance_function_constraints[p](x).clip(0)

        standardized_value = (value - mean) / np.sqrt(variance)

        if direction == utilities.Direction.LOWER:
          probability_feasibility *= scipy.stats.norm.cdf(standardized_value)
        else:
          probability_feasibility *= 1 - scipy.stats.norm.cdf(
              standardized_value)
    return probability_feasibility

  def evaluate(self, x: np.ndarray) -> np.ndarray:
    """Evaluate the Constrained Expected Improvement."""
    return self.get_improvement(x) * self.get_probability_feasibility(x)

  def evaluate_to_store(self, x: np.ndarray)-> Any:
    """Evaluate the improvement and probability of feasibility separately."""
    return self.get_improvement(x), self.get_probability_feasibility(x)
