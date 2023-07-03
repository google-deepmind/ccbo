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

from typing import Any, Callable, Dict, List, Optional, Tuple

from emukit.model_wrappers import gpy_model_wrappers
import numpy as np

from ccbo.acquisitions import constrained_expected_improvement
from ccbo.acquisitions import noisy_expected_improvement
from ccbo.utils import constraints_functions
from ccbo.utils import utilities


class NoisyConstrainedCausalExpectedImprovement(
    noisy_expected_improvement.NoisyCausalExpectedImprovement,
    constrained_expected_improvement.ConstrainedCausalExpectedImprovement):
  """Implementation of the noisy constrained causal EI acquisition function."""

  def __init__(self, task: utilities.Task,
               mean_function: Callable[[np.ndarray], np.ndarray],
               variance_function: Callable[[np.ndarray], np.ndarray],
               model: Optional[gpy_model_wrappers.GPyModelWrapper],
               model_constraints: Optional[gpy_model_wrappers.GPyModelWrapper],
               constraints: Dict[str, List[Any]],
               constraints_dict: Dict[Tuple[str, ...], Dict[str, Any]],
               mean_function_constraints: Optional[Callable[[np.ndarray],
                                                            np.ndarray]],
               variance_function_constraints: Optional[Callable[[np.ndarray],
                                                                np.ndarray]],
               exploration_set: Tuple[str, ...],
               n_samples: int = 10) -> None:

    noisy_expected_improvement.NoisyCausalExpectedImprovement.__init__(
        self,
        task=task,
        mean_function=mean_function,
        variance_function=variance_function,
        model=model,
        n_samples=n_samples)

    constrained_expected_improvement.ConstrainedCausalExpectedImprovement.__init__(
        self,
        current_global_opt=None,
        task=task,
        mean_function=mean_function,
        variance_function=variance_function,
        model=model,
        model_constraints=model_constraints,
        constraints=constraints,
        constraints_dict=constraints_dict,
        mean_function_constraints=mean_function_constraints,
        variance_function_constraints=variance_function_constraints,
        exploration_set=exploration_set)

  def get_best_feasible_point(
      self, x: np.ndarray,
      sample_target_fnc: np.ndarray) -> float:
    """Select feasible point in sample_target_fnc by sampling from the constraints."""

    # If there are constraints we modify is_feasible otherwise the feasibility
    # is one for every input value and the best feasible point is the optimal
    # value in sample_target_fnc
    is_feasible = np.ones_like(x, dtype=bool)
    best_feasible_point = utilities.EVAL_FN[self.task](sample_target_fnc)

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
          mean = self.mean_function_constraints[p](x)
          variance = self.variance_function_constraints[p](x).clip(0)

        sample = np.random.normal(mean + self.jitter, np.sqrt(variance))

        is_feasible = (is_feasible) & (
            constraints_functions.EVAL_CONSTRAINT_OP[direction](sample, value))

      if np.any(is_feasible):
        # There is at least one feasible value. We get the optimal value among
        # the feasible points.
        best_feasible_point = utilities.EVAL_FN[self.task](
            sample_target_fnc[is_feasible])

    return best_feasible_point

  def get_global_opt(self, mean: np.ndarray, standard_deviation: np.ndarray,
                     task: Any, x: np.ndarray) -> float:
    """Get one value of feasible global optimum by sampling."""
    sample_target_fnc = np.random.normal(mean, standard_deviation)
    best_feasible_point = self.get_best_feasible_point(x, sample_target_fnc)
    return best_feasible_point

  def evaluate(self, x: np.ndarray) -> np.ndarray:
    """Evaluate the Constrained Expected Improvement."""
    # Compute get_improvement as in NoisyCausalExpectedImprovement
    # and get_probability_feasibility as in ConstrainedCausalExpectedImprovement
    return self.get_improvement(x) * self.get_probability_feasibility(x)[:, 0]

  def evaluate_to_store(self, x: np.ndarray)-> Tuple[np.ndarray, np.ndarray]:
    """Evaluate the improvement and probability of feasibility separately."""
    return self.get_improvement(x), self.get_probability_feasibility(x)[:, 0]
