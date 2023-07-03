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

"""Implementation of the multi-task causal constrained acquisition functions."""

from typing import Any, Callable, Dict, List, Optional, Tuple

from emukit.model_wrappers import gpy_model_wrappers
import numpy as np

from ccbo.acquisitions import multitask_constrained_expected_improvement as multitask_con_ei
from ccbo.utils import utilities


class NoisyMultiTaskConstrainedCausalExpectedImprovement(
    multitask_con_ei.MultiTaskConstrainedCausalExpectedImprovement):
  """Implementation of the causal constrained EI acquisition function."""

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
               exploration_set: Tuple[str, ...]) -> None:

    base_args = {
        "current_global_opt": None,
        "task": task,
        "mean_function": mean_function,
        "variance_function": variance_function,
        "model": model,
        "model_constraints": model_constraints,
        "constraints": constraints,
        "constraints_dict": constraints_dict,
        "mean_function_constraints": mean_function_constraints,
        "variance_function_constraints": variance_function_constraints,
        "exploration_set": exploration_set
    }
    super().__init__(**base_args)

  def get_global_opt(self, all_sample_f: np.ndarray, is_feasible: np.ndarray
                     ) -> List[float]:
    """Get one value of feasible global optimum by sampling."""

    best_feasible_points = []
    for one_sample_f, one_is_feasible in zip(all_sample_f, is_feasible):
      if np.any(one_sample_f[one_is_feasible]):
        best_feasible_point = utilities.EVAL_FN[self.task](
            one_sample_f[one_is_feasible])
      else:
        best_feasible_point = utilities.EVAL_FN[self.task](one_sample_f)

      best_feasible_points.append(best_feasible_point)

    return best_feasible_points

  def get_improvement(self,
                      x: np.ndarray,
                      montecarlo_estimate: bool = True,
                      n_samples: int = 100,
                      n_samples_min: int = 10) -> np.ndarray:
    """Evaluate the Constrained Expected Improvement."""

    if montecarlo_estimate:
      # Sample from the target function
      is_feasible = self.get_feasibility(x, n_samples_min + n_samples)
      all_sample_f = self.get_samples_target_function(x,
                                                      n_samples_min + n_samples)
      # Get the optimal feasible value for each sample
      current_feasible_global_opt = self.get_global_opt(
          all_sample_f[:n_samples_min, :], is_feasible[:n_samples_min, :])

      sample_f = all_sample_f[n_samples_min:, :]
      sample_g = is_feasible[n_samples_min:, :]
      out = []
      for i in range(n_samples_min):
        diff = np.ones_like(sample_f)*current_feasible_global_opt[i] - sample_f
        ei = np.vstack([
            np.max((np.zeros(diff[i].shape[0]), diff[i]), axis=0)
            for i in range(n_samples)
        ])
        out.append(np.mean(sample_g * ei, axis=0))
      improvement = np.mean(np.vstack(out), axis=0)
    else:
      raise NotImplementedError(
          "Other ways of computing this functions are not implemented")
    return improvement
