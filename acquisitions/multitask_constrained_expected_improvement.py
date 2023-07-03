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

from ccbo.acquisitions import expected_improvement
from ccbo.utils import constraints_functions
from ccbo.utils import utilities


class MultiTaskConstrainedCausalExpectedImprovement(
    expected_improvement.CausalExpectedImprovement):
  """Implementation of the causal constrained EI acquisition function.

  This function computes the constrained expected improvement with respect to
  the joint distribution p(f, G) where f is the target function and G is the
  set of constraints functions. When the direction is > for all constraints this
  function evaluates E_{p(f,G)}[max(0, y* - f)*p(G > lambda)] where lambda is
  the vector of threshold values and y* is the currently best feasible value
  observed. As f and G are jointly modelled via a multi-task
  GP model (ICM) their joint distribution does not factorise and the resulting
  constrained expected improvement cannot be computed as the product of expected
  improvement and probability of feasibility. We approximate this expectation
  via Monte Carlo integration.
  """

  def __init__(self, current_global_opt: Optional[float], task: utilities.Task,
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
        "current_global_opt": current_global_opt,
        "task": task,
        "mean_function": mean_function,
        "variance_function": variance_function,
        "model": model
    }
    super().__init__(**base_args)

    self.model_constraints = model_constraints
    self.constraints = constraints
    self.constraints_dict = constraints_dict
    self.mean_function_constraints = mean_function_constraints
    self.variance_function_constraints = variance_function_constraints
    self.exploration_set = exploration_set

  def get_improvement(self,
                      x: np.ndarray,
                      montecarlo_estimate: bool = True,
                      n_samples: int = 1000) -> np.ndarray:
    """Evaluate the Constrained Expected Improvement.

    When using a multi-task model the target function and the constraints
    are correlated thus the acquisition cannot be factorized in an
    improvement * probability of feasibility. We thus compute it by sampling
    from the joint distribution of the function and the constraints.

    Args:
      x: the values at which to evaluate the acquisition.
      montecarlo_estimate: whether to use a Monte Carlo estimate or not.
      n_samples: number of samples to use to get a Monte Carlo estimate.

    Returns:
      The constrained expected improvement estimated at
      x via Monte Carlo integration using n_samples.
    """

    if montecarlo_estimate:
      # Sample from the target function
      is_feasible = self.get_feasibility(x, n_samples)
      sample_f = self.get_samples_target_function(x, n_samples)
      diff = self.current_global_opt - sample_f
      ei = [
          np.max((np.repeat(0, diff[i].shape[0]), diff[i]), axis=0)
          for i in range(n_samples)
      ]
      improvement = np.mean(is_feasible * ei, axis=0)
    else:
      raise NotImplementedError(
          "Other ways of computing this functions are not implemented")
    return improvement

  def get_samples_target_function(self, x: np.ndarray,
                                  n_samples: int) -> np.ndarray:
    # Sample from the target function
    x_f_augmented = np.concatenate(
        [x, np.zeros_like(x)], axis=1)
    sample_f = self._sample_from_model(x_f_augmented, self.model, n_samples)
    return sample_f

  def _sample_from_model(self, x: np.ndarray,
                         model: gpy_model_wrappers.GPyModelWrapper,
                         n_samples: int, seed: Optional[int] = 0) -> np.ndarray:
    # Sample from GP model
    # We avoid changing the seed of the algorithm by storing it, sampling the
    # functions and then resetting the old seed
    if seed:
      old_seed = np.random.get_state()
      np.random.seed(seed)

    mean, _ = model.predict(x)
    cov = model.predict_covariance(x)
    sample = np.random.multivariate_normal(mean[:, 0], cov, n_samples)

    if seed:
      np.random.set_state(old_seed)
    return sample

  def get_feasibility(self, x: np.ndarray, n_samples: int) -> np.ndarray:
    is_feasible = np.ones((n_samples, x.shape[0])).astype(bool)
    # Sample from the constraint functions - if there are no constraints
    # is_feasible is not changed and all x values are feasible.
    if self.model_constraints:
      for i, model_c in enumerate(
          list(self.model_constraints.values())):
        x_augmented = np.concatenate(
            [x, np.ones_like(x)*(i+1)], axis=1)
        sample_g = self._sample_from_model(x_augmented, model_c, n_samples)

        # Get the direction and threshold for the current variable
        c_var = self.constraints_dict[self.exploration_set]["vars"][i]
        direction = self.constraints[c_var][0]
        threshold = self.constraints[c_var][1]

        # Check feasibility of all samples for the current variable and merge
        # it with the feasibility of the previous one(s)
        is_feasible = (
            is_feasible & constraints_functions.EVAL_CONSTRAINT_OP[direction](
                sample_g, threshold))
    return is_feasible

  def probability_feasibility(self, x: np.ndarray) -> Optional[None]:
    raise NotImplementedError(
        "Computation of the probability of feasibility is not implemented.")

  def evaluate_to_store(self, x: np.ndarray)-> Any:
    """Evaluate the improvement and probability of feasibility separately."""
    return self.get_improvement(x), np.zeros((x.shape[0], 1))
