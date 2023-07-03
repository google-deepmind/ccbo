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

"""Implementation of the causal acquisition functions."""

import functools
from typing import Any, Callable, Dict, List, Optional, OrderedDict, Tuple

from emukit.core import acquisition as acq
from emukit.core import parameter_space
from emukit.model_wrappers import gpy_model_wrappers
import numpy as np

from ccbo.acquisitions import constrained_expected_improvement as con_ei
from ccbo.acquisitions import expected_improvement
from ccbo.acquisitions import multitask_constrained_expected_improvement as multitask_con_ei
from ccbo.acquisitions import noisy_constrained_expected_improvement as noisy_con_ei
from ccbo.acquisitions import noisy_expected_improvement as noisy_ei
from ccbo.acquisitions import noisy_multitask_constrained_expected_improvement as noisy_multitask_ei
from ccbo.utils import cost_utils
from ccbo.utils import initialisation_utils as init_utils
from ccbo.utils import plotting_utils
from ccbo.utils import utilities


def numerical_optimization(
    acquisition: acq.Acquisition,
    inputs: np.ndarray,
    exploration_set: Tuple[str, ...],
) -> Tuple[Any, Any]:
  """Numerically optimize a function evaluating it on the inputs."""

  # Finds the new best point by evaluating the function in a set of inputs
  _, d = inputs.shape

  improvements = acquisition.evaluate(inputs)

  # Notice that here we always want to maximize the acquisition function as we
  # have multiplied the improvement with a minus sign when solving a
  # maximization problem.
  idx = np.argmax(improvements)

  # Get point with best improvement, the x new should be taken from the inputs
  x_new = inputs[idx]
  y_new = np.max(improvements)

  # Reshape point
  if len(x_new.shape) == 1 and len(exploration_set) == 1:
    x_new = utilities.make_column_shape_2d(x_new)
  elif len(exploration_set) > 1 and len(x_new.shape) == 1:
    x_new = x_new.reshape(1, -1)
  else:
    raise ValueError("The new point is not an array.")

  if x_new.shape[0] == d:
    # The function make_column_shape_2d might convert a (d, ) array
    # in a (d,1) array that needs to be reshaped
    x_new = np.transpose(x_new)

  assert x_new.shape[1] == inputs.shape[1], "New point has a wrong dimension"
  return y_new, x_new


def optimize_acquisition(
    acquisition: acq.Acquisition,
    intervention_domain: parameter_space.ParameterSpace,
    exploration_set: Tuple[str, ...],
    cost_functions: OrderedDict[str, Callable[[Any], Any]],
    target: str,
    num_anchor_points: int = 100,
    sample_anchor_points: bool = False,
    seed_anchor_points: Optional[int] = None,
)-> Tuple[np.ndarray, np.ndarray]:
  """Optimize the acquisition function rescaled by the cost."""
  assert isinstance(intervention_domain, parameter_space.ParameterSpace)
  dim = intervention_domain.dimensionality
  assert dim == len(exploration_set)

  cost_of_acquisition = cost_utils.Cost(cost_functions, exploration_set, target)
  acquisition_over_cost = (acquisition / cost_of_acquisition)

  if dim > 1:
    num_anchor_points = int(np.sqrt(num_anchor_points))

  if sample_anchor_points:
    # Ensure the points are different every time we call the function
    if seed_anchor_points is not None:
      np.random.seed(seed_anchor_points)
    else:
      np.random.seed()

    sampled_points = intervention_domain.sample_uniform(
        point_count=num_anchor_points)
  else:
    limits = [list(tup) for tup in intervention_domain.get_bounds()]
    sampled_points = init_utils.create_n_dimensional_intervention_grid(
        limits, num_anchor_points)

  y_new, x_new = numerical_optimization(acquisition_over_cost, sampled_points,
                                        exploration_set)

  return y_new, x_new


def evaluate_acquisition_function(
    intervention_domain: parameter_space.ParameterSpace,
    bo_model: Optional[gpy_model_wrappers.GPyModelWrapper],
    mean_function: Callable[[np.ndarray], np.ndarray],
    variance_function: Callable[[np.ndarray], np.ndarray],
    current_global_opt: float,
    exploration_set: Tuple[str, ...],
    cost_functions: OrderedDict[str, Callable[[Any], Any]],
    task: utilities.Task,
    target: str,
    noisy_acquisition: bool = False,
    num_anchor_points: int = 100,
    sample_anchor_points: bool = False,
    seed_anchor_points: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
  """Define and optimize the acquisition function for a unconstrained problem."""
  if noisy_acquisition:
    # When working with noisy observations different plausible
    # current_global_opt are obtained by repeatedly sampling from the bo_model
    expected_improvement_cls = noisy_ei.NoisyCausalExpectedImprovement
  else:
    expected_improvement_cls = functools.partial(
        expected_improvement.CausalExpectedImprovement, current_global_opt)

  acquisition = expected_improvement_cls(task, mean_function, variance_function,
                                         bo_model)

  y_acquisition, x_new = optimize_acquisition(
      acquisition, intervention_domain, exploration_set, cost_functions,
      target, num_anchor_points, sample_anchor_points, seed_anchor_points)

  if verbose:
    # Plot acquisition function. We avoid changing the global seed by
    # storing it and refixing it after the evaluation
    old_seed = np.random.get_state()
    np.random.seed(0)
    limits = [list(tup) for tup in intervention_domain.get_bounds()]
    sampled_points = init_utils.create_n_dimensional_intervention_grid(
        limits, num_anchor_points)
    improvement = acquisition.evaluate(sampled_points)

    np.random.set_state(old_seed)

    plotting_utils.plot_acquisition(sampled_points, improvement, x_new)

  return y_acquisition, x_new


def evaluate_constrained_acquisition_function(
    intervention_domain: parameter_space.ParameterSpace,
    bo_model: Optional[gpy_model_wrappers.GPyModelWrapper],
    mean_function: Callable[[np.ndarray], np.ndarray],
    variance_function: Callable[[np.ndarray], np.ndarray],
    current_global_opt: float,
    exploration_set: Tuple[str, ...],
    cost_functions: OrderedDict[str, Callable[[Any], Any]],
    task: utilities.Task,
    target: str,
    bo_model_constraints: Optional[gpy_model_wrappers.GPyModelWrapper],
    mean_function_constraints: Callable[[np.ndarray], np.ndarray],
    variance_function_constraints: Callable[[np.ndarray], np.ndarray],
    constraints: Dict[str, List[Any]], constraints_dict: Dict[Tuple[str, ...],
                                                              Dict[str, Any]],
    verbose: bool = False,
    noisy_acquisition: bool = False,
    num_anchor_points: int = 100,
    sample_anchor_points: bool = False,
    seed_anchor_points: Optional[int] = None,
    multi_task_model: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

  """Define and optimize the acquisition functions for a constrained problem."""
  # Define acquisition function
  if not multi_task_model:
    if noisy_acquisition:
      expected_improvement_cls = (
          noisy_con_ei.NoisyConstrainedCausalExpectedImprovement)
    else:
      expected_improvement_cls = functools.partial(
          con_ei.ConstrainedCausalExpectedImprovement, current_global_opt)
  else:
    if noisy_acquisition:
      expected_improvement_cls = (
          noisy_multitask_ei.NoisyMultiTaskConstrainedCausalExpectedImprovement)
    else:
      expected_improvement_cls = functools.partial(
          multitask_con_ei.MultiTaskConstrainedCausalExpectedImprovement,
          current_global_opt)

  acquisition = expected_improvement_cls(task, mean_function, variance_function,
                                         bo_model, bo_model_constraints,
                                         constraints, constraints_dict,
                                         mean_function_constraints,
                                         variance_function_constraints,
                                         exploration_set)

  # Get new point
  y_acquisition, x_new = optimize_acquisition(
      acquisition, intervention_domain, exploration_set, cost_functions,
      target, num_anchor_points, sample_anchor_points, seed_anchor_points)

  # Plot the acquisition function is es is one dimensional and verbose is True
  if verbose and len(exploration_set) == 1:
    # Plot acquisition function. We avoid changing the global seed by
    # storing it and refixing it after the evaluation
    old_seed = np.random.get_state()
    np.random.seed(0)
    # Evaluate improvement and feasibility separately to plot them
    limits = [list(tup) for tup in intervention_domain.get_bounds()]
    sampled_points = init_utils.create_n_dimensional_intervention_grid(
        limits, num_anchor_points)
    improvement, probability_feasibility = acquisition.evaluate_to_store(
        sampled_points)
    plotting_utils.plot_acquisition(
        sampled_points,
        improvement,
        probability_feasibility=probability_feasibility,
        multi_task_model=multi_task_model,
        x_new=x_new)
    np.random.set_state(old_seed)
  else:
    improvement = np.zeros((num_anchor_points, 1))
    probability_feasibility = np.zeros((num_anchor_points, 1))

  return y_acquisition, x_new, improvement, probability_feasibility
