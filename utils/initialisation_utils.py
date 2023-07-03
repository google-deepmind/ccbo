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

"""General utilities to initialise dicts and object to store results."""
from __future__ import annotations

import copy
import itertools
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from emukit import core
from networkx.classes import multidigraph
import numpy as np

from ccbo.utils import sampling_utils


def powerset(iterable: List[str])-> itertools.chain[Tuple[str, ...]]:
  """Compute the power set of a list of values."""
  # this returns e.g. powerset([1,2]) --> (1,) (2,) (1,2)
  s = list(iterable)
  return itertools.chain.from_iterable(
      itertools.combinations(s, r) for r in range(1,
                                                  len(s) + 1))


def create_n_dimensional_intervention_grid(limits: Any,
                                           size_intervention_grid: int = 100
                                          ) -> np.ndarray:
  """Usage: combine_n_dimensional_intervention_grid([[-2,2],[-5,10]],10)."""
  if not any(isinstance(el, list) for el in limits):
    # We are just passing a single list
    return np.linspace(limits[0], limits[1], size_intervention_grid)[:, None]
  else:
    extrema = np.vstack(limits)
    inputs = [
        np.linspace(i, j, size_intervention_grid)
        for i, j in zip(extrema[:, 0], extrema[:, 1])
    ]
    return np.dstack(np.meshgrid(*inputs)).ravel("F").reshape(len(inputs), -1).T


def assign_interventions(
    variables: Tuple[str, ...], levels: Tuple[float, ...],
    n_samples_per_intervention: int,
    sampling_seed: int, d_i: Dict[Tuple[str, ...],
                                  Any], graph: multidigraph.MultiDiGraph,
    scm_funcs: Any) -> Dict[Tuple[str, ...], Dict[str, Any]]:
  """Assign initial intervention levels to d_i."""
  intervention = make_intervention_dict(graph)
  for var, level in zip(variables, levels):
    intervention[var] = level

  # Sample from the true interventional distribution
  intervention_samples = sampling_utils.sample_scm(
      scm_funcs=scm_funcs,
      graph=None,
      interventions=intervention,
      n_samples=n_samples_per_intervention,
      compute_moments=True,
      moment=0,
      seed=sampling_seed)

  d_i[variables] = intervention_samples
  return d_i


def get_interventional_grids(
    exploration_set: List[Tuple[str, ...]],
    intervention_limits: Dict[str, Sequence[float]],
    size_intervention_grid: int = 100
) -> Dict[Tuple[str, ...], Optional[np.ndarray]]:
  """Build the n-dimensional interventional grids for the exploration sets."""
  # Create grids
  intervention_grid = {k: None for k in exploration_set}
  for es in exploration_set:
    if len(es) == 1:
      intervention_grid[es] = create_n_dimensional_intervention_grid(
          intervention_limits[es[0]], size_intervention_grid)
    else:
      if size_intervention_grid >= 100 and len(es) > 1:
        # Reduce number of point to reduce computational cost of evaluating
        # the acquisition function.
        size_intervention_grid = 10

      intervention_grid[es] = create_n_dimensional_intervention_grid(
          [intervention_limits[j] for j in es], size_intervention_grid)
  return intervention_grid


def make_intervention_dict(graph: multidigraph.MultiDiGraph) -> Dict[str, Any]:
  return {v: None for v in graph.nodes}


def initialise_interventional_objects(
    exploration_sets: List[Tuple[str, ...]],
    d_i: Dict[Tuple[str, ...], Dict[str, Any]],
    target: str,
    task: Any,
    nr_interventions: Optional[int] = None,
) -> Tuple[List[Tuple[str, ...]], Optional[List[np.ndarray]],
           List[Optional[np.ndarray]], Dict[Tuple[str, ...],
                                            Optional[np.ndarray]],
           Dict[Tuple[str, ...], Optional[np.ndarray]]]:
  """Initialize interventional dataset."""

  assert isinstance(d_i, dict)
  target_values = {es: None for es in exploration_sets}

  interventions = copy.deepcopy(target_values)
  intervention_data_x = copy.deepcopy(target_values)
  intervention_data_y = copy.deepcopy(target_values)

  for es in exploration_sets:
    if es not in d_i:
      pass
    else:
      # Interventional data contains a dictionary of dictionaries.
      # Each corresponds to one type (es) of intervention.
      interventional_samples = d_i[es]  # es on keys and nd.array on values

      assert isinstance(interventional_samples,
                        dict), (es, type(interventional_samples), d_i)
      assert target in interventional_samples
      assert isinstance(interventional_samples[target], np.ndarray)

      if nr_interventions:
        raise NotImplementedError("Not yet implemented")
      else:
        # Only have one interventional sample per intervention to start with
        data_subset = interventional_samples

      # Find the corresponding target values at these coordinates [array]
      target_values[es] = np.array(data_subset[target]).reshape(-1, 1)
      assert target_values[es] is not None

      # Find the corresponding interventions [array]
      if len(es) == 1:
        interventions[es] = np.array(data_subset[es[0]]).reshape(-1, 1)
      else:
        tmp = []
        for var in es:
          tmp.append(data_subset[var])
        interventions[es] = np.expand_dims(np.hstack(tmp), axis=0)
      assert interventions[es] is not None

      # Set the interventional data to use
      intervention_data_x[es] = interventions[es]
      intervention_data_y[es] = target_values[es]

      assert intervention_data_x[es] is not None
      assert intervention_data_y[es] is not None

  # Get best intervention set at each time index
  target_values = {k: v for k, v in target_values.items() if v is not None}
  best_es = task(target_values, key=target_values.get)

  # Interventions
  best_intervention_level = interventions[best_es]
  # Outcomes
  best_target_value = target_values[best_es]

  # Use the best outcome level at t=0 as a prior for all the other timesteps
  best_es_sequence = [best_es]
  best_intervention_levels = [best_intervention_level]
  best_target_levels = [best_target_value]

  return (
      best_es_sequence,
      best_target_levels,
      best_intervention_levels,
      intervention_data_x,
      intervention_data_y,
  )


def initialise_global_outcome_dict_new(
    initial_optimal_target_values: Optional[List[np.ndarray]],
    blank_val: float) -> List[float]:
  """Initialize dict of outcome values."""
  assert isinstance(initial_optimal_target_values, list)
  targets = []
  if initial_optimal_target_values[0]:
    targets.append(float(initial_optimal_target_values[0]))
  else:
    # No interventional data was provided initially so the list is empty.
    targets.append(blank_val)
  return targets


def initialise_optimal_intervention_level_list(
    exploration_sets: List[Tuple[str, ...]],
    initial_optimal_sequential_intervention_sets: Tuple[str, ...],
    initial_optimal_sequential_intervention_levels: List[np.ndarray],
    number_of_trials: int) -> Dict[Tuple[str, ...], Any]:
  """Initialize list of optimal intervention levels."""
  intervention_levels = {
      es: number_of_trials * [None] for es in exploration_sets
  }

  #  Add interventional data that we have at start
  for es in exploration_sets:
    if es == initial_optimal_sequential_intervention_sets:
      intervention_levels[es].insert(
          0, initial_optimal_sequential_intervention_levels)
    else:
      intervention_levels[es].insert(0, None)

  return intervention_levels


def make_parameter_space_for_intervention_set(
    exploration_set: Tuple[str, ...],
    lower_limit: Union[List[float], float],
    upper_limit: Union[List[float], float],
) -> core.ParameterSpace:
  """Set ParameterSpace of intervention for one exploration_set."""
  assert isinstance(exploration_set, tuple)
  if len(exploration_set) == 1:
    assert isinstance(lower_limit, float)
    assert isinstance(upper_limit, float)
    return core.ParameterSpace([
        core.ContinuousParameter(
            str(exploration_set), lower_limit, upper_limit)
    ])
  else:
    multivariate_limits = []
    assert len(exploration_set) == len(lower_limit), exploration_set
    assert len(exploration_set) == len(upper_limit), exploration_set
    for i, var in enumerate(exploration_set):
      multivariate_limits.append(
          core.ContinuousParameter(str(var), lower_limit[i], upper_limit[i]))
    return core.ParameterSpace(multivariate_limits)


def create_intervention_exploration_domain(
    exploration_sets: List[Tuple[str, ...]],
    interventional_variable_limits: Dict[str, Sequence[float]]
) -> Dict[Tuple[str, ...], Any]:
  """Get intervention domain for exploration_set."""
  intervention_exploration_domain = {es: None for es in exploration_sets}
  for es in exploration_sets:
    if len(es) == 1:
      assert es[0] in interventional_variable_limits.keys()
      ll = float(min(interventional_variable_limits[es[0]]))
      ul = float(max(interventional_variable_limits[es[0]]))
    else:
      ll, ul = [], []  # lower-limit and upper-limit
      for var in es:
        ll.append(float(min(interventional_variable_limits[var])))
        ul.append(float(max(interventional_variable_limits[var])))
      assert len(es) == len(ul) == len(ll)
    # Assign
    intervention_exploration_domain[
        es] = make_parameter_space_for_intervention_set(es, ll, ul)
  return intervention_exploration_domain

