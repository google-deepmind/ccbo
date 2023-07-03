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

"""Functions to compute constraints related quantities."""
from __future__ import annotations

import collections
import copy
import operator
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import networkx
from networkx.classes import multidigraph
import numpy as np

from ccbo.utils import intervention_functions
from ccbo.utils import utilities

EVAL_CONSTRAINT_OP = {
    utilities.Direction.LOWER: operator.lt,
    utilities.Direction.HIGHER: operator.gt
}


def best_feasible_initial_es(initial_feasibility: Dict[Any, Any],
                             best_es: Tuple[str, ...], best_level: Any,
                             interventional_data_y: Dict[Tuple[str, ...], Any],
                             interventional_data_x: Dict[Tuple[str, ...], Any],
                             task: Any) -> Tuple[Any, Any]:
  """Check if initial best es if feasible and compute otherwise."""
  feasible_best_es = best_es
  feasible_best_level = best_level
  feasible_sets = {}
  for key, value in initial_feasibility.items():
    if list(value.values()):
      # There exists constraints for this key thus we check if there is at least
      # one feasible initial value
      feasible_sets[key] = np.any(list(value.values()))
    else:
      # There are no constraints for this key thus the set is feasible
      feasible_sets[key] = True

  # Check if best_es is feasible. If yes return it otherwise recompute the
  # best feasible set by filtering out sets that are not feasible and computing
  # the task (min or max) among the filtered values
  if not feasible_sets[best_es]:
    feasible_interventional_data_y = {
        key: task(interventional_data_y[key])
        for key, val in feasible_sets.items()
        if val
    }

    # If the filtered dict is not empty we select the set among the keys
    # otherwise we return the initial best_es
    if feasible_interventional_data_y:
      feasible_best_es = task(
          feasible_interventional_data_y,
          key=feasible_interventional_data_y.get)
      feasible_best_level = interventional_data_x[feasible_best_es]

  return feasible_best_es, feasible_best_level


def initialise_constraints_interventional_objects(
    exploration_sets: Sequence[Set[str]],
    intervention_samples: Dict[Tuple[str, ...], Dict[str, Any]],
    interventional_data_y_constraints: Dict[Any, Any],
    bo_model_constraints: Dict[Any, Any], initial_feasibility: Dict[Any, Any],
    constraints: Dict[str, List[Any]], feasibility: Dict[Any, Any]) -> None:
  """Initialise interventional data for the constraints."""
  assert isinstance(intervention_samples, dict)

  for es in exploration_sets:
    if es not in intervention_samples:
      # No interventional data
      pass
    else:
      # Interventional data contains a dictionary of dictionaries
      # each corresponding to one type (es) of intervention.
      # es on keys and nd.array on values

      data_subset = intervention_samples[es]

      for var in list(bo_model_constraints[es].keys()):
        # Removing the temporal index
        value = data_subset[var].reshape(-1, 1)
        interventional_data_y_constraints[es][var] = value

        # Check if the point is feasible or not
        initial_feasibility[es][var] = EVAL_CONSTRAINT_OP[
            constraints[var][0]](value, constraints[var][1])

      feasibility[es] = [int(all(list(initial_feasibility[es].values())))]


def get_constraints_dicts(
    exploration_sets: List[Tuple[str, ...]],
    constraints: Dict[str, List[float]],
    graph: multidigraph.MultiDiGraph,
    target_variable: str,
    observational_samples: Dict[str, np.ndarray]
) -> Tuple[Dict[Any, Dict[str, Union[int, Any]]], Dict[Any, Any], Dict[
    Any, Any], Dict[Any, Any], Dict[Any, Any]]:
  """Initialise dictionaries of the constraints."""
  bo_model_constraints = {es: None for es in exploration_sets}
  initial_feasibility = {es: None for es in exploration_sets}
  # Initialize object to store the feasibility of the interventional points
  feasibility = {es: None for es in exploration_sets}

  constraints_dict = {}

  protected_variables = list(constraints.keys())

  for es in exploration_sets:
    # For every es, select the variables appearning in the constraints
    # Note that every es has a different number of constraints and vars

    for var in es:
      if var in protected_variables:
        # Get C(X)
        protected_variables.remove(var)

    descendants_vars = list(set().union(*[
        list(networkx.descendants(graph, element_es))
        for element_es in list(es)
    ]))
    descendants_vars.remove(target_variable)
    # The relevant constraints for X are denoted by c_x. This includes the
    # protected variables that are also descendant of X and are not themselves
    # included in es.
    c_x = [var for var in protected_variables if var in descendants_vars]

    # Store the constrainted variables and their number
    constraints_dict[es] = {
        'num': len(c_x),
        'vars': c_x
    }

    # Initialize models for the constrained
    bo_model_constraints[es] = {var: None for var in c_x}

    # Initialise feasibility stores the feasibility of the initial
    # interventional data if provided. It is initialized to 0.
    initial_feasibility[es] = {var: 0 for var in c_x}

    # Check that threshold value is not None or assign a value using d_o
    if es in constraints:
      if constraints[es][1] is None:  # check the threshold
        assert observational_samples is not None, (
            'Specify threshold values or provide D_O')
        constraints[es][1] = np.mean(observational_samples[es][:, 0])

  interventional_data_y_constraints = copy.deepcopy(bo_model_constraints)

  return (constraints_dict, initial_feasibility, bo_model_constraints,
          feasibility, interventional_data_y_constraints)


def compute_constraints_functions(
    exploration_sets: Optional[Any],
    constraints_dict: Dict[str, Dict[str, Any]],
    interventional_grids: Dict[Any, Any],
    scm_funcs: collections.OrderedDict[str, Any],
    graph: multidigraph.MultiDiGraph,
    dict_variables: Dict[str, List[Any]],
    sampling_seed: int,
    n_samples: int = 1) -> Dict[str, Dict[str, List[float]]]:
  """Compute ground truth functions for the constraints."""

  constraints_values_dict = {}

  for es in exploration_sets:
    constraints_values_dict[es] = {}
    for j in range(constraints_dict[es]['num']):
      c_target = constraints_dict[es]['vars'][j]
      _, _, _, _, _, ce_constraints = (
          intervention_functions.get_optimal_interventions(
              graph=graph,
              exploration_sets=[es],
              interventional_grids=interventional_grids,
              scm_funcs=scm_funcs,
              model_variables=list(dict_variables.keys()),
              target_variable=c_target,
              n_samples=n_samples,
              sampling_seed=sampling_seed,
          )
      )

      constraints_values_dict[es][c_target] = ce_constraints[es]
  return constraints_values_dict


def get_constraints_dict(
    exploration_sets: Optional[Any],
    protected_variables: List[str],
    target_variable: str,
    graph: multidigraph.MultiDiGraph) -> Dict[str, Dict[str, Any]]:
  """Get number and constrained variables for each intervention."""
  constraints_dict = {}
  for es in exploration_sets:
    # For every es, select the variables appearing in the constraints
    # Note that every es has a different number of constraints and vars
    for var in es:
      if var in protected_variables:
        # Get P(X)
        protected_variables.remove(var)

    descendants_vars = list(set().union(*[
        list(networkx.descendants(graph, element_es))
        for element_es in list(es)
    ]))

    descendants_vars.remove(target_variable)

    # c_x are the variables that are constrained for the intervention set X
    c_x = [value for value in protected_variables if value in descendants_vars]

    # Store the constrainted variables and their number
    constraints_dict[es] = {
        'num': len(c_x),
        'vars': c_x
    }
  return constraints_dict


def verify_feasibility(
    exploration_sets: List[Tuple[str, ...]],
    all_ce: Dict[Tuple[str, ...], List[Any]],
    constraints: Dict[str, List[Any]], constraints_dict: Dict[str,
                                                              Dict[str, Any]],
    scm_funcs: collections.OrderedDict[str, Any],
    graph: multidigraph.MultiDiGraph,
    dict_variables: Dict[str, List[Any]],
    interventional_grids: Dict[Tuple[str, ...], Optional[np.ndarray]],
    sampling_seed: int,
    optimal_level: Optional[np.ndarray] = None,
    optimal_unconstrained_set: Optional[Tuple[str, ...]] = None,
    n_samples: int = 1,
    task: utilities.Task = utilities.Task.MIN
) -> Tuple[bool, Tuple[str, ...], Any, Any]:
  """Verify feasibility and get constrained solution."""

  # Optimal unconstrained solution
  if optimal_unconstrained_set is None:
    optimal_unconstrained_set = exploration_sets[utilities.ARG_EVAL_FN[task](([
        utilities.EVAL_FN[task](all_ce[var])
        for var in exploration_sets
        if all_ce[var]
    ]))]
  if optimal_level is None:
    optimal_level = interventional_grids[optimal_unconstrained_set][
        utilities.ARG_EVAL_FN[task](all_ce[optimal_unconstrained_set])]

  optimal_y = utilities.EVAL_FN[task](all_ce[optimal_unconstrained_set])

  feasibility_list = []
  if constraints_dict[optimal_unconstrained_set]['num'] == 0:
    # No constraints
    feasibility_list.append(1)
  else:
    # Get value for the constraints
    for p in constraints_dict[optimal_unconstrained_set]['vars']:
      (_, _, _, _, _,
       constrain_values) = intervention_functions.get_optimal_interventions(
           exploration_sets=[optimal_unconstrained_set],
           interventional_grids={optimal_unconstrained_set: [optimal_level]},
           scm_funcs=scm_funcs,
           graph=graph,
           model_variables=list(dict_variables.keys()),
           target_variable=p,
           n_samples=n_samples,
           sampling_seed=sampling_seed)

      # Check if constrain is satisfied
      tmp = EVAL_CONSTRAINT_OP[constraints[p][0]](
          constrain_values[optimal_unconstrained_set][0], constraints[p][1])

      feasibility_list.append(tmp)

  # Check if all constraints are satisfied
  is_feasible = all(feasibility_list)

  return (is_feasible, optimal_unconstrained_set, optimal_level, optimal_y)
