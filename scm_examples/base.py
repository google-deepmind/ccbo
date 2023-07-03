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

"""Define a base class for all the DAG examples."""

from __future__ import annotations

import collections
import copy
from typing import Any, Dict, Optional, Tuple

from networkx.classes import multidigraph
import numpy as np

from ccbo.utils import constraints_functions
from ccbo.utils import initialisation_utils
from ccbo.utils import intervention_functions
from ccbo.utils import utilities


class ScmExample:
  """Base class for the SCM examples."""

  def __init__(self, variables: Dict[str, list[Any]],
               constraints: Dict[str, list[Any]],
               scm_funcs: collections.OrderedDict[str, Any],
               graph: multidigraph.MultiDiGraph):
    self.variables = variables
    self.constraints = constraints
    self.scm_funcs = scm_funcs
    self.graph = graph

  def get_target_name(self) -> str:
    # Get the name of the target variable as a string
    target_variable = [
        key for key, values in self.variables.items()
        if values[0] == utilities.VariableType.TARGET.value
    ][0]
    return target_variable

  def setup(
      self,
      exploration_sets: Optional[list[Tuple[str, ...]]] = None,
      n_grid_points: int = 100,
      n_samples: int = 1,
      sampling_seed: Optional[int] = 1,
      task: utilities.Task = utilities.Task.MIN,
  ) -> Tuple[list[str], list[Tuple[str, ...]], Dict[str, Any], Any,
             Dict[Tuple[str, ...], Optional[np.ndarray[Any, np.dtype]]],
             Dict[str, Dict[str, list[float]]], Any, Any, Any, Tuple[str, ...]]:
    """Setup variables and dictionaries needed for optimization."""
    dict_variables = self.variables
    scm_funcs = self.scm_funcs
    constraints = self.constraints

    # Get set of variables
    target_variable = self.get_target_name()

    manipulative_variables = [
        key for key, values in dict_variables.items()
        if values[0] == utilities.VariableType.MANIPULATIVE.value
    ]
    protected_variables = list(constraints.keys())

    # Get graph structure
    graph = self.graph

    # Specify all the exploration sets based on the manipulative variables
    if exploration_sets is None:
      exploration_sets = list(
          initialisation_utils.powerset(manipulative_variables))

    # Specify the intervention domain for each variable
    intervention_domain = {
        key: dict_variables[key][1] for key in manipulative_variables
    }

    # Set up a grid for each es and use it to find the best intervention value
    interventional_grids = initialisation_utils.get_interventional_grids(
        exploration_sets,
        intervention_domain,
        size_intervention_grid=n_grid_points,
    )

    # Compute unconstrained ground truth optimal interventions
    _, _, optimal_uncontrained_y, _, _, all_ce = (
        intervention_functions.get_optimal_interventions(
            exploration_sets=exploration_sets,
            interventional_grids=interventional_grids,
            scm_funcs=scm_funcs,
            graph=graph,
            model_variables=list(dict_variables.keys()),
            target_variable=target_variable,
            n_samples=n_samples,
            sampling_seed=sampling_seed,
        )
    )

    # Store the initial interventinal grid and all_ce before changing it
    # to find the constrained solution
    complete_interventional_grid = copy.deepcopy(interventional_grids)
    complete_all_ce = copy.deepcopy(all_ce)

    # Get number and constrained variables for each intervention set
    constraints_dict = constraints_functions.get_constraints_dict(
        exploration_sets, protected_variables, target_variable, graph)

    # Compute constraints functions on the interventional grids
    constraints_values = constraints_functions.compute_constraints_functions(
        exploration_sets,
        constraints_dict,
        interventional_grids,
        scm_funcs,
        graph,
        dict_variables,
        n_samples=n_samples,
        sampling_seed=sampling_seed)

    # Find optimal constrained output value
    is_feasible = False
    while not is_feasible:
      (is_feasible, optimal_set, opt_level,
       optimal_y) = constraints_functions.verify_feasibility(
           exploration_sets,
           all_ce,
           constraints,
           constraints_dict,
           scm_funcs,
           graph,
           dict_variables,
           interventional_grids,
           n_samples=n_samples,
           sampling_seed=sampling_seed,
           task=task)

      if not is_feasible:
        # Remove unfeasible target values
        # Remove unfeasible intervention level from interventional grid
        all_ce[optimal_set].remove(optimal_y)
        condition = True
        for i in range(opt_level.shape[0]):
          condition = condition & (
              interventional_grids[optimal_set][:, i] == opt_level[i])

        interventional_grids[optimal_set] = interventional_grids[optimal_set][
            ~condition]

    return (manipulative_variables, exploration_sets, intervention_domain,
            complete_all_ce, complete_interventional_grid, constraints_values,
            optimal_uncontrained_y, optimal_y, opt_level, optimal_set)
