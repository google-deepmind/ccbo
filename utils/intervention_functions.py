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

"""Intervention function utilities."""
from __future__ import annotations

import collections
import copy
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from absl import logging
from networkx.classes import multidigraph
import numpy as np

from ccbo.utils import initialisation_utils
from ccbo.utils import sampling_utils
from ccbo.utils import utilities


def assign_initial_intervention_level(
    exploration_set: Tuple[str, ...],
    intervention_level: np.ndarray,
    variables: Sequence[str]
) -> Dict[str, Any]:
  """Intervention assignment."""
  intervention_blanket = {key: None for key in variables}

  if len(exploration_set) == 1:
    # Assign the intervention
    intervention_blanket[exploration_set[0]] = float(intervention_level)

  else:
    # Intervention happening on _multiple_ variables
    for variable, lvl in zip(exploration_set, np.transpose(intervention_level)):
      # Assign the intervention
      intervention_blanket[variable] = float(lvl)

  return intervention_blanket


def evaluate_target_function(
    scm_funcs: collections.OrderedDict[str, Any],
    exploration_set: Tuple[str, ...],
    all_vars: Sequence[str],
    noisy_observations: bool = False,
    seed: Optional[int] = None,
    n_samples_for_interventions: int = 1
) -> Callable[[str, np.ndarray], float]:
  """Evaluates the target function."""

  def compute_target_function(target: str,
                              intervention_levels: np.ndarray) -> float:

    # Assign interventions
    interventions = assign_initial_intervention_level(
        exploration_set=exploration_set,
        intervention_level=intervention_levels,
        variables=all_vars
    )

    # Set the noise in the SCM to be equal or different from zero
    if not noisy_observations:
      # We sample from the SCM with zero noise therefore we have no randomess
      # and don't need to average over samples
      n_samples = 1
    else:
      assert n_samples_for_interventions > 1, ("Noisy evaluations require a set"
                                               " of samples to compute the "
                                               "average causal effects.")
      n_samples = n_samples_for_interventions

    if seed is not None:
      np.random.seed(seed)

    # Sample from the true interventional distribution
    interventional_samples = sampling_utils.sample_scm(
        scm_funcs=scm_funcs,
        graph=None,
        interventions=interventions,
        n_samples=n_samples,
        compute_moments=True,
        moment=0,
        seed=seed)

    # Compute the avearage effect of intervention(s) that is the target function
    target_response = float(interventional_samples[target])
    return target_response

  return compute_target_function


def get_optimal_interventions(
    exploration_sets: List[Tuple[str, ...]],
    interventional_grids: Dict[Any, Any],
    scm_funcs: collections.OrderedDict[str, Any],
    graph: multidigraph.MultiDiGraph,
    model_variables: List[str],
    target_variable: str,
    task: utilities.Task = utilities.Task.MIN,
    n_samples: int = 1,
    sampling_seed: Optional[int] = None
) -> Tuple[Any, ...]:
  """Gets the optimal interventions across exploration sets."""

  logging.warning("Set high number of n_samples to ensure good estimation of "
                  "the ground truth but remember that the higher the number "
                  "of samples the slower the computations.")

  assert target_variable in model_variables

  optimal_interventions = {setx: None for setx in exploration_sets}

  y_stars = copy.deepcopy(optimal_interventions)
  interventions = initialisation_utils.make_intervention_dict(graph)

  ce = {es: [] for es in exploration_sets}

  # E[Y | do( . )]
  for s in exploration_sets:
    # Reset intervention to avoid carrying over levels from
    # previous exploration set
    intervention_on_s = copy.deepcopy(interventions)
    for level in interventional_grids[s]:
      # Univariate intervention
      if len(s) == 1:
        intervention_on_s[s[0]] = float(level)
      # Multivariate intervention
      else:
        for var, val in zip(s, level):
          intervention_on_s[var] = val

      # Sample from the true interventional distribution
      out = sampling_utils.sample_scm(
          scm_funcs=scm_funcs,
          graph=None,
          interventions=intervention_on_s,
          n_samples=n_samples,
          compute_moments=True,
          moment=0,
          seed=sampling_seed)

      ce[s].append(out[target_variable])

  local_target_values = []
  for s in exploration_sets:
    if task.value == utilities.Task.MIN.value:
      idx = np.array(ce[s]).argmin()
    else:
      idx = np.array(ce[s]).argmax()
    local_target_values.append((s, idx, ce[s][idx]))
    y_stars[s] = ce[s][idx]
    optimal_interventions[s] = interventional_grids[s][idx]

  # Find best intervention
  best_s, best_idx, best_objective_value = min(local_target_values,
                                               key=lambda t: t[2])
  best_s_value = interventional_grids[best_s][best_idx]

  return (
      best_s_value,
      best_s,
      best_objective_value,
      y_stars,
      optimal_interventions,
      ce,
  )
