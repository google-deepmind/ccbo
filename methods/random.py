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

"""Implementation of Causal Bayesian Optimisation."""
from __future__ import annotations

import collections
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from absl import logging
from networkx.classes import multidigraph
import numpy as np

from ccbo.methods import cbo
from ccbo.utils import utilities


class Random(cbo.CBO):
  """Causal Bayesian Optimisation class."""

  def __init__(
      self,
      graph: multidigraph.MultiDiGraph,
      scm: Any,
      make_scm_estimator: Callable[
          [
              multidigraph.MultiDiGraph,
              Dict[Tuple[Optional[Any], ...], Any],
              collections.OrderedDict[str, Any],
          ],
          Any,
      ],
      observation_samples: Dict[str, np.ndarray],
      intervention_domain: Dict[str, Sequence[float]],
      intervention_samples: Optional[
          Dict[Tuple[str, ...], Optional[Dict[str, Any]]]
      ],
      exploration_sets: List[Tuple[str, ...]],
      number_of_trials: int,
      ground_truth: Optional[Sequence[float]] = None,
      task: utilities.Task = utilities.Task.MIN,
      n_restart: int = 1,
      cost_type: int = 1,
      hp_prior: bool = True,
      num_anchor_points: int = 100,
      seed: int = 1,
      sample_anchor_points: bool = False,
      seed_anchor_points: Optional[int] = None,
      causal_prior: bool = True,
      verbose: bool = False,
      sampling_seed: Optional[int] = None,
      noisy_observations: bool = False,
      noisy_acquisition: bool = False,
      fix_likelihood_noise_var: bool = True,
      n_samples_per_intervention: int = 1,
      use_prior_mean: bool = False,
      size_intervention_grid: int = 100,
      use_true_scm: bool = False):
    super().__init__(
        graph=graph,
        scm=scm,
        make_scm_estimator=make_scm_estimator,
        observation_samples=observation_samples,
        intervention_domain=intervention_domain,
        intervention_samples=intervention_samples,
        exploration_sets=exploration_sets,
        task=task,
        cost_type=cost_type,
        number_of_trials=number_of_trials,
        ground_truth=ground_truth,
        n_restart=n_restart,
        num_anchor_points=num_anchor_points,
        verbose=verbose,
        seed=seed,
        sampling_seed=sampling_seed,
        noisy_observations=noisy_observations,
        noisy_acquisition=noisy_acquisition,
        n_samples_per_intervention=n_samples_per_intervention,
        fix_likelihood_noise_var=fix_likelihood_noise_var,
        size_intervention_grid=size_intervention_grid,
        use_true_scm=use_true_scm,
    )

    self.sample_anchor_points = sample_anchor_points
    self.seed_anchor_points = seed_anchor_points

  def run(self) -> None:
    """Run the BO loop."""
    # Get current target and best_es
    target = self.target_variable
    best_es = self.best_initial_es
    best_initial_level = self.best_initial_level

    for it in range(self.number_of_trials):
      logging.info("Trial: %s", it)

      if it == 0:
        # Store the initial set
        self.best_es_over_trials_list.append(best_es)
        self.best_level_over_trials_list.append(best_initial_level[0])
        # Observe the system
        self.observation_trial(best_es, target, it)
      else:
        # Intervene in the system
        self._intervention_trial(target, it)

    # Store optimal intervention
    self.optimal_intervention = {
        "set": self.best_es_over_trials,
        "level": self.best_level_over_trials,
        "outcome": self.optimal_outcome_values_during_trials[-1]
    }

  def _select_next_point(self, current_best_global_target: float,
                         it: int) -> Tuple[Tuple[str, ...], np.ndarray]:
    best_es = random.choice(self.exploration_sets)
    new_interventional_data_x = random.choice(
        self.interventional_grids[best_es])

    # Reshape point
    if len(new_interventional_data_x.shape) == 1 and len(best_es) == 1:
      new_interventional_data_x = utilities.make_column_shape_2d(
          new_interventional_data_x)
    elif len(best_es) > 1 and len(new_interventional_data_x.shape) == 1:
      new_interventional_data_x = new_interventional_data_x.reshape(1, -1)
      if new_interventional_data_x.shape[1] != len(best_es):
        new_interventional_data_x = np.transpose(new_interventional_data_x)
    else:
      raise ValueError("The new point is not an array.")
    return best_es, new_interventional_data_x
