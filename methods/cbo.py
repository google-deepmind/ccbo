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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from absl import logging
from networkx.classes import multidigraph
import numpy as np

from ccbo.acquisitions import evaluate_acquisitions
from ccbo.methods import base
from ccbo.utils import scm_utils
from ccbo.utils import utilities


class CBO(base.BaseMethod):
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
      ground_truth: Optional[Dict[Tuple[str, ...], Any]] = None,
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
        causal_prior=causal_prior,
        seed=seed,
        sampling_seed=sampling_seed,
        noisy_observations=noisy_observations,
        noisy_acquisition=noisy_acquisition,
        n_samples_per_intervention=n_samples_per_intervention,
        fix_likelihood_noise_var=fix_likelihood_noise_var,
        size_intervention_grid=size_intervention_grid,
        use_prior_mean=use_prior_mean,
        use_true_scm=use_true_scm,
        hp_prior=hp_prior)

    self.sample_anchor_points = sample_anchor_points
    self.seed_anchor_points = seed_anchor_points
    # Fit Gaussian processes to functions in the SCM
    self.fitted_scm_fncs = scm_utils.fit_scm_fncs(self.graph,
                                                  self.observational_samples,
                                                  self.scm_funcs,
                                                  self.n_restart)

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

  def _update_all_surrogate_models(self)-> None:
    """Update surrogate models (GPs) on the target functions."""
    for es in self.exploration_sets:
      if (self.interventional_data_x[es] is not None and
          self.interventional_data_y[es] is not None):
        self._update_bo_model(
            data_x=self.interventional_data_x,
            data_y=self.interventional_data_y,
            mean_functions=self.mean_function,
            variance_functions=self.variance_function,
            bo_model=self.bo_model,
            exploration_set=es,
            hp_prior=self.hp_prior,
            fix_likelihood_noise_var=self.fix_likelihood_noise_var,
            interventional_limits=self.interventional_variable_limits,
            n_samples_per_intervention=self.n_samples_per_intervention)

  def _intervention_trial(self, target: str, it: int)-> None:
    """Run one intervention trial of the BO loop."""

    # Update surrogate models if we have observe at the previous trial
    if self.trial_type[-1] == utilities.Trial.OBSERVATION:
      self._update_all_surrogate_models()

    # Run the actual per trial computation
    self._per_trial_computations(it, target)

  def observation_trial(
      self,
      best_es: Tuple[str, ...],
      target: str,
      it: int = 0,
      observation_cost: float = 0.0) -> Callable[
          [multidigraph.MultiDiGraph, Dict[Tuple[Optional[Any], ...],
                                           Any]], Any]:
    """Run one observation trial of the BO loop."""
    self.trial_type.append(utilities.Trial.OBSERVATION)
    # Given the fitted functions construct an approximate simulator
    fitted_scm = self.make_scm_estimator(self.graph, self.fitted_scm_fncs,
                                         self.scm_funcs)

    # Create mean functions and var functions (prior parameters) for the GPs
    # on the target functions. This is done using the observational data.
    self._update_sufficient_statistics(target=target, fitted_scm=fitted_scm)

    # Store the current optimal value. As we have observed at this trial
    # we don't have a new value so we store the previous one.
    self.optimal_outcome_values_during_trials.append(
        self.outcome_values[-1])
    if self.interventional_data_x[best_es] is None:
      self.optimal_intervention_levels[best_es][it] = np.nan

    # Store the cost of observing which is assumed to be zero
    self.per_trial_cost.append(observation_cost)

    return fitted_scm

  def _evaluate_acquisition_functions(self, current_best_global_target: float,
                                      it: int) -> None:
    """Evaluate the acquisition function given the surrogate models."""
    for es in self.exploration_sets:
      # Get the GP model for es
      if (
          self.interventional_data_x[es] is not None
          and self.interventional_data_y[es] is not None
      ):
        bo_model = self.bo_model[es]
      else:
        bo_model = None

      # The seed of the anchor points is used when the points at which
      # to evaluate the acquisition functions are sampled uniformly
      if self.seed_anchor_points is None:
        seed_to_pass = None
      else:
        # Use a fixed seed for reproducibility but ensure the anchor points
        # used for optimization of the acqusition functions are different
        seed_to_pass = int(self.seed_anchor_points * it)

      (
          self.y_acquired[es],
          self.corresponding_x[es],
      ) = evaluate_acquisitions.evaluate_acquisition_function(
          self.intervention_exploration_domain[es],
          bo_model,
          self.mean_function[es],
          self.variance_function[es],
          current_best_global_target,
          es,
          self.cost_functions,
          self.task,
          self.target_variable,
          noisy_acquisition=self.noisy_acquisition,
          num_anchor_points=self.num_anchor_points,
          sample_anchor_points=self.sample_anchor_points,
          seed_anchor_points=seed_to_pass,
          verbose=self.verbose)

  def _select_next_point(self, current_best_global_target: float,
                         it: int) -> Tuple[Tuple[str, ...], np.ndarray]:
    # Compute acquisition function given the updated BO models for the
    # interventional data. Notice that we use current_best_global_target
    # and the costs to compute the acquisition functions.
    self._evaluate_acquisition_functions(current_best_global_target, it)

    # Best exploration set based on acquired target-values.
    # Notice that independently on the maximization or minimization task
    # here we always need to optimize to select the point giving the maximum of
    # the expected improvement
    best_es = max(self.y_acquired, key=self.y_acquired.get)

    # Get the correspoding intervention value for best_es
    new_interventional_data_x = self.corresponding_x[best_es]
    self._check_new_point(best_es)
    return best_es, new_interventional_data_x
