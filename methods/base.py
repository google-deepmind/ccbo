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

"""Abstract class for Bayesian optimization methods."""
from __future__ import annotations

import abc
import collections
import copy
import functools
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from absl import logging
from emukit.model_wrappers import gpy_model_wrappers
from GPy import core
from GPy import models
from GPy.core.parameterization import priors
from GPy.kern.src import rbf
from networkx.classes import multidigraph
import numpy as np

from ccbo.kernels import causal_kernel
from ccbo.scm_examples import base
from ccbo.utils import cost_utils
from ccbo.utils import gp_utils
from ccbo.utils import initialisation_utils
from ccbo.utils import intervention_functions
from ccbo.utils import plotting_utils
from ccbo.utils import utilities


class BaseMethod(abc.ABC):
  """Base class with common methods and variables for all BO methods."""

  def __init__(
      self,
      graph: multidigraph.MultiDiGraph,
      scm: base.ScmExample,
      observation_samples: Dict[str, np.ndarray],
      intervention_domain: Dict[str, Sequence[float]],
      exploration_sets: List[Tuple[str, ...]],
      intervention_samples: Optional[Dict[Tuple[str, ...],
                                          Dict[str, Any]]] = None,
      make_scm_estimator: Optional[Callable[[
          multidigraph.MultiDiGraph, Dict[Tuple[Optional[Any], ...], Any],
          collections.OrderedDict[str, Any]], Any]] = None,
      task: utilities.Task = utilities.Task.MIN,
      cost_type: int = 1,
      number_of_trials: int = 10,
      ground_truth: Optional[Dict[Tuple[str, ...], Any]] = None,
      n_restart: int = 1,
      num_anchor_points: int = 100,
      verbose: bool = False,
      size_intervention_grid: int = 100,
      causal_prior: bool = True,
      seed: int = 1,
      sampling_seed: Optional[int] = None,
      noisy_observations: bool = False,
      noisy_acquisition: bool = False,
      n_samples_per_intervention: int = 1,
      fix_likelihood_noise_var: bool = True,
      gt_samples: int = 100,
      hp_prior: bool = True,
      use_prior_mean: bool = False,
      use_true_scm: bool = False):

    # Set the seed used for GP optimization
    self.seed = seed
    # Set the seed used for sampling from the estimated SCM and for sampling
    # the values of the target functions when assuming noisy observations
    self.sampling_seed = sampling_seed

    # The true SCM is used in the target function evaluation
    self.scm_funcs = scm.scm_funcs

    # Build estimator for the function in the SCM
    self.make_scm_estimator = make_scm_estimator

    # Get the DAG
    self.graph = graph

    # Number of optimization restart for GPs
    self.n_restart = n_restart

    # Observational data
    self.observational_samples = observation_samples

    # List of all variables names
    self.all_vars = list(self.observational_samples.keys())

    # Target variable
    self.target_variable = scm.get_target_name()

    # Number of trials for BO loop
    self.number_of_trials = number_of_trials

    # Initialise optimal value according to min / max objective function
    self.task = task
    self.init_val = (
        np.inf if self.task.value == utilities.Task.MIN.value else -np.inf
    )

    # Manipulative variables
    self.manipulative_variables = [
        list(scm.variables.keys())[i]
        for i, val in enumerate(list(scm.variables.values()))
        if utilities.VariableType.MANIPULATIVE.value in val
    ]

    # Interventional domain
    self.interventional_variable_limits = intervention_domain
    assert self.manipulative_variables == list(intervention_domain.keys())

    # Intervention sets to explore
    if exploration_sets:
      assert isinstance(exploration_sets, list)
      self.exploration_sets = exploration_sets
    else:
      self.exploration_sets = list(
          initialisation_utils.powerset(self.manipulative_variables))

    # Get the interventional grids for plotting
    self.interventional_grids = (
        initialisation_utils.get_interventional_grids(
            self.exploration_sets,
            intervention_domain,
            size_intervention_grid=size_intervention_grid))

    # Whether to use observational data to build the prior
    self.causal_prior = causal_prior
    # Use the estimated causal effect as prior mean of the surrogate model
    self.use_prior_mean = use_prior_mean
    # Objective function params.
    self.bo_model = {es: None for es in self.exploration_sets}
    # Target functions for Bayesian optimisation - ground truth.
    self.target_functions = copy.deepcopy(self.bo_model)
    # Initialize a dictionary to store the noiseless values of the target
    # functions is dealing with noisy observations
    self.noiseless_target_functions = copy.deepcopy(self.bo_model)
    # Store true objective function.
    self.ground_truth = ground_truth
    # Number of points where to evaluate acquisition function.
    self.num_anchor_points = num_anchor_points
    # Hyperparameters for GPs assigned during optimisation.
    self.mean_function = copy.deepcopy(self.bo_model)
    self.variance_function = copy.deepcopy(self.bo_model)
    # Store dicts for mean and var values computed in the acquisition function.
    self.mean_dict_store = {es: {} for es in self.exploration_sets}
    self.var_dict_store = copy.deepcopy(self.mean_dict_store)

    # Initial optimal solutions
    if intervention_samples:
      # If initial interventional data is provided
      (
          initial_optimal_intervention_sets,
          initial_optimal_target_values,
          initial_optimal_intervention_levels,
          self.interventional_data_x,
          self.interventional_data_y,
      ) = initialisation_utils.initialise_interventional_objects(
          self.exploration_sets,
          intervention_samples,
          target=self.target_variable,
          task=utilities.EVAL_FN[self.task])
    else:
      # No initial interventional data is provided
      initial_optimal_intervention_sets = random.choice(self.exploration_sets)
      initial_optimal_target_values = None
      initial_optimal_intervention_levels = None
      self.interventional_data_x = copy.deepcopy(self.bo_model)
      self.interventional_data_y = copy.deepcopy(self.bo_model)

    # Dict indexed by the global exploration sets, stores the best
    self.outcome_values = (
        initialisation_utils.initialise_global_outcome_dict_new(
            initial_optimal_target_values, self.init_val
        )
    )

    # Initialize list to store the optimal outcome values
    self.optimal_outcome_values_during_trials = []

    # Set the observations to be noisy or noiseless evaluations of the target
    self.noisy_observations = noisy_observations

    # Set the acquisition function to be the noisy version
    self.noisy_acquisition = noisy_acquisition

    self.optimal_intervention_levels = (
        initialisation_utils.initialise_optimal_intervention_level_list(
            self.exploration_sets,
            initial_optimal_intervention_sets,
            initial_optimal_intervention_levels,
            number_of_trials,
        )
    )
    self.best_initial_es = initial_optimal_intervention_sets[0]
    self.best_initial_level = initial_optimal_intervention_levels
    self.best_es_over_trials_list = []
    self.best_level_over_trials_list = []

    # Whether to learn the variance of the likelihood noise or set it
    self.fix_likelihood_noise_var = fix_likelihood_noise_var

    # Set the number of samples from the interventional distribution we get
    # every time we perform an intervention. This is used when
    # noisy_observations = True. When instead noisy_observations = False there
    # is no randomness in the samples thus we only need one.
    self.n_samples_per_intervention = n_samples_per_intervention
    # Store true target function to simulate interventions for each set
    for es in self.exploration_sets:
      self.target_functions[
          es] = intervention_functions.evaluate_target_function(
              self.scm_funcs, es, self.all_vars, self.noisy_observations,
              self.sampling_seed, self.n_samples_per_intervention)

    # Store the noiseless target function so we can evaluata a posteriori if
    # the algorithm collected feasible interventions. This is used when
    # gt_samples is different from self.n_samples_per_intervention so that every
    # experiments gives a noisy evaluation of the target and the constraints
    for es in self.exploration_sets:
      self.noiseless_target_functions[
          es] = intervention_functions.evaluate_target_function(
              self.scm_funcs, es, self.all_vars, self.noisy_observations,
              self.sampling_seed, gt_samples)

    # Parameter space for optimisation
    self.intervention_exploration_domain = (
        initialisation_utils.create_intervention_exploration_domain(
            self.exploration_sets, intervention_domain))

    # Optimisation specific parameters to initialise
    self.trial_type = []  # If we observed or intervened during the trial
    self.cost_functions = cost_utils.define_costs(self.manipulative_variables,
                                                  cost_type)
    self.per_trial_cost = []

    # Acquisition function specifics
    self.y_acquired = {es: None for es in self.exploration_sets}
    self.corresponding_x = copy.deepcopy(self.y_acquired)

    # Initialise best intervention set and best intervention level over trials
    self.best_es_over_trials = self.best_initial_es
    self.best_level_over_trials = self.best_initial_level

    # Initialise the variable for storing the optimal intervention
    self.optimal_intervention = None

    # Use hyperprior on the hyperparameters of the GP model
    self.hp_prior = hp_prior

    # Debugging
    self.verbose = verbose
    self.use_true_scm = use_true_scm

  def _update_bo_model(
      self,
      data_x: Any,
      data_y: Dict[Tuple[str, ...], np.ndarray],
      mean_functions: Dict[
          Optional[Tuple[str, ...]], Callable[[np.ndarray], np.ndarray]
      ],
      variance_functions: Dict[
          Optional[Tuple[str, ...]], Callable[[np.ndarray], np.ndarray]
      ],
      bo_model: Dict[
          Optional[Tuple[str, ...]],
          Optional[gpy_model_wrappers.GPyModelWrapper],
      ],
      exploration_set: Tuple[str, ...],
      n_samples_per_intervention: int,
      alpha: float = 2,
      beta: float = 0.5,
      beta_l: float = 1.5,
      lengthscale: float = 1.0,
      variance: float = 1.0,
      fix_likelihood_noise_var: bool = True,
      interventional_limits: Optional[Dict[str, Sequence[float]]] = None,
      ard: bool = False,
      hp_prior: bool = True,
      intervention_set=None,
  ) -> None:
    """Update GP model on causal effect for exploration_set."""
    # Check data for the model exist
    assert data_y[exploration_set] is not None

    # Get the data
    x = data_x[exploration_set] if isinstance(data_x, dict) else data_x
    y = data_y[exploration_set]

    input_dim = len(intervention_set) if intervention_set else len(
        exploration_set)
    # Set the likelihood noise proportional to the number of interventional
    # samples we get after each experiment
    lik_noise_var = (1./n_samples_per_intervention)

    partial_model = functools.partial(
        models.GPRegression, X=x, Y=y, noise_var=lik_noise_var)

    # Specify mean function
    if not self.use_prior_mean:
      mf = None
    else:
      mf = core.Mapping(input_dim, 1)
      mf.f = mean_functions[exploration_set]
      mf.update_gradients = lambda a, b: None

    # Initialize the model
    if self.causal_prior:
      # Set kernel
      kernel = causal_kernel.CausalRBF(
          input_dim=input_dim,
          variance_adjustment=variance_functions[exploration_set],
          lengthscale=lengthscale,
          variance=variance,
          ard=ard)
    else:
      kernel = rbf.RBF(input_dim, lengthscale=lengthscale, variance=variance)

    model = partial_model(kernel=kernel, mean_function=mf)

    # Place a prior on kernel hyperparameters to get a MAP
    if hp_prior:
      # Numerical stability issues
      # see https://github.com/SheffieldML/GPy/issues/735
      gamma = priors.Gamma(a=alpha, b=beta)
      model.kern.variance.set_prior(gamma)

      if interventional_limits:
        # We set the hyperparameter for the GP lenghscale looking at the
        # interventional grid for each variable included in the inputs of the GP
        alpha_l = gp_utils.get_lenghscale_hp(exploration_set,
                                             interventional_limits)
        gamma = priors.Gamma(a=alpha_l, b=beta_l)

      model.kern.lengthscale.set_prior(gamma)

    if fix_likelihood_noise_var:
      # Fix likelihood variance to a very small value
      model.likelihood.variance.fix(1e-5)

    if self.verbose:
      print("Optimizing the model for:", exploration_set)
      print("Model BEFORE optimizing:", model)

    # Prevent randomization from affecting the optimization of the GPs
    old_seed = np.random.get_state()
    np.random.seed(self.seed)
    # With num_restarts we repeat the optimization multiple times and pick the
    # hyperparameters giving the highest likelihood
    model.optimize_restarts(num_restarts=self.n_restart)
    np.random.set_state(old_seed)
    if self.verbose:
      print("Model AFTER optimizing:", model)

    # Assign the model to the exploration set
    bo_model[exploration_set] = gpy_model_wrappers.GPyModelWrapper(model)
    # Avoid numerical issues due to the optization of the kernel hyperpars
    self._safe_optimization(bo_model[exploration_set])

  def _select_next_point(self, *args) -> Tuple[Tuple[str, ...], np.ndarray]:
    raise NotImplementedError(
        "_select_next_point method has not been implemented for"
        "this class")

  def _check_new_point(self, best_es: Tuple[str, ...]) -> None:
    """Check that new intervention point is in the intervention domain."""
    assert best_es is not None, (best_es, self.y_acquired)
    assert best_es in self.exploration_sets

    assert self.intervention_exploration_domain[best_es].check_points_in_domain(
        self.corresponding_x[best_es])[0], (
            best_es,
            self.y_acquired,
            self.corresponding_x,
        )

  def _safe_optimization(self,
                         bo_model: gpy_model_wrappers.GPyModelWrapper,
                         bound_var=1e-02,
                         bound_len=20.0) -> None:
    """Avoid numerical instability in the optimization of the GP hyperpars."""
    if bo_model.model.kern.variance[0] < bound_var:  # pytype: disable=attribute-error
      bo_model.model.kern.variance[0] = 1.0  # pytype: disable=attribute-error

    if bo_model.model.kern.lengthscale[0] > bound_len:  # pytype: disable=attribute-error
      bo_model.model.kern.lengthscale[0] = 1.0  # pytype: disable=attribute-error

  def _get_updated_interventional_data(self, x_new: np.ndarray, y_new: float,
                                       best_es: Tuple[str, ...]) -> None:
    """Updates interventional data."""
    data_x, data_y = utilities.check_reshape_add_data(
        self.interventional_data_x, self.interventional_data_y, x_new, y_new,
        best_es)
    self.interventional_data_x[best_es] = data_x
    self.interventional_data_y[best_es] = data_y

  def _update_sufficient_statistics(
      self, target: str, fitted_scm: Callable[[], Any]) -> None:
    """Update mean and variance functions of the causal prior (GP).

    Args:
      target : The full node name of the target variable.
      fitted_scm : Fitted SCM.
    """
    for es in self.exploration_sets:
      (self.mean_function[es],
       self.variance_function[es]) = gp_utils.update_sufficient_statistics_hat(
           graph=self.graph,
           y=target,
           x=es,
           fitted_scm=fitted_scm,
           true_scm_funcs=self.scm_funcs,
           seed=self.sampling_seed,
           mean_dict_store=self.mean_dict_store,
           var_dict_store=self.var_dict_store,
           n_samples=self.n_samples_per_intervention,
           use_true_scm=self.use_true_scm)

  def _per_trial_computations(self, it: int, target: str) -> None:
    """Performs computations for each trial iteration for specific target."""
    logging.info(">>>")
    logging.info("Iteration: %s", it)
    logging.info("<<<")

    if self.verbose:
      print(">>> Target model BEFORE optimization")
      plotting_utils.plot_models(self.bo_model, self.exploration_sets,
                                 self.ground_truth, self.interventional_grids,
                                 self.interventional_data_x,
                                 self.interventional_data_y)

    # Presently find the optimal value of Y_t
    current_best_global_target = utilities.EVAL_FN[self.task](
        self.outcome_values)
    if self.verbose:
      logging.info("Current_best_global_target: %s", current_best_global_target)

    # Indicate that in this trial we are explicitly intervening in the system
    self.trial_type.append(utilities.Trial.INTERVENTION)

    best_es, new_interventional_data_x = self._select_next_point(
        current_best_global_target, it)

    # Get the correspoding outcome values for best_es
    y_new = self.target_functions[best_es](
        target, np.squeeze(new_interventional_data_x))

    # Store intervened set
    self.best_es_over_trials_list.append(best_es)
    self.best_level_over_trials_list.append(new_interventional_data_x)

    if self.verbose:
      logging.info("Selected set: %s", best_es)
      logging.info("Intervention value: %s", new_interventional_data_x)
      logging.info("Outcome: %s", y_new)

    # Update interventional data
    self._get_updated_interventional_data(new_interventional_data_x, y_new,
                                          best_es)

    # Evaluate cost of intervention
    self.per_trial_cost.append(
        cost_utils.total_intervention_cost(
            best_es,
            self.cost_functions,
            self.interventional_data_x[best_es],
        ))

    # Store optimal outcome values
    self.outcome_values.append(y_new)
    self.optimal_outcome_values_during_trials.append(
        utilities.EVAL_FN[self.task](y_new, current_best_global_target))

    new_best_solution = utilities.ARG_EVAL_FN[self.task](
        (y_new, current_best_global_target))
    self.best_es_over_trials = (best_es,
                                self.best_es_over_trials)[new_best_solution]
    self.best_level_over_trials = (
        new_interventional_data_x,
        self.best_level_over_trials)[new_best_solution]
    # Store the intervention
    if len(new_interventional_data_x.shape) != 2:
      self.optimal_intervention_levels[best_es][
          it] = utilities.make_column_shape_2d(new_interventional_data_x)
    else:
      self.optimal_intervention_levels[best_es][it] = new_interventional_data_x

    #  Update the BO model for best_es
    self._update_bo_model(
        data_x=self.interventional_data_x,
        data_y=self.interventional_data_y,
        mean_functions=self.mean_function,
        variance_functions=self.variance_function,
        bo_model=self.bo_model,
        exploration_set=best_es,
        hp_prior=self.hp_prior,
        fix_likelihood_noise_var=self.fix_likelihood_noise_var,
        interventional_limits=self.interventional_variable_limits,
        n_samples_per_intervention=self.n_samples_per_intervention)

    if self.verbose:
      print(">>> Target model AFTER optimization")
      plotting_utils.plot_models(self.bo_model, self.exploration_sets,
                                 self.ground_truth, self.interventional_grids,
                                 self.interventional_data_x,
                                 self.interventional_data_y)
