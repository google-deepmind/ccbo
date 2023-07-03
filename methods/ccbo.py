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

"""Define method for constrained causal Bayesian Optimization."""
from __future__ import annotations

import collections
import copy
import functools
import itertools
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from absl import logging
from emukit.model_wrappers import gpy_model_wrappers
from GPy import core
from GPy.core.parameterization import priors
from GPy.kern.src import rbf
from GPy.util import multioutput
from networkx.classes import multidigraph
import numpy as np

from ccbo.acquisitions import evaluate_acquisitions
from ccbo.kernels import causal_coregionalize_kernel
from ccbo.methods import cbo
from ccbo.utils import constraints_functions
from ccbo.utils import cost_utils
from ccbo.utils import gp_utils
from ccbo.utils import intervention_functions
from ccbo.utils import plotting_utils
from ccbo.utils import sampling_utils
from ccbo.utils import scm_utils
from ccbo.utils import utilities


class CCBO(cbo.CBO):
  """Constrained Causal Bayesian Optimisation class."""

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
      constraints: Optional[Dict[str, Any]] = None,
      ground_truth_constraints: Optional[Dict[str, Dict[str,
                                                        List[float]]]] = None,
      multi_task_model: bool = False,
      sampling_seed: Optional[int] = None,
      n_kernel_samples: int = 1,
      noisy_observations: bool = False,
      noisy_acquisition: bool = False,
      fix_likelihood_noise_var: bool = True,
      n_samples_per_intervention: int = 1,
      add_rbf_kernel: bool = False,
      update_scm: bool = False,
      use_prior_mean: bool = False,
      size_intervention_grid: int = 100,
      use_true_scm: bool = False):
    """Constrained Causal Bayesian Optimisation class."""

    super().__init__(
        graph=graph,
        scm=scm,
        make_scm_estimator=make_scm_estimator,
        observation_samples=observation_samples,
        intervention_domain=intervention_domain,
        intervention_samples=intervention_samples,
        exploration_sets=exploration_sets,
        number_of_trials=number_of_trials,
        ground_truth=ground_truth,
        task=task,
        n_restart=n_restart,
        cost_type=cost_type,
        hp_prior=hp_prior,
        num_anchor_points=num_anchor_points,
        seed=seed,
        sample_anchor_points=sample_anchor_points,
        seed_anchor_points=seed_anchor_points,
        causal_prior=causal_prior,
        verbose=verbose,
        sampling_seed=sampling_seed,
        noisy_observations=noisy_observations,
        noisy_acquisition=noisy_acquisition,
        n_samples_per_intervention=n_samples_per_intervention,
        fix_likelihood_noise_var=fix_likelihood_noise_var,
        size_intervention_grid=size_intervention_grid,
        use_true_scm=use_true_scm,
        use_prior_mean=use_prior_mean,
    )

    # Initialization specific to the constrained CBO algorithm
    self.constraints = constraints
    # Verify that constraints exist
    assert self.constraints is not None, "cCBO requires a list of constraints"

    # Initialise the list and dict for the models on the constraints
    (self.constraints_dict, self.initial_feasibility, self.bo_model_constraints,
     self.feasibility, self.interventional_data_y_constraints
    ) = constraints_functions.get_constraints_dicts(self.exploration_sets,
                                                    self.constraints,
                                                    self.graph,
                                                    self.target_variable,
                                                    self.observational_samples)
    self.feasibility_noiseless = {es: [] for es in self.exploration_sets}

    # Initialize mean and var functions for GPs on constraints functions
    self.mean_function_constraints = copy.deepcopy(self.bo_model_constraints)
    self.var_function_constraints = copy.deepcopy(self.bo_model_constraints)

    # Initialize best feasible exploration sets over trials
    (self.best_es_over_trials, self.best_level_over_trials
    ) = constraints_functions.best_feasible_initial_es(
        self.initial_feasibility, self.best_initial_es, self.best_initial_level,
        self.interventional_data_y, self.interventional_data_x,
        utilities.EVAL_FN[self.task])

    # Store values for prior mean and variance
    self.mean_constraints_dict_store = {
        es: {p: {} for p in self.constraints_dict[es]["vars"]
            } for es in self.exploration_sets
    }
    self.var_constraints_dict_store = copy.deepcopy(
        self.mean_constraints_dict_store)

    # Initialize object to store the observed constraints values
    # The interventional input values are the same for target and constraints
    constraints_functions.initialise_constraints_interventional_objects(
        self.exploration_sets, intervention_samples,
        self.interventional_data_y_constraints, self.bo_model_constraints,
        self.initial_feasibility, constraints, self.feasibility)

    # Get interventional domains for both manipulative and not manipulative vars
    self.interventional_constraint_variable_limits = {  # pylint: disable=g-complex-comprehension
        key: scm.variables[key][1]
        for key in self.all_vars
        if scm.variables[key][0] in [
            utilities.VariableType.NONMANIPULATIVE.value,
            utilities.VariableType.MANIPULATIVE.value
        ]
    }

    # Store acquisition function values
    self.improvement_dict = {es: [] for es in self.exploration_sets}
    self.prob_feasibility_dict = {es: [] for es in self.exploration_sets}
    self.ground_truth_constraints = ground_truth_constraints

    # Whether to use a multi-task model for the target and the constraints
    self.multi_task_model = multi_task_model

    # Number of samples for estimating the kernel of a multi-task GP
    self.n_kernel_samples = n_kernel_samples

    # Add RBF kernel to the coregionalized kernel
    self.add_rbf_kernel = add_rbf_kernel

    # Update the SCM fitting as we collect interventional data
    self.update_scm = update_scm

    self.dict_mean_product = copy.deepcopy(
        self.mean_dict_store)
    self.dict_constraints_mean_product = copy.deepcopy(
        self.mean_constraints_dict_store)
    self.dict_product_mean = copy.deepcopy(
        self.mean_dict_store)
    self.dict_constraints_product_mean = copy.deepcopy(
        self.mean_constraints_dict_store)

  def run(self) -> None:
    """Run the BO loop."""
    if self.verbose:
      assert self.ground_truth is not None, "Provide ground truth"
      assert self.ground_truth_constraints is not None, "Provide ground truth"

    # Get current target and best_es
    target = self.target_variable
    best_es = self.best_initial_es

    for it in range(self.number_of_trials):
      logging.info("Trial: %s", it)

      if it == 0:  # First trial
        self.best_es_over_trials_list.append([
            best_es,
            int(np.all(list(self.initial_feasibility[best_es].values())))
        ])
        self.best_level_over_trials_list.append(self.best_initial_level)
        fitted_scm = self.observation_trial(best_es, target, it)

        # Update model on the constraints
        self._update_sufficient_statistics_constraints(fitted_scm=fitted_scm)  # pytype: disable=wrong-arg-types  # dynamic-method-lookup
        # When observe thus append previous feasibility values for logs
        for es in self.exploration_sets:
          self.feasibility[es].append(self.feasibility[es][-1])

        if self.verbose:
          logging.info("Current feasibility: %s", self.feasibility)
          logging.info("Current optimal y values: %s",
                       self.optimal_outcome_values_during_trials)
      else:
        if it > 1 and self.update_scm:
          # Refit Gaussian processes to functions in the SCM using observational
          # and interventional data. These are then used to update the
          # multi-task kernel when using a causal prior.
          logging.warning("The functions in the SCM are refitted at each"
                          " iteration and the causal kernel is thus updated. "
                          " This might significantly slow down the algorithm!")

          # Refit the functions
          self.fitted_scm_fncs = scm_utils.fit_scm_fncs(
              self.graph, self.observational_samples, self.scm_funcs,
              self.n_restart)
          # Update the simulator
          fitted_scm = self.make_scm_estimator(self.graph, self.scm_fncs,
                                               self.scm)

          # Reset the dict to store the sampled mean causal effects as we don't
          # want to reuse values obtained earlier with different SCM functions
          self.mean_dict_store = {es: {} for es in self.exploration_sets}
          self.mean_constraints_dict_store = {
              es: {p: {} for p in self.constraints_dict[es]["vars"]
                  } for es in self.exploration_sets
          }

        self._intervention_trial(target, it, self.multi_task_model, fitted_scm)

    # Store optimal intervention
    self.optimal_intervention = {
        "set": self.best_es_over_trials,
        "level": self.best_level_over_trials,
        "outcome": self.optimal_outcome_values_during_trials[-1],
    }

  def _intervention_trial(
      self,
      target: str,
      it: int,
      multi_task_model: bool = False,
      fitted_scm: Optional[Callable[[], Any]] = None,
  ) -> None:
    """Run one intervention trial of the BO loop."""
    # Update models if we have only observed so far
    if self.trial_type[-1] == utilities.Trial.OBSERVATION:
      if multi_task_model:
        assert fitted_scm
        self._update_all_models(fitted_scm=fitted_scm)
      else:
        self._update_all_surrogate_models()
        self._update_all_surrogate_models_constraints()

    # Run the actual per trial computation
    self._per_trial_computations(it, target, fitted_scm)

  def _update_all_models(self, fitted_scm: Callable[[], Any]) -> None:
    """Update surrogate models (GPs) on the target and the constraints."""
    for es in self.exploration_sets:
      # If there is data for an exploration set we want to update the models
      if (self.interventional_data_x[es] is not None and
          self.interventional_data_y[es] is not None):
        if self.constraints_dict[es]["num"] == 0:
          # If there are no constraints we just update the surrogate model
          update_model_cls = functools.partial(
              self._update_bo_model,
              data_x=self.interventional_data_x,
              data_y=self.interventional_data_y,
              mean_functions=self.mean_function,
              variance_functions=self.variance_function,
              bo_model=self.bo_model)
        else:
          # Update the multi-task model
          update_model_cls = functools.partial(
              self._update_multi_task_model,
              causal_prior=self.causal_prior,
              graph=self.graph,
              mean_function=self.mean_function,
              mean_function_constraints=self.mean_function_constraints,
              fitted_scm=fitted_scm)

        update_model_cls(
            exploration_set=es,
            hp_prior=self.hp_prior,
            fix_likelihood_noise_var=self.fix_likelihood_noise_var,
            interventional_limits=self
            .interventional_constraint_variable_limits,
            n_samples_per_intervention=self.n_samples_per_intervention)

  def _update_multi_task_model(
      self,
      exploration_set: Set[str],
      n_samples_per_intervention: int,
      causal_prior: bool = False,
      graph: Optional[multidigraph.MultiDiGraph] = None,
      mean_function: Optional[Dict[Set[str], Any]] = None,
      mean_function_constraints: Optional[Dict[Set[str], Any]] = None,
      fitted_scm: Optional[Callable[[], Any]] = None,
      hp_prior: bool = True,
      lengthscale: float = 1.,
      variance: float = 1.,
      alpha: float = 2,
      beta: float = 0.5,
      beta_l: float = 1.5,
      n_optimization_restarts: int = 1,
      verbose: bool = False,
      fix_likelihood_noise_var: bool = True,
      interventional_limits: Optional[Dict[str, Sequence[float]]] = None):
    """Update surrogate models (GPs) with a multi-task structure."""

    input_dim = len(exploration_set)

    x = self.interventional_data_x[exploration_set]
    y = self.interventional_data_y[exploration_set]
    y_constraints = list(
        self.interventional_data_y_constraints[exploration_set].values())

    # Define multi-task outputs
    y_multi_task = [y] + y_constraints
    # The number of outputs is given by the constraints plus the target
    num_outputs = self.constraints_dict[exploration_set]["num"] + 1
    assert len(y_multi_task) == num_outputs

    # Define multi-task inputs
    x_multi_task = [x] * num_outputs

    # Define RBF kernel and put priors on hyperparameters
    kernel = rbf.RBF(input_dim, lengthscale=lengthscale, variance=variance)

    # If hp_prior is True, we place a prior on kernel hyperparameters of each
    # function to get a MAP. This is for numerical stability issues.
    if hp_prior:
      gamma = priors.Gamma(a=alpha, b=beta)
      kernel.variance.set_prior(gamma)

      if interventional_limits:
        all_vars_multitask_model = list(
            exploration_set
        ) + self.constraints_dict[exploration_set]["vars"]
        alpha_l = gp_utils.get_lenghscale_hp(all_vars_multitask_model,
                                             interventional_limits)
        gamma = priors.Gamma(a=alpha_l, b=beta_l)

      kernel.lengthscale.set_prior(gamma)

    # Merge all estimated mean functions
    total_mean_list = [mean_function[exploration_set]] + [
        *mean_function_constraints[exploration_set].values()
    ]
    if not self.use_prior_mean:
      mean_function = None
    else:
      # Define prior mean function
      mean_function = core.Mapping(input_dim + 1, 1)
      mean_function.f = gp_utils.mean_function_multitask_model(total_mean_list)
      mean_function.update_gradients = lambda a, b: None

    # Define a kernel for the multi-task model
    if not causal_prior:
      # Define an ICM type of kernel
      multitask_kernel = multioutput.ICM(
          input_dim=input_dim, num_outputs=num_outputs, kernel=kernel)
    else:
      # Use a kernel giving a correlation structure among outputs that
      # reflects the DAG structure. The kernel is numerically approximated.
      # active_dims is used to indicate which dimension of the inputs should
      # be used in the kernel computation. Here we want to use also the task
      # type that is the function index. We thus adjust the input dim so that
      # the kernel internally considers it.

      target_list = [self.target_variable
                    ] + self.constraints_dict[exploration_set]["vars"]

      # Notice that the input dimensionality here is given by the original
      # input dimensionality plus the additional dimension given by
      # the task index (or function index)
      multitask_kernel = causal_coregionalize_kernel.CausalCoregionalize(
          input_dim=input_dim + 1,
          target_list=target_list,
          graph=graph,
          target_variable=self.target_variable,
          exploration_set=exploration_set,
          fitted_scm=fitted_scm,
          true_scm_funcs=self.scm_funcs,
          dict_mean_product=self.dict_mean_product,
          dict_constraints_mean_product=self.dict_constraints_mean_product,
          dict_product_mean=self.dict_product_mean,
          dict_constraints_product_mean=self.dict_constraints_product_mean,
          seed=self.sampling_seed,
          n_samples=self.n_kernel_samples,
          use_true_scm=self.use_true_scm)

      if self.add_rbf_kernel:
        # Add an RBF kernel to the multi-task kernel to increase the model
        # flexibility
        multitask_kernel += kernel

    # Initialize the multi-task model with the defined kernel and mean function
    model = gp_utils.GPCausalCoregionalizedRegression(
        x_multi_task,
        y_multi_task,
        kernel=multitask_kernel,
        mean_function=mean_function)

    if fix_likelihood_noise_var:
      # Fix all likelihood variances to zero
      for param in model.likelihood.parameters:
        param.fix(1e-5)
    else:
      # Initialize the value of the liklihood variance considering the number
      # of interventional samples we get from each experiment
      for param in model.likelihood.parameters:
        lik_noise_var = (1./n_samples_per_intervention)
        param.variance = lik_noise_var

    # Assign to all models for exploration_set the same multi-task GP
    # This will be used to compute the acqusition function
    multi_task_model = gpy_model_wrappers.GPyMultiOutputWrapper(
        gpy_model=model,
        n_outputs=num_outputs,
        n_optimization_restarts=n_optimization_restarts,
        verbose_optimization=verbose)

    # Optimize multi-task model but prevent randomization from affecting
    # the optimization of the GP hyperparameters
    if self.verbose:
      print("Optimizing the multi task model for:", exploration_set)
      print("Model BEFORE optimizing:", model)

    old_seed = np.random.get_state()
    np.random.seed(self.seed)
    multi_task_model.optimize()
    np.random.set_state(old_seed)

    if self.verbose:
      print("Model AFTER optimizing:", model)

    self.bo_model[exploration_set] = multi_task_model
    for var in self.constraints_dict[exploration_set]["vars"]:
      self.bo_model_constraints[exploration_set][var] = multi_task_model

  def _update_all_surrogate_models_constraints(self) -> None:
    """Update all surrogate models (GPs) on the constraints."""
    for es in self.exploration_sets:
      if (self.interventional_data_x[es] is not None and
          self.interventional_data_y[es] is not None):
        self._update_bo_model_constraints(es)

  def _update_bo_model_constraints(self, es: Tuple[str, ...]) -> None:
    """Update surrogate model (GPs) on the constraints for es."""
    constraints_targets_es = list(self.bo_model_constraints[es].keys())
    assert set(constraints_targets_es) == set(self.constraints_dict[es]["vars"])
    for p in constraints_targets_es:
      self._update_bo_model(
          exploration_set=p,
          intervention_set=es,
          data_x=self.interventional_data_x[es],
          data_y=self.interventional_data_y_constraints[es],
          mean_functions=self.mean_function_constraints[es],
          variance_functions=self.var_function_constraints[es],
          bo_model=self.bo_model_constraints[es],
          hp_prior=self.hp_prior,
          fix_likelihood_noise_var=self.fix_likelihood_noise_var,
          interventional_limits=self.interventional_constraint_variable_limits,
          n_samples_per_intervention=self.n_samples_per_intervention)

  def _evaluate_acquisition_functions(self, current_best_global_target,
                                      it: int) -> None:
    """Evaluate the acquisition function given the surrogate models."""
    for es in self.exploration_sets:
      if (self.interventional_data_x[es] is not None and
          self.interventional_data_y[es] is not None):
        # If DI, the model exists
        bo_model = self.bo_model[es]
        bo_model_constraints = self.bo_model_constraints[es]
      else:
        # If no DI, the model does not exist yet.
        # We initialise the standard mean and variance function
        # and use a single-task model.
        bo_model = None
        bo_model_constraints = None
        self.mean_function[es] = utilities.standard_mean_function
        self.variance_function[es] = utilities.zero_variance_adjustment

        # There are constraints for es
        if self.constraints_dict[es]:
          for j in range(self.constraints_dict[es]["num"]):
            self.mean_function_constraints[es][
                j] = utilities.standard_mean_function
            self.var_function_constraints[es][
                j] = utilities.zero_variance_adjustment

      if self.seed_anchor_points is None:
        seed_to_pass = None
      else:
        seed_to_pass = int(self.seed_anchor_points * it)

      (self.y_acquired[es], self.corresponding_x[es], improvement,
       pf) = evaluate_acquisitions.evaluate_constrained_acquisition_function(
           self.intervention_exploration_domain[es],
           bo_model,
           self.mean_function[es],
           self.variance_function[es],
           current_best_global_target,
           es,
           self.cost_functions,
           self.task,
           self.target_variable,
           bo_model_constraints,
           self.mean_function_constraints[es],
           self.var_function_constraints[es],
           self.constraints,
           self.constraints_dict,
           verbose=self.verbose,
           num_anchor_points=self.num_anchor_points,
           sample_anchor_points=self.sample_anchor_points,
           seed_anchor_points=seed_to_pass,
           multi_task_model=self.multi_task_model,
           noisy_acquisition=self.noisy_acquisition)
      self.improvement_dict[es].append(improvement)
      self.prob_feasibility_dict[es].append(pf)

  def _per_trial_computations(self, it: int, target: str,
                              fitted_scm: Callable[[], Any]) -> None:
    """Performs computations for each trial iteration for specific target."""
    logging.info(">>>")
    logging.info("Iteration: %s", it)
    logging.info("<<<")
    if self.verbose:
      print(">>> Target model BEFORE optimization")
      plotting_utils.plot_models(self.bo_model, self.exploration_sets,
                                 self.ground_truth, self.interventional_grids,
                                 self.interventional_data_x,
                                 self.interventional_data_y,
                                 self.multi_task_model)
      print(">>> Constraints models BEFORE optimization")
      plotting_utils.plot_models(
          self.bo_model_constraints, self.exploration_sets,
          self.ground_truth_constraints, self.interventional_grids,
          self.interventional_data_x, self.interventional_data_y_constraints,
          self.multi_task_model)

    # Indicate that in this trial we are explicitly intervening in the system
    self.trial_type.append(utilities.Trial.INTERVENTION)

    # Get current best across intervention sets
    current_best_global_target = self._get_current_feasible_global_target()

    # Compute acquisition function given the updated BO models for DI.
    # Notice that we use current_global and the costs to compute it.
    self._evaluate_acquisition_functions(current_best_global_target, it)

    # Best exploration set based on acquired target-values
    best_es = max(self.y_acquired, key=self.y_acquired.get)
    new_interventional_data_x = self.corresponding_x[best_es]
    self._check_new_point(best_es)

    # Get the corresponding outcome values for this intervention set
    y_new = self.target_functions[best_es](
        target, np.squeeze(new_interventional_data_x))

    if self.verbose:
      logging.info("Current best global target: %s", current_best_global_target)
      logging.info("All y values found: %s", self.y_acquired)
      logging.info("Best es found: %s", best_es)
      logging.info("Best x found: %s", new_interventional_data_x)
      logging.info("Best y found: %s", y_new)

    # Get the value for the constraints values (both noisy and noiseless)
    # related to the intervened variable
    y_new_c = {}
    feasibility_list = []
    feasibility_list_noiseless = []
    if self.constraints_dict[best_es]["num"] == 0:
      tmp = 1
      feasibility_list.append(tmp)
    else:
      for j in range(self.constraints_dict[best_es]["num"]):
        c_target = self.constraints_dict[best_es]["vars"][j]
        y_new_var = self.target_functions[best_es](
            target=c_target,
            intervention_levels=np.squeeze(new_interventional_data_x))
        y_new_var_noiseless = self.noiseless_target_functions[best_es](
            target=c_target,
            intervention_levels=np.squeeze(new_interventional_data_x))

        # To evaluate the feasibility
        tmp = constraints_functions.EVAL_CONSTRAINT_OP[
            self.constraints[c_target][0]](y_new_var,
                                           self.constraints[c_target][1])
        tmp_noiseless = constraints_functions.EVAL_CONSTRAINT_OP[
            self.constraints[c_target][0]](y_new_var_noiseless,
                                           self.constraints[c_target][1])

        y_new_c[c_target] = y_new_var
        feasibility_list.append(tmp)
        feasibility_list_noiseless.append(tmp_noiseless)

    if self.verbose:
      logging.info("Selected set: %s", best_es)
      logging.info("Intervention value: %s", new_interventional_data_x)
      logging.info("Outcome: %s", y_new)
      logging.info("Feasible: %s", bool(tmp))

    # Append new interventional observations to refit the SCM at the next trial
    if self.update_scm:
      # Generate the full samples we used to compute the output and the
      # constraints from the true intervened model. These are then appended to
      # the observational data which is then used to refit the functions
      interventions = intervention_functions.assign_initial_intervention_level(
          exploration_set=best_es,
          intervention_level=new_interventional_data_x,
          variables=list(self.observational_samples.keys()))
      # Sample from the true interventional distribution
      out = sampling_utils.sample_scm(
          scm_funcs=self.scm_funcs,
          graph=None,
          interventions=interventions,
          n_samples=self.n_samples_per_intervention,
          compute_moments=True,
          moment=0,
          seed=self.sampling_seed)

      # Append new observations
      self.observational_samples = {
          key: np.vstack((self.observational_samples[key], out[key]))
          for key in out
      }

    # Update interventional data for the target
    self._get_updated_interventional_data(new_interventional_data_x, y_new,
                                          best_es)
    # Update interventional data for the constraints
    self._get_updated_interventional_data_constraints(y_new_c, best_es)

    # Evaluate cost of intervention
    self.per_trial_cost.append(
        cost_utils.total_intervention_cost(
            best_es,
            self.cost_functions,
            self.interventional_data_x[best_es],
        ))

    # Store the optimal feasible outcome corresponding to intervention levels
    self.outcome_values.append(y_new)
    self.feasibility[best_es].append(int(all(feasibility_list)))
    self.feasibility_noiseless[best_es].append(
        int(all(feasibility_list_noiseless)))

    # If the new point is feasible check if it is optimal otherwise discard it
    if all(feasibility_list):
      # If the point collected is feasible we need to compare with the previous
      best_value = utilities.EVAL_FN[self.task](y_new,
                                                current_best_global_target)

      self.optimal_outcome_values_during_trials.append(best_value)
      # Before moving to the next iteration we store the currently found
      # best intervention set which corresponds to the one given the min
      # between y_new and current_best_global_target
      new_best_solution = utilities.ARG_EVAL_FN[self.task](
          (y_new, current_best_global_target))
      self.best_es_over_trials = (best_es,
                                  self.best_es_over_trials)[new_best_solution]
      self.best_level_over_trials = (
          new_interventional_data_x,
          self.best_level_over_trials)[new_best_solution]
    else:
      # If the current point is not feasible store the previous value
      # in this case the best_es_over_trials does not need to change
      self.optimal_outcome_values_during_trials.append(
          current_best_global_target)

    # Store intervened set and whether it is feasible
    self.best_es_over_trials_list.append([best_es, int(all(feasibility_list))])
    self.best_level_over_trials_list.append(new_interventional_data_x)

    # Store the intervention
    if len(new_interventional_data_x.shape) != 2:
      self.optimal_intervention_levels[best_es][
          it] = utilities.make_column_shape_2d(new_interventional_data_x)
    else:
      self.optimal_intervention_levels[best_es][it] = new_interventional_data_x

    # Update the best_es BO model and the related constraints
    if self.multi_task_model and self.constraints_dict[best_es]["num"] > 0:
      update_model_cls = functools.partial(
          self._update_multi_task_model,
          causal_prior=self.causal_prior,
          graph=self.graph,
          mean_function=self.mean_function,
          mean_function_constraints=self.mean_function_constraints,
          fitted_scm=fitted_scm)
    else:
      update_model_cls = functools.partial(
          self._update_bo_model,
          data_x=self.interventional_data_x,
          data_y=self.interventional_data_y,
          mean_functions=self.mean_function,
          variance_functions=self.variance_function,
          bo_model=self.bo_model)
      self._update_bo_model_constraints(best_es)

    update_model_cls(
        exploration_set=best_es,
        fix_likelihood_noise_var=self.fix_likelihood_noise_var,
        interventional_limits=self.interventional_constraint_variable_limits,
        n_samples_per_intervention=self.n_samples_per_intervention,
        hp_prior=self.hp_prior)

    if self.verbose:
      print(">>> Target model AFTER optimization")
      plotting_utils.plot_models(self.bo_model, self.exploration_sets,
                                 self.ground_truth, self.interventional_grids,
                                 self.interventional_data_x,
                                 self.interventional_data_y,
                                 self.multi_task_model)
      print(">>> Constraint model AFTER optimization")
      plotting_utils.plot_models(
          self.bo_model_constraints, self.exploration_sets,
          self.ground_truth_constraints, self.interventional_grids,
          self.interventional_data_x, self.interventional_data_y_constraints,
          self.multi_task_model)

      logging.info("Current feasibility: %s", self.feasibility)
      logging.info("Current optimal y values: %s",
                   self.optimal_outcome_values_during_trials)

  def _update_sufficient_statistics_constraints(
      self, fitted_scm: Callable[[], Any]) -> None:
    for es in self.exploration_sets:
      for p in self.mean_function_constraints[es].keys():
        (
            self.mean_function_constraints[es][p],
            self.var_function_constraints[es][p],
        ) = gp_utils.update_sufficient_statistics_hat(
            graph=self.graph,
            x=es,
            y=p,
            fitted_scm=fitted_scm,
            true_scm_funcs=self.scm_funcs,
            seed=self.sampling_seed,
            mean_dict_store=self.mean_constraints_dict_store,
            var_dict_store=self.var_constraints_dict_store,
            n_samples=self.n_samples_per_intervention,
            use_true_scm=self.use_true_scm)

  def _get_current_feasible_global_target(self) -> Any:
    """Get the current feasible optimal target."""
    out = []
    feasible_interventions = {}
    is_feasible = {}
    for es in self.exploration_sets:
      feasible_interventions[es] = list(
          itertools.compress(self.interventional_data_y[es],
                             list(map(bool, self.feasibility[es][1:]))))
      if not feasible_interventions[es]:
        # There are no feasible interventions for es therefore we store the
        # initial value we have for that set and denote it as not feasible
        feasible_interventions[es] = self.interventional_data_y[es][0]
        is_feasible[es] = False
      else:
        # If the list feasible_interventions[es] is not empty we have at least
        # one feasible intervention
        is_feasible[es] = True

      # Take the optimal value
      res = utilities.EVAL_FN[self.task](feasible_interventions[es])
      out.append(res)

    # Check if there is at least one feasible value. If yes only focus on those.
    feasible_results = []
    for key, val in is_feasible.items():
      if val:
        # There is at least one feasible value thus I only want to take the
        # optimum among these values
        feasible_results.append(
            utilities.EVAL_FN[self.task](feasible_interventions[key])
        )

    if feasible_results:
      out = feasible_results

    return utilities.EVAL_FN[self.task](out)[0]

  def _get_updated_interventional_data_constraints(
      self, y_new_c: Dict[str, float], best_es: Tuple[str, ...]
  ) -> None:
    """Update interventional data for the constraints."""
    for var in list(y_new_c.keys()):
      if self.interventional_data_y_constraints[best_es][var] is not None:
        # Append the new value
        self.interventional_data_y_constraints[best_es][var] = np.append(
            self.interventional_data_y_constraints[best_es][var],
            np.array(y_new_c[var])[np.newaxis, np.newaxis],
        )[:, np.newaxis]
      else:
        # Assign the first value
        self.interventional_data_y_constraints[best_es][var] = np.array(
            y_new_c[var]
        )[np.newaxis, np.newaxis]
