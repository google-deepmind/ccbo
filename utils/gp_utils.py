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

"""Gaussian process utils."""
from __future__ import annotations

import collections
import functools
import itertools
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from GPy import core
from GPy import kern
from GPy import likelihoods
from GPy.models import gp_regression
from GPy.util import multioutput
from networkx.classes import multidigraph
import numpy as np

from ccbo.utils import initialisation_utils
from ccbo.utils import sampling_utils
from ccbo.utils import utilities


class GPCausalCoregionalizedRegression(core.GP):
  """Gaussian Process model for causal multioutput regression.

  With respect to the build in function in GPy, this function gives the user the
  possibility to specify a mean function that can vary across tasks, that is
  across causal effects.
  """

  def __init__(self,
               x_list: List[float],
               y_list: List[float],
               kernel: kern.Kern,
               likelihoods_list: Optional[List[likelihoods.Likelihood]] = None,
               mean_function: Optional[Mapping[np.ndarray, np.ndarray]] = None,
               name: str = 'GP_CausalCR'):
    # Define inputs and outputs
    x, y, self.output_index = multioutput.build_XY(x_list, y_list)

    # Define likelihood for each task
    likelihood = multioutput.build_likelihood(y_list, self.output_index,
                                              likelihoods_list)

    # Initialize model
    super().__init__(
        x,
        y,
        kernel,
        likelihood,
        mean_function=mean_function,
        Y_metadata={'output_index': self.output_index})


def mean_function_multitask_model(
    total_mean_list: List[Callable[[Any], np.ndarray]]
) -> Callable[[np.ndarray], np.ndarray]:
  """Computes the mean functions for a multi-task model ."""
  def mean_function_multitask_internal(values: np.ndarray) -> np.ndarray:
    # The argument values gives the input values at which to compute the mean
    # functions. Here the first dimension gives the x value whereas the second
    # gives the function index and therefore which mapping to use out of the
    # total_mean_list
    out = []
    for i in range(values.shape[0]):
      single_value = values[i, :]
      # Get the index of the function. This is the last dimension of the inputs.
      index = int(single_value[-1])
      # Compute the mean function corresponding to the index at the input value
      # which is given in the D-1 rows of single_value
      res = total_mean_list[index]([single_value[:-1]])
      out.append(res)
    return np.vstack(out)
  return mean_function_multitask_internal


def get_causal_effect_by_sampling(
    values: np.ndarray,
    y: str,
    x: Tuple[str, ...],
    graph: multidigraph.MultiDiGraph,
    fitted_scm: Callable[[], Any],
    true_scm_funcs: collections.OrderedDict[str, Any],
    dict_store: Dict[Tuple[str, ...], Dict[str, Any]],
    seed: Optional[int] = None,
    moment: int = 0,
    n_samples: int = 10,
    compute_moments: bool = True,
    use_true_scm: bool = False) -> np.ndarray:
  """Get mean or variance of the causal effect by sampling with a given seed."""
  interventions = initialisation_utils.make_intervention_dict(graph)
  out = []
  for xval in values:
    # Check if we have a nested dictionary. This is needed to distinguish
    # between constrained and unconstrained settings. In unconstrained settings
    # dict_store is {v: {'xval': }}. In constrained it is {v: {c: {'xval': }}}
    # thus a nested dict. Notice that we might also have empty inner dict thus
    # we also need to check the len of the resulting values list.
    # if len(list(dict_store[x].values())) and isinstance(
    #     list(dict_store[x].values())[0], dict):
    if list(dict_store[x].values()) and isinstance(
        list(dict_store[x].values())[0], dict):
      # Computing var for the constraints
      stored_values = dict_store[x][y]
    else:
      # Computing var for the target
      stored_values = dict_store[x]

    # Check if the var value for xval has already been computed
    if str(xval) in stored_values:
      # If we have stored a list of samples and we want to compute the moments
      # in this function we need to take the average of the samples
      if isinstance(stored_values[str(xval)], list) and compute_moments:
        out.append(np.mean(stored_values[str(xval)]))
      else:
        out.append(stored_values[str(xval)])
    else:
      # Otherwise compute it and store it
      for intervention_variable, xx in zip(x, xval):
        interventions[intervention_variable] = xx

      get_samples = functools.partial(
          sampling_utils.sample_scm,
          interventions=interventions,
          n_samples=n_samples,
          compute_moments=compute_moments,
          moment=moment,
          seed=seed)

      if use_true_scm:
        # Sample from the true interventional distribution
        sample = get_samples(scm_funcs=true_scm_funcs, graph=None)
      else:
        # Sample from the estimated interventional distribution
        sample = get_samples(
            scm_funcs=fitted_scm().functions(),
            graph=graph)

      out.append(sample[y])

      stored_values[str(xval)] = sample[y]

  return np.vstack(out)


def get_product_mean_functions(
    graph: multidigraph.MultiDiGraph,
    target_variable: str,
    target_list: List[str],
    exploration_set: Set[str],
    fitted_scm: Callable[[], Any],
    true_scm_funcs: Callable[[], Any],
    mean_dict_store: Dict[Tuple[str, ...], Dict[str, Any]],
    mean_constraints_dict_store: Dict[
        Tuple[str, ...], Dict[str, Dict[str, Any]]
    ],
    seeds: List[int],
    n_samples: int = 10,
    compute_moments: bool = False,
    use_true_scm: bool = False,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
  """Wrapper around mean_function_internal to compute product of mean funcs."""

  def product_mean_function(x: np.ndarray,
                            x2: Optional[np.ndarray]) -> np.ndarray:
    mean_func = functools.partial(
        get_causal_effect_by_sampling,
        x=exploration_set,
        graph=graph,
        fitted_scm=fitted_scm,
        true_scm_funcs=true_scm_funcs,
        n_samples=n_samples,
        compute_moments=compute_moments,
        use_true_scm=use_true_scm)

    get_stored_values = functools.partial(
        utilities.get_stored_values,
        target_variable=target_variable,
        mean_dict_store=mean_dict_store,
        mean_constraints_dict_store=mean_constraints_dict_store)

    # If x2 is not None, we need compute the full covariance matrix across
    # points in x and in x2. The values to consider are given by all
    # couples where the first value is in x and the second is in x2. If instead
    # x2 is None we want to compute the diagonal of the correlation matrix which
    # is given by iterating over the couples of points of x with itself.
    if x2 is not None:
      values_to_compute = itertools.product(x, x2)
    else:
      values_to_compute = zip(x, x)

    out = []

    for xval, x2val in list(values_to_compute):
      target_1 = target_list[int(xval[-1])]
      target_2 = target_list[int(x2val[-1])]

      mean_1 = mean_func(
          values=[xval[:-1]],
          dict_store=get_stored_values(target=target_1),
          y=target_1,
          seed=seeds[0])
      mean_2 = mean_func(
          values=[x2val[:-1]],
          dict_store=get_stored_values(target=target_2),
          y=target_2,
          seed=seeds[1])

      # We need to squeeze the product ss the function
      # get_causal_effect_by_sampling returns a 3d tensor where the second and
      # third dimensions are one - with this we only keep a 1d vector
      product = np.squeeze(mean_1 * mean_2)
      if not compute_moments:
        # Average is NOT done in get_causal_effect_by_sampling
        # as we need to get the product before averaging the samples
        out.append(np.mean(product))
      else:
        # Average is already done in get_causal_effect_by_sampling
        # thus there is not need to do it here
        out.append(product)

    if x2 is not None:
      # Computing the full covariance matrix
      res = np.reshape(out, (x.shape[0], x2.shape[0]))
    else:
      # Computing the diagonal terms which gives a vector
      res = np.vstack(out)
    return res

  return product_mean_function


def update_sufficient_statistics_hat(
    graph: multidigraph.MultiDiGraph,
    y: str,
    x: Tuple[str, ...],
    fitted_scm: Callable[[], Any],
    true_scm_funcs: collections.OrderedDict[str, Any],
    mean_dict_store: Dict[Tuple[str, ...], Any],
    var_dict_store: Dict[Tuple[str, ...], Any],
    seed: Optional[int] = None,
    n_samples: int = 10,
    use_true_scm: bool = False,
) -> Tuple[Callable[[np.ndarray], Any], Callable[[np.ndarray], Any]]:
  """Updates the mean and variance functions (priors) on the causal effects."""
  # Initialize the function to compute the mean and variance of the causal
  # effects by sampling from the estimated SCM.
  mean_var_function = functools.partial(
      get_causal_effect_by_sampling,
      y=y,
      x=x,
      graph=graph,
      fitted_scm=fitted_scm,
      true_scm_funcs=true_scm_funcs,
      n_samples=n_samples,
      seed=seed,
      use_true_scm=use_true_scm)

  # moment=0 is the default thus we only need to pass the values at which to
  # compute the mean and the dict to stored the computed values
  def mean_function(values: np.ndarray) -> np.ndarray:
    return mean_var_function(values=values, dict_store=mean_dict_store)

  # To compute the variance of the samples we need to set moment=1 and provide
  # the relevant dict where to store  the values
  def variance_function(values: np.ndarray) -> np.ndarray:
    return mean_var_function(values=values, moment=1, dict_store=var_dict_store)

  return mean_function, variance_function


def fit_gp(
    x: np.ndarray,
    y: np.ndarray,
    lengthscale: float = 1.0,
    variance: float = 1.0,
    noise_var: float = 1.0,
    ard: bool = False,
    n_restart: int = 10,
    seed: int = 0,
):
  """Fits a Gaussian process."""
  # The random seed ensures that given the same data the optimization
  # of the GP model leads to the same optimized hyper-parameters.
  np.random.seed(seed)
  kernel = kern.RBF(
      x.shape[1], ARD=ard, lengthscale=lengthscale, variance=variance)

  model = gp_regression.GPRegression(
      X=x, Y=y, kernel=kernel, noise_var=noise_var)
  model.optimize_restarts(n_restart, verbose=False, robust=True)
  return model


def get_lenghscale_hp(all_vars: Sequence[str],
                      interventional_limits: Dict[str, Sequence[float]],
                      ratio_factor: float = 2.) -> float:
  """Get hyperparameter for the lenghscale of the RBF kernel of the GP model."""
  dist = 0.
  # If all_vars only include one variable transform this into a tuple
  all_vars = (all_vars,) if isinstance(all_vars, str) else all_vars

  for var in list(all_vars):
    limits = interventional_limits[var]
    dist += np.linalg.norm(limits[0] - limits[1])

  prior_lenghscale = (dist/len(all_vars))/ratio_factor
  return prior_lenghscale
