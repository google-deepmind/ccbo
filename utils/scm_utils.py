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

"""SCM utilities."""
from __future__ import annotations

import collections
from typing import Any, Callable, Dict, Optional, Tuple

from networkx.classes import multidigraph
import numpy as np
from sklearn import neighbors  #  StatsModels works better

from ccbo.utils import gp_utils
from ccbo.utils import graph_functions
from ccbo.utils import sampling_utils


def _make_marginal(
    scm_fncs: Dict[Tuple[Optional[Any], ...], Any]
) -> Callable[[Any], np.ndarray]:
  """Get a function that samples from the marginal distribution.

  Args:
    scm_fncs : fitted functions for the SCM.

  Returns:
    Function that returns a sample from a marginal distribution.
  """
  # Get a sample for the exogenous node.
  return lambda var: scm_fncs[var].sample()


def _make_conditional(
    scm_fncs: Dict[Tuple[Optional[Any], ...], Any]
) -> Callable[[Any, Any, Any], np.ndarray]:
  """Get a function that samples from the conditional distribution.

  Args:
    scm_fncs : fitted functions for the SCM.
  Returns:
    Function that returns a sample from a conditional distribution corresponding
    to the fitted SCM function for input_vars-output_var.
  """

  # This function constructs a function that returns a sample for an endogenous
  # node. As we only get one sample we need np.squeeze(.,axis=2) to get rid of
  # the 3rd dimension returned by posterior_samples_f which is equal to one.

  def sample_endogenous(input_vars, output_var, sample):
    # Get the values of the input variables in the sample
    selected_sample = sampling_utils.select_sample(sample, input_vars)
    # Get the estimated variance for the likelihood noise corresponding to the
    # GP function mapping input_vars to output_var.
    variance_likelihood_noise = scm_fncs[(input_vars,
                                          output_var)].likelihood.variance[0]
    # We want to sample each endogenous node including the exogenous random
    # variables which is assumed to be normally distributed with 0 mean and
    # variance given by variance_likelihood_noise. We thus sample from a
    # Gaussian random variable with the variance_likelihood_noise learned by
    # maximum likelihood when fitting the functions in the SCM.
    sample_likelihood_noise = np.random.normal(
        loc=np.zeros((1, 1)),
        scale=np.ones((1, 1)) * np.sqrt(variance_likelihood_noise))

    # Sample from the fitted function in the SCM and add the exogenous noise.
    sample = np.squeeze(
        scm_fncs[(input_vars, output_var)].posterior_samples_f(
            selected_sample, full_cov=True, size=1),
        axis=2) + sample_likelihood_noise
    return sample

  return sample_endogenous


def build_fitted_scm(
    graph: multidigraph.MultiDiGraph,
    fitted_scm_fncs: Dict[Tuple[Optional[Any], ...], Any],
    scm_fncs: collections.OrderedDict[str, Any]) -> Any:
  """Create the fitted SCM using the estimated functions for the graph edges.

  Args:
    graph : causal graph.
    fitted_scm_fncs : fitted functions for the SCM.
    scm_fncs : true SCM.

  Returns:
    A fitted SCM class with functions to sample from it.
  """

  class FittedSCM:
    """Fitted SCM class."""

    def __init__(self):
      self.graph = graph
      self.fitted_scm_fncs = fitted_scm_fncs
      self.scm_fncs = scm_fncs

    def functions(self) -> collections.OrderedDict[str, Any]:
      """Store functions sampling from the fitted SCM functions."""
      # SCM functions
      f = collections.OrderedDict()
      for v in list(self.scm_fncs.keys()):
        if self.graph.in_degree[v] == 0:
          # Exogenous node
          f[v] = _make_marginal(self.fitted_scm_fncs)
        else:
          # Endogenous node
          f[v] = _make_conditional(self.fitted_scm_fncs)
      return f

  return FittedSCM


def fit_scm_fncs(
    graph: multidigraph.MultiDiGraph,
    data: Dict[str, Any],
    scm: collections.OrderedDict[str, Any],
    n_restart: int = 10,
    kernel_density: str = "gaussian") -> Dict[Tuple[Optional[Any], ...], Any]:
  """Fit functions in the SCM.

  Args:
    graph : causal graph.
    data : observations from the true SCM.
    scm : true SCM.
    n_restart: n_restart for hyperparameters optimization.
    kernel_density: Type of kernel to fit to estimate the distributions of the
      exogenous variables. Options are "gaussian", "tophat", "epanechnikov",
      "exponential", "linear" and "cosine". Default is "gaussian".

  Returns:
    Dictionary containing the estimated SCM functions. For the endogenous nodes
    in the graph (nodes detemined by other variables in the graph), the key in
    the dictionary is a tuple giving (parents_of_node, node) while each value is
    the estimated function (currently this is a GP) for the SCM.
    For the exogenous nodes in the DAG (nodes without parents), the key in the
    dictionary is (Node, node) while the associated value is given by a kernel
    density estimator (KDE).
  """
  # Get all nodes in the graph
  nodes = list(scm.keys())
  # Get parents for each node
  parents = graph_functions.get_all_parents(graph)
  # Initialise SCM functions
  fncs = {}

  # Assign estimators to each node in the graph
  for v in nodes:
    if not parents[v]:
      # Exogenous node, add function to the dict with key = (None, v)
      xx = data[v]
      fncs[(None, v)] = neighbors.KernelDensity(kernel=kernel_density).fit(xx)
    else:
      # Endogenous nodes, add function to the dict with
      # key = (tuple(parents[v]), v)
      data_parents = []
      for j in parents[v]:
        data_parents.append(data[j])
      xx = np.hstack(data_parents)
      yy = data[v]
      fncs[(tuple(parents[v]), v)] = gp_utils.fit_gp(
          x=xx, y=yy, n_restart=n_restart)

  return fncs
