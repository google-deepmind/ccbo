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

"""Sampling utilities."""
from __future__ import annotations

import collections
from typing import Any, Dict, Optional, List

from networkx.classes import multidigraph
import numpy as np

from ccbo.utils import graph_functions


def sample_scm(
    scm_funcs: collections.OrderedDict[str, Any],
    graph: Optional[multidigraph.MultiDiGraph] = None,
    interventions: Optional[Dict[str, Any]] = None,
    n_samples: int = 10,
    compute_moments: bool = True,
    moment: Optional[int] = None,
    seed: Optional[int] = None) -> Dict[str, Any]:
  """Get samples or moments of samples from a SCM using the true or estimated functions.

  When using the estimated SCM functions these are currently fitted using
  Gaussian processes.

  Args:
    scm_funcs : functions in the SCM.
    graph : causal graph. If None the true functions in the SCM are used.
    interventions : Interventions to be implemented in the SCM. If None this
    functions samples from the observational distribution.
    n_samples: number of samples to get for each node in the SCM.
    compute_moments : Whether to aggregate the samples from the SCM to compute
      the moments of the observational or interventional distribution.
      If False the full array of samples is return.
    moment: which moment (0 or 1) to compute given the samples from the SCM.
      If moment = 0 this function returns the expected value.
      If moment = 1 this function returns the variance.
    seed: Seed to use to sample the exogenous variables in the SCM.
  Returns:
    Samples or moments of samples from the true or estimated distributions
    (observational or interventional) associated to the SCM.

  """
  if seed is not None:
    # This seed is controlling the sampling of the exogenous variables and of
    # the estimated functions in the SCM. When this is fixed both sources of
    # randomness are fixed.
    np.random.seed(seed)

  # Dictionary to store the average of the samples.
  sample = collections.OrderedDict([(k, []) for k in scm_funcs.keys()])

  for _ in range(n_samples):
    epsilon = {k: np.random.randn(1) for k in scm_funcs.keys()}

    # Dictionary to store one sample.
    tmp = collections.OrderedDict([(k, np.zeros(1)) for k in scm_funcs.keys()])

    # Loop over the nodes in the DAG and either assign the intervention
    # value or sample from the true or estimated functions in the SCM.
    for var, function in scm_funcs.items():
      if interventions and var in interventions and interventions[
          var] is not None:
        # Assign the intervened value. Note that if interventions exist they
        # take precedence.
        tmp[var] = interventions[var]
      else:
        # If the graph is given this function samples from the estimated
        # functions in the SCM. If it is not given the true functions are
        # used to sample.
        if graph:
          # Get the parents of the variable we are sampling. The parents are
          # used to get the right function to sample from in the dictionary of
          # estimated SCM functions.
          parents = graph_functions.get_node_parents(graph, var)
          if parents:
            # If the variable has parents sample from the estimated function.
            tmp[var] = function(parents, var, tmp)
          else:
            # If the variable does not have parents sample from
            # marginal distribution.
            tmp[var] = function((None, var))
        else:
          # Sample from true SCM.
          tmp[var] = function(epsilon[var], tmp)

      # Store a single sample.
      sample[var].append(tmp[var])

  # Aggregate the samples if compute moments is True or return the full stacked
  # array of samples otherwise.
  if compute_moments:
    if moment == 0:
      # Take the average of the samples for each node.
      sample = {k: np.array(np.mean(v)) for k, v in sample.items()}
    elif moment == 1:
      # Take the variance of the samples for each node.
      sample = {k: np.array(np.var(v)) for k, v in sample.items()}
    else:
      raise NotImplementedError('Moment {moment} not implemented.')
  else:
    # Stack the full list of samples obtained for each node.
    sample = {k: np.vstack(v) for k, v in sample.items()}

  return sample


def select_sample(sample: Dict[str, np.ndarray],
                  input_variables: List[str]) -> np.ndarray:
  """Returns a sample for the set of input variables.

  Args:
    sample : a sample from the SCM.
    input_variables : variables for which we want to get the values in sample.

  Returns:
    The sampled value(s) for the variable(s) given in input_variables.
  """
  if isinstance(input_variables, str):
    # input_variables only includes one variable which is given by a string
    return sample[input_variables].reshape(-1, 1)
  else:
    # input_variables includes multiple variables that are given
    # by a tuple() or a list()
    samp = []
    for node in input_variables:
      samp.append(sample[node].reshape(-1, 1))
    return np.hstack(samp)
