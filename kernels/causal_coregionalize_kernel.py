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

"""Implement the kernel for a multi-task model with coavriance structure induced by the causal graph."""
import functools
from typing import Any, Dict, Optional, Callable, List, Set, Tuple

from GPy import kern
from networkx.classes import multidigraph
import numpy as np

from ccbo.utils import gp_utils


class CausalCoregionalize(kern.Kern):
  """Covariance function for causal multi-task model.

    input_dim: input dimensionality.
    target_list: list of target variables e.g. Y and the constrained variables.
    graph: DAG.
    target_variable: name of target variable.
    exploration_set: intervened variables.
    fitted_scm: estimated SCM object.
    true_scm_funcs: true functions in the SCM.
    dict_mean_product: dictionary of stored values for the mean of the product
    of the target causal effects computed at different intervention levels.
    dict_constraints_mean_product: dictionary of stored values for the mean of
    the product of the constraint causal effects computed at different
    intervention levels.
    dict_product_mean: dictionary of stored values for the product of the mean
    causal effect on the target variable.
    dict_constraints_product_mean: dictionary of stored values for the product
    of the mean causal effects on the constrained variables.
    active_dims: active dimensions of inputs X we will work on. When the
    dimensionsality of an input X is D this parameter determined if all D
    dimensions should be used to compute the kernel. This is the case when
    active_dims = None. When instead active_dims is not None and, for instance,
    we have active_dims=0, the kernel will be computed with only the first
    column that is x[:,0].
    seed: random seed used to sample from the estimated SCM.
    n_samples: number of samples for the Monte Carlo estimates.
    name: name of the kernel.
  """

  def __init__(
      self,
      input_dim: int,
      target_list: List[str],
      graph: multidigraph.MultiDiGraph,
      target_variable: str,
      exploration_set: Set[str],
      fitted_scm: Optional[Callable[[], Any]],
      true_scm_funcs: Optional[Callable[[], Any]],
      dict_mean_product: Dict[Tuple[str, ...], Any],
      dict_constraints_mean_product: Dict[Tuple[str, ...], Dict[str, Any]],
      dict_product_mean: Dict[Tuple[str, ...], Any],
      dict_constraints_product_mean: Dict[Tuple[str, ...], Dict[str, Any]],
      active_dims: Optional[int] = None,
      seed: int = 1,
      n_samples: int = 10,
      use_true_scm: bool = False,
      name: str = "causal_coregionalize"):
    args = {"input_dim": input_dim, "active_dims": active_dims, "name": name}
    super().__init__(**args)
    self.graph = graph
    self.exploration_set = exploration_set

    # Initializing the function to compute the product of the mean or the mean
    # of the product of the causal effects by sampling from the SCM to get
    # a Monte Carlo estimate.
    get_product_mean_functions = functools.partial(
        gp_utils.get_product_mean_functions,
        graph=graph,
        target_variable=target_variable,
        target_list=target_list,
        exploration_set=exploration_set,
        fitted_scm=fitted_scm,
        true_scm_funcs=true_scm_funcs,
        n_samples=n_samples,
        use_true_scm=use_true_scm)

    # When computing the mean of the product we first multiply the causal
    # effects on the individual variables and then take the average. With
    # compute_moments=False we avoid computing the moments inside the sampling
    # function and do it only after having multiplied the samples
    self.mean_product = get_product_mean_functions(
        compute_moments=False, seeds=[seed, seed],
        mean_dict_store=dict_mean_product,
        mean_constraints_dict_store=dict_constraints_mean_product
        )
    # When computing the product of the mean we can take the moment of the
    # samples inside the sampling function (compute_moments=True) and multiply
    # those afterwards
    self.product_mean = get_product_mean_functions(
        compute_moments=True, seeds=[seed, seed],
        mean_dict_store=dict_product_mean,
        mean_constraints_dict_store=dict_constraints_product_mean
        )

  def K(self, x: np.ndarray, xprime: Optional[np.ndarray] = None) -> np.ndarray:
    # The kernel is computed as E[mu_Xmu_X2] - E[mu_X]E[mu_X2] where mu is the
    # causal effect of the exploration set on X and X2. These are the variables
    # corresponding to the values of the last dimension of the input x and x2.
    if xprime is None:
      xprime = x
    return self.mean_product(x, xprime) - self.product_mean(x, xprime)

  def Kdiag(self, x: np.ndarray) -> np.ndarray:
    return (self.mean_product(x, None) - self.product_mean(x, None))[:, 0]

  def to_dict(self) -> Dict[str, Any]:
    input_dict = self._save_to_input_dict()
    input_dict["class"] = "GPy.kern.CausalCoregionalize"
    return input_dict

  def gradients_X(self, *unused_args):
    pass

  def gradients_X_diag(self, *unused_args):
    pass

  def update_gradients_full(self, *unused_args):
    pass

  def update_gradients_diag(self, *unused_args):
    pass
