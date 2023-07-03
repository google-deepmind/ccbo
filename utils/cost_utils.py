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

"""Functions to define the cost structure."""
import collections
from typing import Tuple, Callable, List, Any, OrderedDict

from emukit.core import acquisition as acq
import numpy as np


class Cost(acq.Acquisition):
  """Class for computing the cost of each intervention."""

  def __init__(self, costs_functions: OrderedDict[str, Callable[[Any], Any]],
               exploration_set: Tuple[str, ...], target: str):
    self.costs_functions = costs_functions
    self.exploration_set = exploration_set
    self.target = target

  def evaluate(self, x: Any) -> np.ndarray[Any, Any]:
    if len(self.exploration_set) == 1:
      #  Univariate intervention
      return self.costs_functions[self.exploration_set[0]](x)
    else:
      # Multivariate intervention
      cost = []
      for i, es_member in enumerate(self.exploration_set):
        cost.append(self.costs_functions[es_member](x[:, i]))
    return sum(cost)

  @property
  def has_gradients(self)-> bool:
    return True


def define_costs(manipulative_variables: List[str],
                 type_cost: int,
                 fix_cost: float = 1.0
                ) -> OrderedDict[str, Callable[[Any], Any]]:
  """Initialize dict with functions to compute cost."""

  if type_cost == 1:
    fix_cost_function = lambda x: fix_cost
    costs = collections.OrderedDict([
        (var, fix_cost_function) for var in manipulative_variables
    ])
  else:
    raise NotImplementedError("Not yet implemented")
  return costs


def total_intervention_cost(es: Tuple[str, ...],
                            costs: OrderedDict[str, Callable[[Any], Any]],
                            x: np.ndarray) -> float:
  total_cost = 0.0
  for i, es_member in enumerate(es):
    total_cost += costs[es_member](x[:, i])
  return total_cost
