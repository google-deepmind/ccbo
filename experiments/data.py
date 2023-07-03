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

"""Implementation of SCM examples that we run experiments on."""

from __future__ import annotations

import abc
import collections
from typing import Any, Optional, Tuple

import graphviz
from networkx.classes import multidigraph
from networkx.drawing import nx_agraph
import numpy as np
import pygraphviz
from scipy import stats

from ccbo.scm_examples import scm
from ccbo.utils import utilities


class BaseExample(abc.ABC):
  """Abstract class for experiment examples."""

  def __init__(self):
    self._variables = None
    self._constraints = None

  @property
  def constraints(self) -> Any:
    return self._constraints

  @property
  def variables(self) -> Any:
    """Returns the variables dictionary."""
    return self._variables

  @abc.abstractproperty   # pylint: disable=deprecated-decorator
  def scm_funcs(self) -> collections.OrderedDict[str, Any]:
    """Returns the functions of the structural causal model."""
    raise NotImplementedError("scm_funcs should be implemented")

  @abc.abstractmethod
  def structural_causal_model(self, variables: Optional[Tuple[str, ...]],
                              lambdas: Optional[Tuple[float, ...]]) -> Any:
    """Returns the scm with fncs, variables and constraints."""
    raise NotImplementedError("structural_causal_model should be implemented")


class SyntheticExample1(BaseExample):
  """Synthetic example #1 - corresponds to DAG 1(c) in the cCBO paper."""

  @property
  def scm_funcs(self) -> collections.OrderedDict[str, Any]:
    """Define functions in SCM."""
    x = lambda noise, sample: noise
    z = lambda noise, sample: np.exp(-sample["X"]) + noise
    y = lambda noise, sample: np.cos(sample["Z"]) - np.exp(-sample["Z"] / 20.0  # pylint: disable=g-long-lambda
                                                          ) + noise
    return collections.OrderedDict([("X", x), ("Z", z), ("Y", y)])

  def structural_causal_model(self,
                              variables: Optional[Tuple[str, ...]],
                              lambdas: Optional[Tuple[float, ...]]) -> Any:
    self._variables = {
        "X": ["m", [-3, 2]],
        "Z": ["m", [-1, 1]],
        "Y": ["t"],
    }
    if variables is not None and lambdas is not None:
      self._constraints = {
          var: [utilities.Direction.LOWER, val]
          for var, val in zip(variables, lambdas)
      }

    return scm.Scm(
        constraints=self.constraints,
        scm_funcs=self.scm_funcs,
        variables=self.variables)


class SyntheticExample2(BaseExample):
  """Synthetic example #2 - corresponds to DAG 1(d) in the cCBO paper."""

  @property
  def scm_funcs(self) -> collections.OrderedDict[str, Any]:
    """Define functions in SCM."""
    a = lambda noise, sample: noise
    b = lambda noise, sample: noise
    c = lambda noise, sample: np.exp(-sample["A"]) / 5. + noise
    d = lambda noise, sample: np.cos(sample["B"]) + sample["C"] / 10. + noise
    e = lambda noise, sample: np.exp(-sample["C"]) / 10. + noise
    y = lambda noise, sample: np.cos(sample["D"]) - sample["D"] / 5. + np.sin(  # pylint: disable=g-long-lambda
        sample["E"]) - sample["E"] / 4. + noise
    return collections.OrderedDict([("A", a), ("B", b), ("C", c), ("D", d),
                                    ("E", e), ("Y", y)])

  def graph(self) -> multidigraph.MultiDiGraph:
    """Define causal graph structure."""
    ranking = []
    nodes = ["A", "B", "C", "D", "E", "Y"]
    myedges = ["A -> C; C -> E; B -> D; D -> Y; C -> D; E -> Y"]
    ranking.append("{{ rank=same; {} }} ".format(" ".join(nodes)))
    ranking = "".join(ranking)
    edges = "".join(myedges)
    graph = "digraph {{ rankdir=LR; {} {} }}".format(edges, ranking)
    dag = nx_agraph.from_agraph(
        pygraphviz.AGraph(graphviz.Source(graph).source))
    return dag

  def structural_causal_model(self,
                              variables: Optional[Tuple[str, ...]],
                              lambdas: Optional[Tuple[float, ...]]) -> Any:
    self._variables = {
        "A": ["m", [-5, 5]],
        "B": ["nm", [-4, 4]],
        "C": ["nm", [0, 10]],
        "D": ["m", [-1, 1]],
        "E": ["m", [-1, 1]],
        "Y": ["t"],
    }

    if variables is not None and lambdas is not None:
      self._constraints = {
          var: [utilities.Direction.LOWER, val]
          for var, val in zip(variables, lambdas)
      }

    return scm.Scm(
        constraints=self.constraints,
        scm_funcs=self.scm_funcs,
        variables=self.variables,
        graph=self.graph())


class HealthExample(BaseExample):
  """Real example #1 - corresponds to Fig 1(a) in the cCBO paper."""

  @property
  def scm_funcs(self) -> collections.OrderedDict[str, Any]:
    """Define equations in SCM."""
    a = lambda noise, sample: np.random.uniform(low=55, high=75)  # age
    # bmr - base metabolic rate
    b = lambda noise, sample: stats.truncnorm.rvs(-1, 2) * 10 + 1500.
    c = lambda noise, sample: np.random.uniform(low=-100, high=100)  # calories
    # height
    d = lambda noise, sample: stats.truncnorm.rvs(-0.5, 0.5) * 10 + 175.

    e = lambda noise, sample: (sample["B"] + 6.8 * sample["A"] - 5 * sample["D"]  # pylint: disable=g-long-lambda
                              ) / 13.7 + sample["C"] * 150. / 7716.  # weight

    f = lambda noise, sample: sample["E"] / ((sample["D"] / 100)**2)  # bmi

    g = lambda noise, sample: np.random.uniform(low=0, high=1)  # statin
    h = lambda noise, sample: utilities.sigmoid(-8.0 + 0.10 * sample["A"] + 0.03  # pylint: disable=g-long-lambda
                                                * sample["F"])  # aspirin
    i = lambda noise, sample: utilities.sigmoid(2.2 - 0.05 * sample[  # pylint: disable=g-long-lambda
        "A"] + 0.01 * sample["F"] - 0.04 * sample["G"] + 0.02 * sample["H"]
                                               )  # cancer

    y = lambda noise, sample: np.random.normal(  # pylint: disable=g-long-lambda
        6.8 + 0.04 * sample["A"] - 0.15 * sample["F"] - 0.60 * sample["G"] +
        0.55 * sample["H"] + 1.00 * sample["I"], 0.4)  # psa
    return collections.OrderedDict([("A", a), ("B", b), ("C", c), ("D", d),
                                    ("E", e), ("F", f), ("G", g), ("H", h),
                                    ("I", i), ("Y", y)])

  def graph(self) -> multidigraph.MultiDiGraph:
    """Define causal graph structure."""
    ranking = []
    nodes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "Y"]
    myedges = [
        "A -> F; A -> G;  A -> I; A -> H; A -> E; B -> E; C -> E; D -> E; D ->"  # pylint: disable=implicit-str-concat
        " F; E -> F; F -> H;  F -> I; G -> I; G -> Y; H -> Y; H -> I; I -> Y"
    ]

    ranking.append("{{ rank=same; {} }} ".format(" ".join(nodes)))
    ranking = "".join(ranking)
    edges = "".join(myedges)
    graph = "digraph {{ rankdir=LR; {} {} }}".format(edges, ranking)
    dag = nx_agraph.from_agraph(
        pygraphviz.AGraph(graphviz.Source(graph).source))
    return dag

  def structural_causal_model(self,
                              variables: Optional[Tuple[str, ...]],
                              lambdas: Optional[Tuple[float, ...]]) -> Any:
    self._variables = {
        "A": ["nm", [55, 75]],
        "B": ["nm", [1450, 1550]],
        "C": ["m", [-400, +400]],
        "D": ["nm", [169, 180]],
        "E": ["nm", [68, 86]],
        "F": ["nm", [19, 25]],
        "G": ["m", [0, 1]],
        "H": ["m", [0, 1]],
        "I": ["nm", [0.2, 0.5]],
        "Y": ["t"],
    }

    if variables is not None and lambdas is not None:
      self._constraints = {
          var: [utilities.Direction.LOWER, val]
          for var, val in zip(variables, lambdas)
      }

    return scm.Scm(
        constraints=self.constraints,
        scm_funcs=self.scm_funcs,
        variables=self.variables,
        graph=self.graph())


EXAMPLES_DICT = {
    "synthetic1": SyntheticExample1,
    "synthetic2": SyntheticExample2,
    "health": HealthExample,
}
