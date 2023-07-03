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

"""Graph functions."""
import itertools
from typing import Any, Dict, List, Optional, Tuple, Union

import graphviz
from networkx.algorithms import dag
from networkx.classes import multidigraph


def get_sorted_nodes(graph: multidigraph.MultiDiGraph) -> Dict[str, int]:
  """Returns dict of nodes in topological order."""
  sorted_nodes = {val: ix for ix, val in enumerate(dag.topological_sort(graph))}
  return sorted_nodes


def get_node_parents(
    graph: multidigraph.MultiDiGraph, node: str
) -> Tuple[str, ...]:
  """Returns the parents of the given node."""
  assert node in graph.nodes()
  # The fitted SCM functions expect inputs in a specific order
  # (the topological order). Hence the additional sorting.
  sorted_nodes = get_sorted_nodes(graph)
  return tuple(sorted(graph.predecessors(node), key=sorted_nodes.get))


def get_all_parents(graph: multidigraph.MultiDiGraph) -> Dict[str, Any]:
  """Get the parents for each node in the graph."""
  parents = {}
  for var in list(graph.nodes):
    parents[var] = list(get_node_parents(graph, var))
  return parents


def make_graphical_model(
    topology: str,
    nodes: List[str],
    target_node: Optional[str] = None,
    verbose: bool = False) -> Union[multidigraph.MultiDiGraph, str]:
  """Generic Bayesian network.

  Args:
    topology: Choice of independent and dependent causal topology
    nodes: List containing the nodes of the CGM
           e.g. nodes=['X', 'Z', 'Y']
    target_node: If we are using a independent spatial topology then we need to
                 specify the target node
    verbose : Whether to print the graph or not

  Returns:
    The DOT format of the graph or a networkx object
  """
  assert topology in ["dependent", "independent"]
  assert nodes

  if topology == "independent":
    assert target_node is not None
    assert isinstance(target_node, str)

  spatial_edges = []
  ranking = []
  # Check if target node is in the list of nodes, and if so remove it
  if topology == "independent":
    if target_node in nodes:
      nodes.remove(target_node)
    node_count = len(nodes)
    assert target_node not in nodes
    connections = node_count * "{} -> {}; "
    edge_pairs = list(sum([(item, target_node) for item in nodes], ()))
  else:
    node_count = len(nodes)
    connections = (node_count - 1) * "{} -> {}; "
    edge_pairs = []
    for pair in list(zip(nodes, nodes[1:])):
      for item in pair:
        edge_pairs.append(item)

  if topology == "independent":
    # X --> Y; Z --> Y
    all_nodes = nodes + [target_node]
    iters = [iter(edge_pairs)]
    inserts = list(itertools.chain(map(next, itertools.cycle(iters)), *iters))
    spatial_edges.append(connections.format(*inserts))
    ranking.append("{{ rank=same; {} }} ".format(" ".join(
        [item for item in all_nodes])))
  elif topology == "dependent":
    # X --> Z; Z --> Y
    iters = [iter(edge_pairs)]
    inserts = list(itertools.chain(map(next, itertools.cycle(iters)), *iters))
    spatial_edges.append(connections.format(*inserts))
    ranking.append("{{ rank=same; {} }} ".format(" ".join(
        [item for item in nodes])))
  else:
    raise ValueError("Not a valid spatial topology.")

  ranking = "".join(ranking)
  spatial_edges = "".join(spatial_edges)

  graph = "digraph {{ rankdir=LR; {} {} }}".format(spatial_edges, ranking)

  if verbose:
    return graphviz.Source(graph)
  else:
    return graph
