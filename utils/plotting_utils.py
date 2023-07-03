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

"""Plotting utilities."""

from typing import Any, Dict, List, Optional, Tuple

from emukit.model_wrappers import gpy_model_wrappers
from matplotlib import pyplot as plt
import numpy as np

from ccbo.utils import utilities


def plot_acquisition(inputs: np.ndarray, improvement: np.ndarray,
                     x_new: np.ndarray,
                     probability_feasibility: Optional[np.ndarray] = None,
                     multi_task_model: bool = False) -> None:
  """Plot the acquisition function."""
  # Plot expected improvement
  plt.plot(inputs, improvement, label='EI')
  # Plot probability_feasibility is this is not none
  if probability_feasibility is not None:
    if not isinstance(probability_feasibility, float) and not multi_task_model:
      # If probability of feasibility is one everywhere do not plot it
      if int(np.sum(probability_feasibility)) != inputs.shape[0]:
        plt.plot(inputs, probability_feasibility, label='PF')

      plt.plot(inputs, improvement * probability_feasibility, label='cEI')
  # Plot new selected point
  plt.axvline(
      x=x_new, color='red', linestyle='-', label='new point is:' + str(x_new))
  plt.legend()
  plt.show()


def plot_outcome(
    n: int,
    outcomes: List[Any],
    labels: List[str],
    title: Optional[str] = None,
    true_objective_values: Optional[List[float]] = None) -> None:
  """Plot convergence results."""
  _, ax = plt.subplots(1, figsize=(6, 6), sharex=True)

  for ii, out in enumerate(outcomes):
    ax.plot(out, lw=2, label=labels[ii], alpha=0.5)
  if true_objective_values:
    ax.hlines(
        true_objective_values,
        0,
        n,
        'red',
        ls='--',
        lw=1,
        alpha=0.7,
        label='Ground truth')

  ax.set_ylabel(r'$y^*$')
  ax.grid(True)
  ax.legend(
      ncol=3,
      fontsize='medium',
      loc='center',
      frameon=False,
      bbox_to_anchor=(0.5, 1.2))

  ax.set_xlabel(r'Trials')
  ax.set_xlim(0, n)
  if title:
    plt.title(title)

  plt.subplots_adjust(hspace=0)
  plt.show()


def plot_save_outcome(
    n: float,
    outcomes: List[Any],
    labels: List[str],
    true_objective_values: Optional[List[float]] = None,
) -> None:
  """Plot convergence results."""
  _, ax = plt.subplots(1, figsize=(6, 6), sharex=True)

  j = 0
  for ii, out in enumerate(outcomes):
    ax.plot(out[j][1:], lw=2, label=labels[ii], alpha=0.5)
  if true_objective_values:
    ax.hlines(
        true_objective_values[j],
        0,
        n,
        'red',
        ls='--',
        lw=1,
        alpha=0.7,
        label='Ground truth')
  ax.set_ylabel(r'$y^*_{}$'.format(j))
  ax.grid(True)
  ax.legend(
      ncol=3,
      fontsize='medium',
      loc='center',
      frameon=False,
      bbox_to_anchor=(0.5, 1.2))

  ax.set_xlabel(r'Trials')
  ax.set_xlim(0, n - 2)

  plt.subplots_adjust(hspace=0)

  plt.close()


def plot_models(
    model: Any,  # Can be dict for constraints or bo model for target
    exploration_sets: List[Tuple[str, ...]],
    ground_truth: Any,
    interventional_grids: Dict[Tuple[str, ...], np.ndarray],
    interventional_data_x: Dict[Tuple[str, ...], Any],
    interventional_data_y: Dict[Tuple[str, ...], Any],
    multi_task_model: bool = False) -> None:

  """Plots a set models."""
  for es in exploration_sets:
    # Only plot is the input space is one dimensional
    if len(es) == 1:
      inputs = np.asarray(interventional_grids[es])

      if isinstance(model[es], dict):
        # We are plotting the constraints
        for i, p in enumerate(list(model[es].keys())):
          true_vals = ground_truth[es][p]
          plot_single_model(inputs, i + 1, model[es][p], multi_task_model,
                            true_vals, interventional_data_x[es],
                            interventional_data_y[es][p])
      else:
        # We are plotting the target
        true_vals = utilities.make_column_shape_2d(ground_truth[es])
        plot_single_model(inputs, 0, model[es], multi_task_model, true_vals,
                          interventional_data_x[es], interventional_data_y[es])


def plot_single_model(inputs: np.ndarray, task: int,
                      single_model: gpy_model_wrappers.GPyModelWrapper,
                      multi_task_model: bool, ground_truth: np.ndarray,
                      data_x: np.ndarray, data_y: np.ndarray) -> None:
  """Plots a single model."""
  if single_model is not None:
    if multi_task_model:
      # The constraint functions correspond to the indices 0-p where
      # p is the total number of tasks in a multi-task model. In order to
      # predict the inputs need to be augmented with the task index.
      inputs = np.concatenate([inputs, task * np.ones((inputs.shape[0], 1))],
                              axis=1)

    mean, var = single_model.predict(inputs)

    plt.scatter(data_x, data_y)
    # GP variance
    plt.fill_between(
        inputs[:, 0], (mean - var)[:, 0], (mean + var)[:, 0], alpha=0.2)
    # GP mean
    plt.plot(inputs[:, 0], mean, 'b', label='posterior mean')

    # True function
    plt.plot(inputs[:, 0], ground_truth, 'r', label='True')

    plt.legend()
    plt.show()
