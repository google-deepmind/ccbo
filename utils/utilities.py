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

"""General utilities."""
import enum
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import stats


class Trial(enum.Enum):
  """Type of trial, i.e. interventional or observational."""
  INTERVENTION = 0
  OBSERVATION = 1


class Task(enum.Enum):
  """Task can be either minimization or maximation."""
  MIN = "min"
  MAX = "max"


class Direction(enum.Enum):
  """The direction of the constraint can be either < or >."""
  LOWER = "<"
  HIGHER = ">"


class VariableType(enum.Enum):
  """The types of variables included in the SCM.

  These can be of four types:
    - Target variables = "t"
    - Manipulative variable = "m"
    - Non Manipulative variable = "nm"
    - Unobserved counfounder = "u"
    - Protected variable = "p"
  """
  TARGET = "t"
  MANIPULATIVE = "m"
  NONMANIPULATIVE = "nm"
  UNOBSERVED = "u"


EVAL_FN = {Task.MIN: min, Task.MAX: max}
ARG_EVAL_FN = {Task.MIN: np.argmin, Task.MAX: np.argmax}
A_EVAL_FN = {Task.MIN: np.amin, Task.MAX: np.amax}


def sigmoid(x: float) -> float:
  return 1 / (1 + math.exp(-x))


def get_stored_values(
    target: str, target_variable: str,
    mean_dict_store: Dict[Tuple[str, ...], Dict[str, Any]],
    mean_constraints_dict_store: Dict[Tuple[str, ...],
                                      Dict[str, Dict[str, Any]]]
) -> Dict[Tuple[str, ...], Dict[str, Any]]:
  if target == target_variable:
    dict_store = mean_dict_store
  else:
    dict_store = mean_constraints_dict_store
  return dict_store


def get_standard_normal_pdf_cdf(
    x: float, mean: np.ndarray, standard_deviation: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Returns pdf and cdf of standard normal evaluated at (x - mean)/sigma."""
  u = (x - mean) / standard_deviation
  pdf = stats.norm.pdf(u)
  cdf = stats.norm.cdf(u)
  return u, pdf, cdf


def standard_mean_function(x: np.ndarray) -> np.ndarray:
  """Function to get zero mean for the causal kernel."""
  return np.zeros_like(x)


def zero_variance_adjustment(x: np.ndarray) -> np.ndarray:
  """Function to get zero adjustment for the variance of the causal kernel."""
  return np.zeros_like(x)


def make_column_shape_2d(x: Any) -> Any:
  """Reshapes an array to create a 2-d column."""
  return np.array([x]).reshape(-1, 1)


def check_reshape_add_data(
    interventional_data_x: Dict[Tuple[str, ...], Optional[Any]],
    interventional_data_y: Dict[Tuple[str, ...], Optional[Any]],
    new_interventional_data_x: Any,
    y_new: float, best_es: Tuple[str, ...],
) -> Tuple[Optional[Any], Optional[Any]]:
  """Checks whether interventional data needs reshaping and adds values."""
  if (interventional_data_x[best_es] is not None and
      interventional_data_y[best_es] is not None):
    if len(new_interventional_data_x.shape) == 1:
      new_interventional_data_x = make_column_shape_2d(
          new_interventional_data_x)

    assert interventional_data_x[best_es].shape[
        1] == new_interventional_data_x.shape[1]

    # Update interventional data X
    interventional_data_x[best_es] = np.vstack(
        (interventional_data_x[best_es], new_interventional_data_x)
    )
    # Update interventional data Y
    interventional_data_y[best_es] = np.vstack(
        (interventional_data_y[best_es], make_column_shape_2d(y_new),)
    )
  else:
    # Assign new interventional data

    if len(new_interventional_data_x.shape) == 1 and len(best_es) == 1:
      reshaped_new_interventional_data_x = make_column_shape_2d(
          new_interventional_data_x)
    elif len(best_es) > 1 and len(new_interventional_data_x.shape) == 1:
      reshaped_new_interventional_data_x = new_interventional_data_x.reshape(
          1, -1)
    else:
      reshaped_new_interventional_data_x = new_interventional_data_x

    #  Assign X and Y
    interventional_data_x[best_es] = reshaped_new_interventional_data_x
    interventional_data_y[best_es] = make_column_shape_2d(y_new)

  assert (
      np.shape(interventional_data_x[best_es])[0]
      == np.shape(interventional_data_y[best_es])[0]
  )

  return (
      interventional_data_x[best_es],
      interventional_data_y[best_es],
  )


def get_monte_carlo_expectation(
    intervention_samples: Dict[str, np.ndarray]) -> Dict[str, Any]:
  """Returns the expected value of the intervention via MC sampling."""
  expectation = {k: None for k in intervention_samples.keys()}
  for es in expectation.keys():
    expectation[es] = intervention_samples[es].mean(axis=0)

  return expectation

