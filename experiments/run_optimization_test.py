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

"""Tests for experiments.run_optimization."""

import unittest

import numpy as np

from ccbo.experiments import data
from ccbo.methods import cbo
from ccbo.methods import ccbo
from ccbo.utils import initialisation_utils
from ccbo.utils import sampling_utils
from ccbo.utils import scm_utils


class RunOptimizationTest(unittest.TestCase):

  def test_run_optimization(self):
    example = data.EXAMPLES_DICT["synthetic1"]()
    scm = example.structural_causal_model(
        variables=("X", "Z"), lambdas=(1., 2.))
    constraints = scm.constraints
    graph = scm.graph
    exploration_sets = (("X",), ("Z",))
    intervention_domain = {"X": [-3, 2], "Z": [-1, 1]}
    precision = 5

    expected_values = {
        "cbo":
            np.array([-0.27992, -0.27992, -0.39242]),
        "ccbo_single_task":
            np.array([-0.27992, -0.39242, -0.39242]),
        "ccbo_single_task_causal_prior":
            np.array([-0.27992, -0.91705, -0.91705]),
        "ccbo_multi_task":
            np.array([-0.27992, -0.29237, -0.39242]),
        "ccbo_multi_task_causal_prior":
            np.array([-0.27992, -0.39242, -0.39242]),
        "ccbo_dag_multi_task":
            np.array([-0.27992, -0.39242, -0.39242])
    }
    # Generate observational data by sampling from the true
    # observational distribution
    d_o = sampling_utils.sample_scm(
        scm_funcs=scm.scm_funcs,
        graph=None,
        n_samples=5,
        compute_moments=False,
        seed=1)

    # Generate interventional data
    d_i = {k: None for k in exploration_sets}

    for var, level in zip(exploration_sets, ((1.,), (0.,))):
      initialisation_utils.assign_interventions(
          variables=var,
          levels=level,
          n_samples_per_intervention=100,
          sampling_seed=1,
          d_i=d_i,
          graph=graph,
          scm_funcs=scm.scm_funcs)

    for model_name in expected_values:
      use_causal_prior = model_name in [
          "ccbo_single_task_causal_prior", "ccbo_dag_multi_task"
      ]
      is_multi_task = model_name in [
          "ccbo_multi_task", "ccbo_multi_task_causal_prior",
          "ccbo_dag_multi_task"
      ]
      use_prior_mean = model_name in ["ccbo_single_task_causal_prior",
                                      "ccbo_multi_task_causal_prior",
                                      "ccbo_dag_multi_task"]
      # Setup input params
      input_params = {
          "graph": graph,
          "scm": scm,
          "make_scm_estimator": scm_utils.build_fitted_scm,
          "exploration_sets": list(exploration_sets),
          "observation_samples": d_o,
          "intervention_samples": d_i,
          "intervention_domain": intervention_domain,
          "number_of_trials": 3,
          "sample_anchor_points": False,
          "num_anchor_points": 5,
          "sampling_seed": 1,
          "n_restart": 1,
          "causal_prior": use_causal_prior,
          "hp_prior": True,

          # Noisy observations
          "noisy_observations": True,
          "n_samples_per_intervention": 100
      }

      if model_name == "cbo":
        model = cbo.CBO(**input_params)
      else:
        # Add constraints
        input_params["constraints"] = constraints
        input_params["multi_task_model"] = is_multi_task
        input_params["use_prior_mean"] = use_prior_mean

        if model_name == "ccbo_dag_multi_task":
          # Monte Carlo construction of the kernel
          input_params["n_kernel_samples"] = 10

        model = ccbo.CCBO(**input_params)

      # Run method
      model.run()
      np.testing.assert_array_almost_equal(
          model.optimal_outcome_values_during_trials,
          expected_values[model_name], precision)


if __name__ == "__main__":
  unittest.main()
