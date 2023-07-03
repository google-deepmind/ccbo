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

"""Config for Synthetic2 experiment."""

import ml_collections


def get_config():
  """Return the default configuration for synthetic2 (Fig 1(d) in the paper) example."""
  config = ml_collections.ConfigDict()

  # Name associated with this SCM
  config.example_name = 'synthetic2'

  config.n_trials = 60  # Number of trials to run.
  config.n_samples_obs = 100  # Number of initial observational data points.
  # Number of samples per interventional distribution.
  config.n_samples_per_intervention = 100

  # Number to sample to use to get the ground truth function
  config.n_samples_ground_truth = 100

  # Seed to use to sample the anchor points.
  config.seed_anchor_points = 1
  # Use a regular grid of points to evaluate the acquisition function
  # or sample points uniformly.
  config.sample_anchor_points = False

  # Number of points on a regular grid to evaluate the acquisition function.
  config.n_grid_points = 100

  # Learn or fix the likelihoood noise in the GP model.
  config.fix_likelihood_noise_var = True

  # Learn or fix the likelihoood noise in the GP model.
  config.noisy_acquisition = False

  # First type
  config.intervention_variables = (
      ('A',), ('D',), ('E',), ('A', 'D'), ('A', 'E'), ('D', 'E')
  )  # Intervention variables.
  config.intervention_levels = (
      (0.,), (1.,), (1.,), (0., 1.), (0., 1.), (1., 1.))  # Intervention values.

  config.constraints = ml_collections.ConfigDict()
  config.constraints.variables = ('C', 'D', 'E')
  config.constraints.lambdas = (10., 10., 10.)  # Constraint values

  # Exploration sets
  config.exploration_sets = (('A',), ('D',), ('E',), ('A', 'D'), ('A', 'E'),
                             ('D', 'E'))

  # Sum the RBF kernel to the Monte Carlo one
  config.add_rbf_kernel = False

  # Whethere to update the SCM at every iteration for G-MTGP
  config.update_scm = False

  # Use hp_prior in kernel
  config.use_hp_prior = True

  # Number of samples for the kernel computation
  config.n_kernel_samples = 10

  # Specify which model to run with possible values:
  # "cbo", "ccbo_single_task", "ccbo_single_task_causal_prior",
  # "ccbo_multi_task", "ccbo_multi_task_causal_prior", "ccbo_dag_multi_task"
  config.model_name = 'ccbo_single_task'

  return config
