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

"""Run experiment."""
from __future__ import annotations

from absl import app
from absl import flags
from absl import logging
from ml_collections import config_flags

from ccbo.experiments import data
from ccbo.methods import cbo
from ccbo.methods import ccbo
from ccbo.methods import random
from ccbo.utils import constraints_functions
from ccbo.utils import initialisation_utils
from ccbo.utils import sampling_utils
from ccbo.utils import scm_utils


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config",
                                "ccbo/experiments/config_synthetic1.py")

### FIXED PARAMETERS ###
# Sampling seed for the ground truth and the sampling of the target function
sampling_seed = 1
# Whether to use noisy observations of the target and the constraints
noisy_observations = True
# Produce plot and print statistics
verbose = False
# Number of restarts of GP optimization
n_restart = 5


def main(_):
  flags_config = FLAGS.config
  logging.info("Flags dict is %s", flags_config)

  ### MISCELLANEOUS PREPARATION ###
  example = data.EXAMPLES_DICT[flags_config.example_name]()
  scm = example.structural_causal_model(**flags_config.constraints)
  graph = scm.graph
  constraints = scm.constraints

  (_, _, intervention_domain, all_ce,
   _, constraints_values, _,
   _, _,
   _) = scm.setup(
       n_grid_points=flags_config.n_grid_points,
       exploration_sets=list(flags_config.exploration_sets),
       n_samples=flags_config.n_samples_ground_truth,
       sampling_seed=sampling_seed)

  ### GENERATE INITIAL DATA ###

  # Generate observational data by sampling from the true
  # observational distribution
  d_o = sampling_utils.sample_scm(
      scm_funcs=scm.scm_funcs,
      graph=None,
      n_samples=flags_config.n_samples_obs,
      compute_moments=False,
      seed=sampling_seed)

  # Generate interventional data
  d_i = {k: None for k in flags_config.exploration_sets}

  for var, level in zip(flags_config.intervention_variables,
                        flags_config.intervention_levels):
    initialisation_utils.assign_interventions(
        variables=var,
        levels=level,
        n_samples_per_intervention=flags_config.n_samples_per_intervention,
        sampling_seed=sampling_seed,
        d_i=d_i,
        graph=graph,
        scm_funcs=scm.scm_funcs)

  ### RUN THE ALGORITHM ###
  model_name = flags_config.model_name
  use_causal_prior = model_name in [
      "ccbo_single_task_causal_prior", "ccbo_dag_multi_task"
  ]
  is_multi_task = model_name in [
      "ccbo_multi_task", "ccbo_multi_task_causal_prior", "ccbo_dag_multi_task"
  ]
  use_prior_mean = model_name in ["ccbo_single_task_causal_prior",
                                  "ccbo_multi_task_causal_prior",
                                  "ccbo_dag_multi_task"]
  add_rbf_kernel = (
      flags_config.add_rbf_kernel and model_name in ["ccbo_dag_multi_task"])
  update_scm = flags_config.update_scm and model_name in ["ccbo_dag_multi_task"]

  # Setup input params
  input_params = {
      "graph": graph,
      "scm": scm,
      "make_scm_estimator": scm_utils.build_fitted_scm,
      "exploration_sets": list(flags_config.exploration_sets),
      "observation_samples": d_o,
      "intervention_samples": d_i,
      "intervention_domain": intervention_domain,
      "number_of_trials": flags_config.n_trials,
      "sample_anchor_points": flags_config.sample_anchor_points,
      "seed_anchor_points": flags_config.seed_anchor_points,
      "num_anchor_points": flags_config.n_grid_points,
      "ground_truth": all_ce,
      "sampling_seed": sampling_seed,
      "n_restart": n_restart,
      "verbose": verbose,
      "causal_prior": use_causal_prior,
      "hp_prior": flags_config.use_hp_prior,
      # Noisy observations
      "noisy_observations": noisy_observations,
      "noisy_acquisition": flags_config.noisy_acquisition,
      "n_samples_per_intervention": flags_config.n_samples_per_intervention,
      "fix_likelihood_noise_var": flags_config.fix_likelihood_noise_var
  }

  if model_name == "cbo":
    model = cbo.CBO(**input_params)
  elif model_name == "random":
    model = random.Random(**input_params)
  else:
    # Add constraints
    input_params["ground_truth_constraints"] = constraints_values
    input_params["constraints"] = constraints
    input_params["multi_task_model"] = is_multi_task
    input_params["use_prior_mean"] = use_prior_mean
    input_params["update_scm"] = update_scm
    input_params["add_rbf_kernel"] = add_rbf_kernel

    if model_name == "ccbo_dag_multi_task":
      # Monte Carlo construction of the kernel
      input_params["n_kernel_samples"] = flags_config.n_kernel_samples

    model = ccbo.CCBO(**input_params)

  # Run method
  model.run()

  # If model is not constrained compute feasibility after running it
  if model_name in ["random", "cbo"]:
    (constraints_dict, _, _, _,
     _) = constraints_functions.get_constraints_dicts(
         flags_config.exploration_sets, constraints, graph,
         model.target_variable, d_o)
    for i, v in enumerate(model.best_es_over_trials_list):
      if len(v) > 1:
        value = model.best_level_over_trials_list[i].tolist()[0]
      else:
        value = model.best_level_over_trials_list[i]
      is_feasible, _, _, _ = constraints_functions.verify_feasibility(
          optimal_unconstrained_set=v,
          optimal_level=value,
          exploration_sets=flags_config.exploration_sets,
          all_ce=all_ce,
          constraints=constraints,
          constraints_dict=constraints_dict,
          scm_funcs=scm.scm_funcs,
          graph=graph,
          dict_variables=scm.variables,
          interventional_grids=model.interventional_grids,
          n_samples=flags_config.n_samples_ground_truth,
          sampling_seed=sampling_seed,
          task=model.task,
      )
      model.best_es_over_trials_list[i] = [v, int(is_feasible)]

  logging.info(
      "The optimal intervention set found by the algorithm is %s",
      model.best_es_over_trials_list[-1][0],
  )

  logging.info(
      "The optimal target effect value found by the algorithm is %s",
      model.optimal_outcome_values_during_trials[-1],
  )

if __name__ == "__main__":
  app.run(main)
