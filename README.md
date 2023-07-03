# ccbo

This project contains the code associated to the following paper:

"Constrained Causal Bayesian Optimization" by Aglietti Virginia, Alan Malek, Ira Ktena, and Silvia Chiappa.
International Conference on Machine Learning. PMLR, 2023.

## Installation
The code requires `python3.10` and `python3.10-dev`.

To install the package and the necessary requirements you can run (run these
commands from the directory that you wish to clone `ccbo` into):

```shell
git clone https://github.com/deepmind/ccbo.git
python3.10 -m venv ccbo_venv
source ccbo_venv/bin/activate
python3.10 -m pip install --upgrade pip
pip install -r ./ccbo/requirements.txt
```

## Usage

The algorithm can be run via the script `run_experiment.py` using the command
`python -m ccbo.experiments.run_optimization` where a config file can be
specified using the flag --config. A notebook `run_experiment.ipynb` is
also provided to allow comparing cCBO and the other methods in the paper.

Experiment configurations are provided in `experiments/`.

## Citing this work

Please cite the ICML paper referenced above. The BibTex is:

    @inproceedings{aglietti2023constrained
        title={Constrained Causal Bayesian Optimization},
        author={Aglietti, Virginia and Malek, Alan and Ktena, Ira and Chiappa, Silvia},
        booktitle={International Conference on Machine Learning},
        year={2023},
    }

## License and disclaimer

Copyright 2023 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
