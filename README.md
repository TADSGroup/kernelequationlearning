# Kernel Equation Learning: KEqL

JAX implementation of 

> 'Data-Efficient Kernel Methods for Learning 
Differential Equations and Their Solution Operators: 
Algorithms and Error Analysis' by Jalalian, Osorio, Hsu, Hosseini and Owhadi. See preprint on arXiv [here](https://arxiv.org/abs/2503.01036).



## Conda Environments

We use two separate conda environments due to compatibility issues between the kernel-based implementation and the neural-based implementation.

- `keql_env`: Main environment to run the kernel methods for equation learning. Additionally, you can install the tools inside `keql_tools` by running the command in the terminal `pip install -e /path/to/keql_tools`.

- `pinnsr_env`: Environment used to run code from the [PINN-based method](https://github.com/isds-neu/EQDiscovery) that we refer to as PINN-SR.


## Usage 

- Reproduce examples: See implementation under `examples/`.

- Learn a new PDE: Use source code implemented in `keql_tools/`.