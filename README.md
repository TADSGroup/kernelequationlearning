# Kernel Equation Learning: KEqL

JAX implementation of 

> 'Data-Efficient Kernel Methods for Learning 
Differential Equations and Their Solution Operators: 
Algorithms and Error Analysis' by Jalalian, Osorio, Hsu, Hosseini and Owhadi.



## Conda Environments

We use two conda environments since the compatibility of the kernel implementation and the neural based implementation are different. 

- `keql_env`: Main environment to run the kernel methods for equation learning.

- `pinnsr_env`: Environment used to run code from the [PINN-based method](https://github.com/isds-neu/EQDiscovery) that we refer to as PINN-SR.


## Usage 

- Reproduce examples: See implementation under `final_examples/`.

- Learn a new PDE: Use source code implemented in `keql_tools/`.



