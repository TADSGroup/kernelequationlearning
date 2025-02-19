# Kernel Equation Learning: KEqL

JAX implementation of 

> 'Data-Efficient Kernel Methods for Learning 
Differential Equations and Their Solution Operators: 
Algorithms and Error Analysis' by Jalalian, Osorio, Hsu, Hosseini and Owhadi.



## Conda Environments

Since we run benchmarks using a [PINN-based method](https://github.com/isds-neu/EQDiscovery) that we refer to as PINN-SR that runs on a different version of Python and uses TensorFlow 1.5, we hace separate conda environments for learning differential equations with the different methods.


## Usage 

To reproduce the results obtained in the paper you can check the results for each experiment under 'final_examples/' using the conda environemnt `keql_env`. Results from PINN-SR can be obtained by running the experiments using the conda environment `pinnsr-env`.

In order to use it for a new PDE, refer to an existing example and how the tools under `keql_tools/` are being implemented.



