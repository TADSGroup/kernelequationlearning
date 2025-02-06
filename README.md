# Equation Learning with kernels

The general structure of the project looks like this:
```
KEQL
├── README.md
├── final_examples
│   ├── Burgers
│   │    ├── benchmark_fixedICs
│   │    ├── benchmark_varyICs
│   │    ├── oneshot_onlybdry
│   │    └── oneshot_shock
│   ├── Darcy
│   │    ├── in_distribution
│   │    ├── in_sample
│   │    ├── operator_learning
│   │    └── out_distribution
│   │
│   └── ODE
│        └── duffing.ipynb
└── keql_tools
    ├── BurgerSolver.py
    ├── darcySolver.py
    ├── EquationModel.py
    ├── HessAnalyze.py
    ├── Kernels.py
    ├── KernelTools.py
    ├── Optimizers.py
    └── setup.py
```

Take a look at `making_env.txt`` to see how to install `keql_tools` and run the experiments.