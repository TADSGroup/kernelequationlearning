# Equation Learning with kernels

The structure of the project looks like this:
```
KEQL
├── README.md
├── examples
│   ├── Burgers
│   │    └── BurgersIBC_tuned.ipynb
│   ├── darcy
│   │    ├── Darcy_constantRHS.ipynb
│   │    ├── Darcy_multipleprescribed.ipynb
│   │    └── Darcy_oneprescribed.ipynb
│   ├── heat
│   │    ├── HeatEquationIBC.ipynb
│   │    └── HeatEquationPoly.ipynb
│   └── transport
│        ├── TransportEqnPoly.ipynb
│        ├── TransportEqnRBF.ipynb
│        └── TransportEqnLinearKernel.ipynb
└── keql_tools
    ├── BurgerSolver.py
    ├── darcySolver.py
    ├── EquationModel.py
    ├── HessAnalyze.py
    ├── Kernels.py
    ├── KernelTools.py
    ├── LM_Solve.py
    └── setup.py
```

To run every `.py` script or `.ipynb` notebook we need to add to `sys.path` the path to the `source` folder which contains all the general functions. This is done by typing

```
import sys
sys.path.append('/myfullpath/KEQL/source')
```