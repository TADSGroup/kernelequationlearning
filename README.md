# Equation Learning with Kernels

The structure of the project looks like this:
```
KEQL
├── README.md
├── examples
│   ├── Pendulum
│   │    ├── data
│   │    ├── pendulum_one_step.ipynb
│   │    └── pendulum_two_step.ipynb
│   ├── darcy
│   │    ├── data
│   │    └── darcy_two_step.ipynb
└── source
    ├── data_loader.py
    ├── kernels.py
    ├── models.py
    ├── optimization.py
    ├── parameter_learning.py
    └── plotlib.py
```

To run every `.py` script or `.ipynb` notebook we need to add to `sys.path` the path to the `source` folder which contains all the general functions. This is done by typing

```
import sys
sys.path.append('/myfullpath/KEQL/source')
```