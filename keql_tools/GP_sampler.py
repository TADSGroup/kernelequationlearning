import jax 
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jit,grad,hessian,jacfwd,jacrev,vmap
import numpy as np
import matplotlib.pyplot as plt

from importlib import reload

import KernelTools
reload(KernelTools)






