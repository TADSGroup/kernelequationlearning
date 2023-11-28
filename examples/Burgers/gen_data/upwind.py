""" 
This file was built to solve numerically 1D Burgers' equation wave equation with the FFT. The equation corresponds to :

$\dfrac{\partial u}{\partial t} + \mu u\dfrac{\partial u}{\partial x} = \nu \dfrac{\partial^2 u}{\partial x^2}$
 
where
 - u represent the signal
 - x represent the position
 - t represent the time
 - nu and mu are constants to balance the non-linear and diffusion terms.

Copyright - © SACHA BINDER - 2021
"""

############## MODULES IMPORTATION ###############
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt



############## SET-UP THE PROBLEM ###############

mu = 1
nu = 0.01 #kinematic viscosity coefficient
    
#Spatial mesh
L_x = 10 #Range of the domain according to x [m]
dx = 0.1 #Infinitesimal distance
N_x = int(L_x/dx) #Points number of the spatial mesh
X = np.linspace(0,L_x,N_x) #Spatial array

#Temporal mesh
L_t = 8 #Duration of simulation [s]
dt = 0.025  #Infinitesimal time
N_t = int(L_t/dt) #Points number of the temporal mesh
T = np.linspace(0,L_t,N_t) #Temporal array

#Wave number discretization
k = 2*np.pi*np.fft.fftfreq(N_x, d = dx)


#Def of the initial condition 
# 
#
ics = np.load('burgers_ics.npy')

#u0 = np.exp(-(X-3)**2/2) #Single space variable function that represent the wave form at t = 0
# viz_tools.plot_a_frame_1D(X,u0,0,L_x,0,1.2,'Initial condition')

solutions = []
############## EQUATION SOLVING ###############

for i in range(100):
    
    # Load initial condition
    u0 = ics[i,:]
    #Definition of ODE system (PDE ---(FFT)---> ODE system)
    def burg_system(u,t,k,mu,nu):
        #Spatial derivative in the Fourier domain
        u_hat = np.fft.fft(u)
        u_hat_x = 1j*k*u_hat
        u_hat_xx = -k**2*u_hat
        
        #Switching in the spatial domain
        u_x = np.fft.ifft(u_hat_x)
        u_xx = np.fft.ifft(u_hat_xx)
        
        #ODE resolution
        u_t = -mu*u*u_x + nu*u_xx
        return u_t.real
        

    #PDE resolution (ODE system resolution)
    u = odeint(burg_system, u0, T, args=(k,mu,nu,), mxstep=5000).T
    
    solutions.append(u)


############## PLOT ###############


t, x = np.meshgrid(T, X)
fig, axs = plt.subplots(nrows=10, ncols=10, sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.01, wspace=0.01)
fig.suptitle('Different solutions for viscous Burgers equation \n 0 <t <8 (dt = 0.025) 0<x<10, (dx=0.1),  m = 100 with N = 32000')
axs = axs.ravel()

for i in range(100):
    axs[i].contourf(t, x, solutions[i], levels=100, cmap='viridis')
    axs[i].set_yticklabels([])
    axs[i].set_xticklabels([])
    axs[i].set_yticks([])
    axs[i].set_xticks([])

plt.savefig('burgers_viscous_solution_contour.png', dpi = 250)
    

solutions = np.array(solutions)
np.save('sols_burgers.npy', solutions)

# plt.figure(figsize=(10, 6))
# plt.contourf(t, x, solutions[6], levels=100, cmap='viridis')
# plt.colorbar()
# plt.xlabel('Spatial coordinate (x)')
# plt.ylabel('Time step (n)')
# plt.title('Contour Plot of Viscous Burgers\' Equation Solution')
# plt.savefig('burgers_viscous_solution_contour.png')
# plt.close()