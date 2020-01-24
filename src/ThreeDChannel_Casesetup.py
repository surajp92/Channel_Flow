# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 20:39:37 2019

@author: suraj
"""

import numpy as np

#-----------------------------------------------------------------------------#
# Channel Flow Solver Case Setup
#-----------------------------------------------------------------------------#

# Select run mode and options
# runmode = 0: Initialization from fixed Re-number (laminar)
# runmode = 1: Re-initialize results from runmode=0 with new grid and
# a randomized disturbance
# runmode = 2: Re-initialize results from runmode=1 onto new, interpolated
# grid


runmode = 2 
retain = 1 # Select whether to save results 
resume = 1 # Select whether to load results
retainoperators = 0 # Select whether to save differential operators
resumeoperators = 1 # Select whether to load differential operators
interpolatenew = 0 # Select whether to interpolate results onto a new grid
                    # This requires runmode = 2

# Select time integration method order (TIM = 1, 2, 3, 4)
# Note that turbulence simulation is not feasible with explicit Euler (1)
TIM = 4 

#ThreeDChannel_Butchertableau;

#-----------------------------------------------------------------------------#
# Simulation duration and time step options
#-----------------------------------------------------------------------------#

nsteps = 1E3 # Provide number of time steps
tstepopt = 1 # Time step option: 0 = fixed, 1 = dynamic
setdt = 4.501E-3 # Fixed time step (requires tstepopt = 0)
CoTarget = 0.5 # Provide target for maximum local Courant number
                # (requires tstepopt = 1)
interval = 5 # Provide output interval
timing = 1 # Select whether the run is timed or not
statinit = 1 # Time step number for statistics initalization

#-----------------------------------------------------------------------------#
# Linear equation solvers: preconditioned biconjugate gradient (BICG),  
# preconditioned conjugate gradient (PCG)
#-----------------------------------------------------------------------------#
 
sbicg = 1
spcg = 0

#-----------------------------------------------------------------------------#
# Set parameters for linear equation solvers. Maximum iterations, diagonal
# compensation, residual tolerance
#-----------------------------------------------------------------------------#

bicgmaxit = 1E6; bicgtol = 1E-3; # BICG setup
pcgmaxit = 300; pcgtol = 1E-3; pcgdiagcomp = 3E-1; # PCG setup
pscheme = 2; # Select order of pressure differentiation scheme for
             # prediction of intermediary pressures in time integration
             
# Nominal setup:
res = 72 # Resolution; must be divisible by 2
Wscale = 1/2 # Geometry span scale 
Lscale = 1/2 # Geometry length scale
N1 = round(2*res*Wscale) # Span
N2 = round(2*res*Lscale) # Length
N3 = res+2 # Wall-normal
ctanh = 5E-2 # tanh-condensing factor, smaller = uniformer

# Geometry based on study by Moser et al.
Length = 4*np.pi*Lscale; Width = Wscale*2*np.pi; Height = 2;

Retau = 180 # Set target shear Reynolds number Re_tau
nu = 1.9E-3 # Kinematic viscosity
ReD = Retau*17.5 # Channel flow (bulk) Reynolds number

gx = 2*(2*Retau*nu)**2/(Height**3) # From shear-bodyforce-equilibrium
gy = 0
gz = 0

Dhyd = Height/2 # Length scale for ReD
Unom = ReD*nu/Dhyd # Starting velocity
Uscale = 1.2*Unom
utnom = np.sqrt(0.5*Height*gx)

if runmode == 0:
    UF = 0
elif runmode == 1:
    UF = 0.4 # Relative fluctuation magnitude for turbulence generation
elif runmode == 2:
    UF = 0


dt = CoTarget*(Length/N2)/Uscale # initial time step
parray = np.zeros((N1*N2*(N3-2),pscheme+1))

#-----------------------------------------------------------------------------#
# Plot settings
#-----------------------------------------------------------------------------#

plo11 = 1 # Velocity and fluctuation profiles
plo12 = 1 # Velocity contours
plo13 = 1 # 3D velocity cutout
plo14 = 1 # 3D Q-criterion isosurface

yplane = 8 # Wall-normal plane for sampling
USC = 1.2 # Set velocity scale for plots
Qset = 15 # Set Q isosurface value


#-----------------------------------------------------------------------------#
# Movie settings
#-----------------------------------------------------------------------------#

recmovie = 0 # Select whether movie is recorded 
nframes = 8 # Select number of frames for recording
frame = 1 # do not change this parameter 
fr = 10 # Video framerate

#-----------------------------------------------------------------------------#
# Load the MKM Reference Data
#-----------------------------------------------------------------------------#
MSRmean = np.loadtxt('chan180.means');
MSRy = MSRmean[:,0].reshape(-1,1);
MSRyplus = MSRmean[:,1];
MSRUmean = MSRmean[:,2];
MSRdUmeandy = MSRmean[:,3];
MSRWmean = MSRmean[:,4];
MSRdWmeandy = MSRmean[:,5];
MSRPmean = MSRmean[:,6];
MSRfluc = np.loadtxt('chan180.reystress');
MSRflucyplus = MSRfluc[:,1];
MSRflucuplus = (MSRfluc[:,2])**0.5;
MSRflucvplus = (MSRfluc[:,3])**0.5;
MSRflucwplus = (MSRfluc[:,4])**0.5;
MSRflucuvplus = MSRfluc[:,5];

















