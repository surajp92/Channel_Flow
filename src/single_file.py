# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 18:04:14 2019

@author: suraj
"""

import numpy as np
from scipy.sparse import spdiags
import scipy.sparse as sp
import scipy.io
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import bicgstab
from scipy.sparse.linalg import spilu
import time
from numba import jit

import matplotlib.pyplot as plt

#%%

############### ThreeDChannel_Casesetup function ##############################

#-----------------------------------------------------------------------------#
# Channel Flow Solver Case Setup
#-----------------------------------------------------------------------------#

# Select run mode and options
# runmode = 0: Initialization from fixed Re-number (laminar)
# runmode = 1: Re-initialize results from runmode=0 with new grid and
# a randomized disturbance
# runmode = 2: Re-initialize results from runmode=1 onto new, interpolated
# grid

runmode = 0
retain = 1 # Select whether to save results 
resume = 0 # Select whether to load results
retainoperators = 1 # Select whether to save differential operators
resumeoperators = 0 # Select whether to load differential operators
interpolatenew = 0 # Select whether to interpolate results onto a new grid
                    # This requires runmode = 2

# Select time integration method order (TIM = 1, 2, 3, 4)
# Note that turbulence simulation is not feasible with explicit Euler (1)
TIM = 4 

############### ThreeDChannel_Butchertableau function #########################

# Input data from relevant Butcher tableau (see literature for more explicit options)

# explicit Euler
if TIM == 1:
    s = 1
    a = 0
    b = 1
    c = 0

# RK2
if TIM == 2:
    s = 2
    a = np.zeros((s,s))
    a[1,0] = 1
    b = np.array([1/2, 1/2])
    c = np.array([0, 1])

# RK3
if TIM == 3:
    s = 3
    a = np.zeros((s,s))
    a[1,0] = 1/2; a[2,0] = -1; a[2,1] = 2;
    b = np.array([1/6, 2/3, 1/6])
    c = np.array([0, 1/2, 1])
    
# RK4
if TIM == 4:
    s = 4
    a = np.zeros((s,s))
    a[1,0] = 1/2; a[2,1] = 1/2; a[3,2] = 1;
    b = np.array([1/6, 1/3, 1/3, 1/6])
    c = np.array([0, 1/2, 1/2, 1])    
    
#-----------------------------------------------------------------------------#
# Simulation duration and time step options
#-----------------------------------------------------------------------------#

nsteps = 5 # Provide number of time steps
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
 
sbicg = 1 # for bicgstab
spcg = 0  # for preconditioned Conjugate gradient

#-----------------------------------------------------------------------------#
# Set parameters for linear equation solvers. Maximum iterations, diagonal
# compensation, residual tolerance
#-----------------------------------------------------------------------------#

bicgmaxit = 1E6; bicgtol = 1E-3; # BICG setup
pcgmaxit = 300; pcgtol = 1E-3; pcgdiagcomp = 3E-1; # PCG setup
pscheme = 2; # Select order of pressure differentiation scheme for
             # prediction of intermediary pressures in time integration
             
# Nominal setup:
res = 4 # Resolution; must be divisible by 2
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
    UF = 0  # laminar flow
elif runmode == 1:
    UF = 0.4 # Relative fluctuation magnitude for turbulence generation
elif runmode == 2:
    UF = 0


dt = CoTarget*(Length/N2)/Uscale # initial time step
parray = np.zeros((N1*N2*(N3-2),pscheme+1))

#-----------------------------------------------------------------------------#
# Plot settings (to be done)
#-----------------------------------------------------------------------------#

plo11 = 1 # Velocity and fluctuation profiles
plo12 = 1 # Velocity contours
plo13 = 1 # 3D velocity cutout
plo14 = 1 # 3D Q-criterion isosurface

yplane = 8 # Wall-normal plane for sampling
USC = 1.2 # Set velocity scale for plots
Qset = 15 # Set Q isosurface value


#-----------------------------------------------------------------------------#
# Movie settings (to be done)
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

#%%
############### ThreeDChannel_CreateFields function ###########################

#-----------------------------------------------------------------------------#
# Initialize enumeration matrix
#-----------------------------------------------------------------------------#

A = np.zeros((N1*N2*N3,1))
SA = np.shape(A)

#for aa in range(SA[0]):
#    A[aa] = aa

pp = 0
A = np.zeros((N1,N2,N3),dtype=int)

for k in range(N3):
    for j in range(N2):
        for i in range(N1):
            A[i,j,k] = int(pp)
            pp = pp +1

#A = np.reshape(A,[N1,N2,N3])

dx = Length/N2; dy = Width/N1; dz = Height/(N3-2);

# define x and y as homogeneous and fit for periodicity
x = np.linspace(dx,Length,N2); x = x-dx/2;
y = np.linspace(dy,Width,N1); y = y-dy/2;

FX = np.zeros((N1,N2,N3)); FX[:,:,:] = dx; # 1D "Cell size"
FY = np.zeros((N1,N2,N3)); FY[:,:,:] = dy; # 1D "Cell size"

#-----------------------------------------------------------------------------#
# Define wall-normal points via a hyperbolic tangent function
#-----------------------------------------------------------------------------#

fz = np.linspace(-(N3/2-1),(N3/2-1),N3-1);
fz = np.tanh(ctanh*fz);
fz = fz-fz[0];

z = np.zeros((N3))
z[0] = -(fz[1]-fz[0])*0.5;

for p in range(1,N3-1):
    z[p] = fz[p-1] + 0.5*(fz[p] - fz[p-1])

z[N3-1] = fz[N3-2] + 0.5*(fz[N3-2] - fz[N3-3])

z = z/fz[N3-2]*Height; fz = fz/fz[N3-2]*Height;

FZ = np.zeros((N1,N2,N3))
for p in range(1,N3-1):
    FZ[:,:,p] = fz[p]-fz[p-1]

FZ[:,:,0] = FZ[:,:,1]; FZ[:,:,N3-1] = FZ[:,:,N3-2];

[X,Y,Z] = np.meshgrid(x,y,z);

X = X[:,:,1:N3-1]
Y = Y[:,:,1:N3-1]
Z = Z[:,:,1:N3-1]

# x index notation
inx = np.linspace(0,N2-1,N2,dtype = int); inx = inx.T;
east = inx+1; west = inx-1;

# y index notation
iny = np.linspace(0,N1-1,N1,dtype = int); iny = iny.T;
north = iny+1; south = iny-1;

# Make a different index notation for the z-direction (wall bounded)
# The current methodology uses a staggered grid with a ghost cell
# within the wall
inz = np.linspace(1,N3-2,N3-2,dtype = int); inz = inz.T; 
air = inz + 1; ground = inz - 1;

# Assign indices for periodic boundaries
east[N2-1] = 0; west[0] = N2-1; north[N1-1] = 0; south[0] = N1-1;

#-----------------------------------------------------------------------------#
# Channel flow initialization
#-----------------------------------------------------------------------------#
U = np.zeros((N1,N2,N3-2))
V = np.zeros((N1,N2,N3-2))
W = np.zeros((N1,N2,N3-2))

if runmode == 0:
    U[:,:,:] = Unom

if runmode == 1:
    data =  np.load('field0.npz') #add variables saved from runmode = 0
    U = data['U']
    V = data['V']
    W = data['W']
    UF1 = UF*(np.random.rand(N1,N2,N3-2) - 0.5)
    UF2 = UF*(np.random.rand(N1,N2,N3-2) - 0.5)
    UF3 = UF*(np.random.rand(N1,N2,N3-2) - 0.5)
    U = U + UF1*np.max(U)
    V = V + UF2*np.max(U)
    W = W + UF3*np.max(U)
    if resume == 1:
        print('Field1 load script to be added resume')
        # load('Field1') add saved variables for runmode = 1

if runmode == 2:
    data =  np.load('field1.npz') #add variables saved from runmode = 0
    U = data['U']
    V = data['V']
    W = data['W']
    #print('Field1 load script to be added')
    if resume == 1:
        data =  np.load('field2.npz') #add variables saved from runmode = 0
        U = data['U']
        V = data['V']
        W = data['W']
        #print('Field2 load script to be added resume')
        
        #??? interpolate script to be added

#-----------------------------------------------------------------------------#
# Matrix numbering reference for discrete differential operators
#-----------------------------------------------------------------------------#
A0 = A[:,:,inz[0]:inz[-1]+1] - N1*N2
AN = A[north,inx[0]:inx[-1]+1,inz[0]:inz[-1]+1] - N1*N2
AS = A[south,inx[0]:inx[-1]+1,inz[0]:inz[-1]+1] - N1*N2
AE = A[iny[0]:iny[-1]+1,east,inz[0]:inz[-1]+1] - N1*N2
AW = A[iny[0]:iny[-1]+1,west,inz[0]:inz[-1]+1] - N1*N2
AA = A[iny[0]:iny[-1]+1,inx[0]:inx[-1]+1,air] - N1*N2
AG = A[iny[0]:iny[-1]+1,inx[0]:inx[-1]+1,ground] - N1*N2

u = np.reshape(U,[-1,1],order='F')
v = np.reshape(V,[-1,1],order='F')
w = np.reshape(W,[-1,1],order='F')

pold = np.zeros((N1*N2*(N3-2),1))

#%% 

############### ThreeDChannel_Differentiate1 function ##########################
#@jit
def ThreeDChannel_Differentiate1(N1,N2,N3,FX,FY,FZ,
                                 inx,iny,inz,A0,AN,AS,AE,AW,AA,AG,east,
                                 west,north,south,air,ground,inb):
    
    data = np.zeros((7,(N1*N2*(N3-2))))
    diags = np.array([-N1*(N3-2), -N1, -1, 0, 1, N1, N1*(N3-2)])
    M = spdiags(data, diags, N1*N2*(N3-2), N1*N2*(N3-2))
    #M = sp.csr_matrix(M)
    M = sp.lil_matrix(M)
        
    for k in inz-1:
        for j in inx:
            for i in iny:
                FY0 = FY[i,j,k+1]
                FYN = FY[north[i],j,k+1]
                FYS = FY[south[i],j,k+1]
                
                FX0 = FX[i,j,k+1]
                FXE = FX[i,east[j],k+1]
                FXW = FX[i,west[j],k+1]
                
                FZ0 = FZ[i,j,k+1]
                FZA = FZ[i,j,air[k]]
                FZG = FZ[i,j,ground[k]]
                
                if inb == 1:
                    M[A0[i,j,k],A0[i,j,k]] = 1/FY0*(FYN/(FY0+FYN)-FYS/(FYS+FY0))
                
                if inb == 2:
                    M[A0[i,j,k],A0[i,j,k]] = 1/FX0*(FXE/(FX0+FXE)-FXW/(FXW+FX0))
                
                if inb == 3:
                    M[A0[i,j,k],A0[i,j,k]] = 1/FZ0*(FZA/(FZ0+FZA)-FZG/(FZG+FZ0))
                    
                if inb == 1:
                    M[A0[i,j,k],AN.ravel(order='F')[A0[i,j,k]]] = 1/FY0*FY0/(FY0+FYN)
                    M[A0[i,j,k],AS.ravel(order='F')[A0[i,j,k]]] = -1/FY0*FY0/(FY0+FYS)
                
                if inb == 2:
                    M[A0[i,j,k],AE.ravel(order='F')[A0[i,j,k]]] = 1/FX0*FX0/(FX0+FXE)
                    M[A0[i,j,k],AW.ravel(order='F')[A0[i,j,k]]] = -1/FX0*FX0/(FX0+FXW)
                
                # Account for wall Dirichlet (no-slip) condition in the wall-normal direction
                if inb == 3:
                    if AG.ravel(order='F')[A0[i,j,k]] >= 0:
                        M[A0[i,j,k],AG.ravel(order='F')[A0[i,j,k]]] = -1/FZ0*FZ0/(FZ0+FZG)
                    else:
                        M[A0[i,j,k],A0[i,j,k]] = 1/FZ0*(FZA/(FZ0+FZA)-FZG/(FZG+FZ0) + \
                                                 FZ0/(FZG+FZ0))
                    
                    if AA.ravel(order='F')[A0[i,j,k]] < N1*N2*(N3-2):
                        M[A0[i,j,k],AA.ravel(order='F')[A0[i,j,k]]] = 1/FZ0*FZ0/(FZ0+FZA)
                    
                    else:
                        M[A0[i,j,k],A0[i,j,k]] = 1/FZ0*(FZA/(FZ0+FZA)-FZG/(FZG+FZ0) - \
                                                 FZ0/(FZA+FZ0))
    
    M = sp.csc_matrix(M, copy=False)
                   
    return M

#%% 

############## ThreeDChannel_Differentiate2 function ##########################
#@jit    
def ThreeDChannel_Differentiate2(N1,N2,N3,FX,FY,FZ,
                                 inx,iny,inz,A0,AN,AS,AE,AW,AA,AG,east,
                                 west,north,south,air,ground,inb):

    data = np.zeros((7,(N1*N2*(N3-2))))
    diags = np.array([-N1*(N3-2), -N1, -1, 0, 1, N1, N1*(N3-2)])
    M = spdiags(data, diags, N1*N2*(N3-2), N1*N2*(N3-2))
    #M = sp.csr_matrix(M)
    M = sp.lil_matrix(M)
    
    for k in inz-1:
        for j in inx:
            for i in iny:
                FY0 = FY[i,j,k+1]
                FYN = FY[north[i],j,k+1]
                FYS = FY[south[i],j,k+1]
                
                FX0 = FX[i,j,k+1]
                FXE = FX[i,east[j],k+1]
                FXW = FX[i,west[j],k+1]
                
                FZ0 = FZ[i,j,k+1]
                FZA = FZ[i,j,air[k]]
                FZG = FZ[i,j,ground[k]]
                
                if inb == 1:
                    M[A0[i,j,k],A0[i,j,k]] = -2/(FY0*(FY0+FYN)) - 2/(FY0*(FY0+FYS))
                
                if inb == 2:
                    M[A0[i,j,k],A0[i,j,k]] = -2/(FX0*(FX0+FXE)) - 2/(FX0*(FX0+FXW))
                
                if inb == 3:
                    M[A0[i,j,k],A0[i,j,k]] = -2/(FZ0*(FZ0+FZA)) - 2/(FZ0*(FZ0+FZG))
                
                if inb == 1:
                    M[A0[i,j,k],AN.ravel(order='F')[A0[i,j,k]]] = 2/(FY0*(FY0+FYN))
                    M[A0[i,j,k],AS.ravel(order='F')[A0[i,j,k]]] = 2/(FY0*(FY0+FYS))
                
                if inb == 2:
                    M[A0[i,j,k],AE.ravel(order='F')[A0[i,j,k]]] = 2/(FX0*(FX0+FXE))
                    M[A0[i,j,k],AW.ravel(order='F')[A0[i,j,k]]] = 2/(FX0*(FX0+FXW))
                
                # Account for wall Dirichlet (no-slip) condition in the wall-normal direction
                if inb == 3:
                    if AG.ravel(order='F')[A0[i,j,k]] >= 0:
                        M[A0[i,j,k],AG.ravel(order='F')[A0[i,j,k]]] = 2/(FZ0*(FZ0+FZG)) 
                    else:
                        M[A0[i,j,k],A0[i,j,k]] = -2/(FZ0*(FZ0+FZA)) - 4/(FZ0*(FZ0+FZG))
                        
                    if AA.ravel(order='F')[A0[i,j,k]] < N1*N2*(N3-2):
                        M[A0[i,j,k],AA.ravel(order='F')[A0[i,j,k]]] = 2/(FZ0*(FZ0+FZA))
                    else:
                        M[A0[i,j,k],A0[i,j,k]] = -4/(FZ0*(FZ0+FZA)) -2/(FZ0*(FZ0+FZG))
    
    M = sp.csc_matrix(M, copy=False)
                    
    return M             
                
#%% 

########### ThreeDChannel_Differentiate1p function ############################
#@jit
def ThreeDChannel_Differentiate1p(N1,N2,N3,FX,FY,FZ,
                                 inx,iny,inz,A0,AN,AS,AE,AW,AA,AG,east,
                                 west,north,south,air,ground,inb):

    data = np.zeros((7,(N1*N2*(N3-2))))
    diags = np.array([-N1*(N3-2), -N1, -1, 0, 1, N1, N1*(N3-2)])
    M = spdiags(data, diags, N1*N2*(N3-2), N1*N2*(N3-2))
    #M = sp.csr_matrix(M)
    M = sp.lil_matrix(M)
    
    for k in inz-1:
        for j in inx:
            for i in iny:
                FY0 = FY[i,j,k+1]
                FYN = FY[north[i],j,k+1]
                FYS = FY[south[i],j,k+1]
                
                FX0 = FX[i,j,k+1]
                FXE = FX[i,east[j],k+1]
                FXW = FX[i,west[j],k+1]
                
                FZ0 = FZ[i,j,k+1]
                FZA = FZ[i,j,air[k]]
                FZG = FZ[i,j,ground[k]]
                
                if inb == 1:
                    M[A0[i,j,k],A0[i,j,k]] = 1/FY0*(FYN/(FY0+FYN)-FYS/(FY0+FYS))
                
                if inb == 2:
                    M[A0[i,j,k],A0[i,j,k]] = 1/FX0*(FXE/(FX0+FXE)-FXW/(FX0+FXW))
                    
                if inb == 3:
                    M[A0[i,j,k],A0[i,j,k]] = 1/FZ0*(FZA/(FZ0+FZA)-FZG/(FZ0+FZG))
                    
                if inb == 1:
                    M[A0[i,j,k],AN.ravel(order='F')[A0[i,j,k]]] = 1/FY0*(FY0/(FY0+FYN))
                    M[A0[i,j,k],AS.ravel(order='F')[A0[i,j,k]]] = -1/FY0*(FY0/(FY0+FYS))
                
                if inb == 2:
                    M[A0[i,j,k],AE.ravel(order='F')[A0[i,j,k]]] = 1/FX0*(FX0/(FX0+FXE))
                    M[A0[i,j,k],AW.ravel(order='F')[A0[i,j,k]]] = -1/FX0*(FX0/(FX0+FXW))
                
                # Account for Neumann (zero-gradient) condition in the wall-normal direction
                
                if inb == 3:
                    if AG.ravel(order='F')[A0[i,j,k]] >= 0:
                        M[A0[i,j,k],AG.ravel(order='F')[A0[i,j,k]]] = -1/FZ0*(FZ0/(FZ0+FZG))
                    else:
                        M[A0[i,j,k],A0[i,j,k]] = 1/FZ0*(FZA/(FZ0+FZA)-FZG/(FZ0+FZG) - \
                                                 FZ0/(FZ0+FZG))
                    
                    if AA.ravel(order='F')[A0[i,j,k]] < N1*N2*(N3-2):
                        M[A0[i,j,k],AA.ravel(order='F')[A0[i,j,k]]] = 1/FZ0*(FZ0/(FZ0+FZA))
                    else:
                        M[A0[i,j,k],A0[i,j,k]] = 1/FZ0*(FZA/(FZ0+FZA)-FZG/(FZ0+FZG) + \
                                                 FZ0/(FZ0+FZA))
    
    M = sp.csc_matrix(M, copy=False)
            
    return M

#%% 

############# ThreeDChannel_CreatePoissonMatrix function ######################
#@jit
def ThreeDChannel_CreatePoissonMatrix(N1,N2,N3,FX,FY,FZ,
                                 inx,iny,inz,A0,AN,AS,AE,AW,AA,AG,east,
                                 west,north,south,air,ground):
    
    data = np.zeros((7,(N1*N2*(N3-2))))
    diags = np.array([-N1*(N3-2), -N1, -1, 0, 1, N1, N1*(N3-2)])
    M = spdiags(data, diags, N1*N2*(N3-2), N1*N2*(N3-2))
    #M = sp.csr_matrix(M)
    M = sp.lil_matrix(M)
        
    for k in inz-1:
        for j in inx:
            for i in iny:
                FY0 = FY[i,j,k+1]
                FYN = FY[north[i],j,k+1]
                FYS = FY[south[i],j,k+1]
                
                FX0 = FX[i,j,k+1]
                FXE = FX[i,east[j],k+1]
                FXW = FX[i,west[j],k+1]
                
                FZ0 = FZ[i,j,k+1]
                FZA = FZ[i,j,air[k]]
                FZG = FZ[i,j,ground[k]]
                
                M[A0[i,j,k],A0[i,j,k]] = -2/(FY0*(FY0+FYN)) - 2/(FY0*(FY0+FYS)) - \
                                          2/(FX0*(FX0+FXE)) - 2/(FX0*(FX0+FXW)) - \
                                          2/(FZ0*(FZ0+FZA)) - 2/(FZ0*(FZ0+FZG))
                
                M[A0[i,j,k],AN.ravel(order='F')[A0[i,j,k]]] = 2/(FY0*(FY0+FYN))
                
                M[A0[i,j,k],AS.ravel(order='F')[A0[i,j,k]]] = 2/(FY0*(FY0+FYS))
                
                M[A0[i,j,k],AE.ravel(order='F')[A0[i,j,k]]] = 2/(FX0*(FX0+FXE))
                
                M[A0[i,j,k],AW.ravel(order='F')[A0[i,j,k]]] = 2/(FX0*(FX0+FXW))
                
                # Implement Neumann wall BC
                if AA.ravel(order='F')[A0[i,j,k]] < N1*N2*(N3-2):
                    M[A0[i,j,k],AA.ravel(order='F')[A0[i,j,k]]] = 2/(FZ0*(FZ0+FZA))
                else:
                    M[A0[i,j,k],A0[i,j,k]] = -2/(FY0*(FY0+FYN)) -2/(FY0*(FY0+FYS)) - \
                                              2/(FX0*(FX0+FXE)) -2/(FX0*(FX0+FXW)) - \
                                              2/(FZ0*(FZ0+FZG))
                    
                if AG.ravel(order='F')[A0[i,j,k]] >= 0:
                    M[A0[i,j,k],AG.ravel(order='F')[A0[i,j,k]]] = 2/(FZ0*(FZ0+FZG))
                else:
                    M[A0[i,j,k],A0[i,j,k]] = -2/(FY0*(FY0+FYN)) -2/(FY0*(FY0+FYS)) - \
                                              2/(FX0*(FX0+FXE)) -2/(FX0*(FX0+FXW)) - \
                                              2/(FZ0*(FZ0+FZA))
                
                
    
    i = round(N1/2)-1; j = int(N2/2-1); k = 1-1;
    
    FY0 = FY[i,j,k+1]
    FYN = FY[north[i],j,k+1]
    FYS = FY[south[i],j,k+1]
    
    FX0 = FX[i,j,k+1]
    FXE = FX[i,east[j],k+1]
    FXW = FX[i,west[j],k+1]
    
    FZ0 = FZ[i,j,k+1]
    FZA = FZ[i,j,air[k]]
    FZG = FZ[i,j,ground[k]]
    
    # Fix pressure on a single bottom wall face to zero (Dirichlet cond.)
    M[A0[i,j,k],A0[i,j,k]] = -2/(FY0*(FY0+FYN)) - 2/(FY0*(FY0+FYS)) - \
                              2/(FX0*(FX0+FXE)) - 2/(FX0*(FX0+FXW)) - \
                              2/(FZ0*(FZ0+FZA)) - 4/(FZ0*(FZ0+FZG))
    
    M = sp.csc_matrix(M, copy=False)
    
    return M


#%%
    
############## ThreeDChannel_DifferentialOperators function ###################
    
#-----------------------------------------------------------------------------#
# This section entails the construction of linear differencing operators.
# Importantly, periodic, Dirichlet and Neumann boundary conditions
# have been built into these operators.
#-----------------------------------------------------------------------------#
    
if resumeoperators == 1:
    print('Loading saved differential operators...')
    if runmode == 0:
        data = np.load('Operators0.npz')
    if runmode == 1:
        data = np.load('Operators1.npz')
    if runmode == 2:
        data = np.load('Operators2.npz')
    
    Dx = data['Dx']
    Dy = data['Dy']
    Dz = data['Dz']
    Dxx = data['Dxx']
    Dyy = data['Dyy']
    Dzz = data['Dzz']
    Dxp = data['Dxp']
    Dyp = data['Dyp']
    Dzp = data['Dzp']
    M = data['M']

else:    
    # Generate discrete operators for first derivatives
    Dx = ThreeDChannel_Differentiate1(N1,N2,N3,FX,FY,FZ,
                                     inx,iny,inz,A0,AN,AS,AE,AW,AA,AG,east,
                                     west,north,south,air,ground,2)
    
    Dy = ThreeDChannel_Differentiate1(N1,N2,N3,FX,FY,FZ,
                                     inx,iny,inz,A0,AN,AS,AE,AW,AA,AG,east,
                                     west,north,south,air,ground,1)
    
    Dz = ThreeDChannel_Differentiate1(N1,N2,N3,FX,FY,FZ,
                                     inx,iny,inz,A0,AN,AS,AE,AW,AA,AG,east,
                                     west,north,south,air,ground,3)
    
    # Generate discrete operators for second derivatives
    Dxx = ThreeDChannel_Differentiate2(N1,N2,N3,FX,FY,FZ,
                                     inx,iny,inz,A0,AN,AS,AE,AW,AA,AG,east,
                                     west,north,south,air,ground,2)
    
    Dyy = ThreeDChannel_Differentiate2(N1,N2,N3,FX,FY,FZ,
                                     inx,iny,inz,A0,AN,AS,AE,AW,AA,AG,east,
                                     west,north,south,air,ground,1)
    
    Dzz = ThreeDChannel_Differentiate2(N1,N2,N3,FX,FY,FZ,
                                     inx,iny,inz,A0,AN,AS,AE,AW,AA,AG,east,
                                     west,north,south,air,ground,3)    
    
    # Generate discrete operators for first pressure derivatives
    Dxp = ThreeDChannel_Differentiate1p(N1,N2,N3,FX,FY,FZ,
                                     inx,iny,inz,A0,AN,AS,AE,AW,AA,AG,east,
                                     west,north,south,air,ground,2)
    
    Dyp = ThreeDChannel_Differentiate1p(N1,N2,N3,FX,FY,FZ,
                                     inx,iny,inz,A0,AN,AS,AE,AW,AA,AG,east,
                                     west,north,south,air,ground,1)
    
    Dzp = ThreeDChannel_Differentiate1p(N1,N2,N3,FX,FY,FZ,
                                     inx,iny,inz,A0,AN,AS,AE,AW,AA,AG,east,
                                     west,north,south,air,ground,3)
    
    # Generate Poisson operator
    M = ThreeDChannel_CreatePoissonMatrix(N1,N2,N3,FX,FY,FZ,
                                     inx,iny,inz,A0,AN,AS,AE,AW,AA,AG,east,
                                     west,north,south,air,ground)

    # save operators script to be added
    if retainoperators == 0:
        np.savez('Operators0.npz', Dx=Dx, Dy=Dy, Dz=Dz, Dxx=Dxx, Dyy=Dyy, Dzz=Dzz,
                 Dxp=Dxp, Dyp=Dyp, Dzp=Dzp, M=M)
        
    if retainoperators == 1:
        np.savez('Operators1.npz', Dx=Dx, Dy=Dy, Dz=Dz, Dxx=Dxx, Dyy=Dyy, Dzz=Dzz,
                 Dxp=Dxp, Dyp=Dyp, Dzp=Dzp, M=M)
    
    if retainoperators == 2:
        np.savez('Operators2.npz', Dx=Dx, Dy=Dy, Dz=Dz, Dxx=Dxx, Dyy=Dyy, Dzz=Dzz,
                 Dxp=Dxp, Dyp=Dyp, Dzp=Dzp, M=M)

#%%

################ ThreeDChannel_Preconditioners function #######################
         
#??? understand precondiytiong and ilu, ichol in python

if resumeoperators == 1 and runmode == 2:
    preconditioners = scipy.io.loadmat('Preconditioners2.mat')

    if spcg == 1:
        C = preconditioners.get('C')
        
    if sbicg == 1:
        Lbicg = preconditioners.get('Lbicg')
        Ubicg = preconditioners.get('Ubicg')

else:
    if spcg == 1:
        C = spilu(M) #??? to be changed to incomplete cholesky
        
    if sbicg == 1:
        P = spilu(M)
        Lbicg = P.L
        Ubicg = P.U

if retainoperators == 1:
    if runmode == 0:
        if spcg == 1:
            scipy.io.savemat('Preconditioners0.mat', {'C':C})
        elif sbicg == 1:
            scipy.io.savemat('Preconditioners0.mat', {'Lbicg':Lbicg, 'Ubicg':Ubicg})
    
    if runmode == 1:
        if spcg == 1:
            scipy.io.savemat('Preconditioners1.mat', {'C':C})
        elif sbicg == 1:
            scipy.io.savemat('Preconditioners1.mat', {'Lbicg':Lbicg, 'Ubicg':Ubicg})
    
    if runmode == 2:
        if spcg == 1:
            scipy.io.savemat('Preconditioners2.mat', {'C':C})
        elif sbicg == 1:
            scipy.io.savemat('Preconditioners2.mat', {'Lbicg':Lbicg, 'Ubicg':Ubicg})


#%%

############## ThreeDChannel_Projection function ##############################

def projection(Dx,Dy,Dz,M,u,v,w,p,pold,C,Lbicg,Ubicg):
#-----------------------------------------------------------------------------#
# Compute divergence of velocity field
#-----------------------------------------------------------------------------#
    
    DIV = Dx*u+Dy*v+Dz*w
    
    if spcg == 1:
        C = C.T*C
        [p,flag] = cg(-M,-DIV,pold,pcgtol,pcgmaxit,C)
    
    if sbicg == 1:
        C = Lbicg*Ubicg
        #[p,flag] = bicgstab(M,DIV,pold,bicgtol,bicgmaxit,C)
        [p,flag] = bicgstab(M,DIV,pold,bicgtol,bicgmaxit)
    
    pold = p;
    
    # Compute pressure gradient
    px = Dxp*p;
    py = Dyp*p;
    pz = Dzp*p;
    
    # Correct velocity field (vectors) with pressure gradient
    u = u-px.reshape(-1,1);
    v = v-py.reshape(-1,1);
    w = w-pz.reshape(-1,1);
    
    return u,v,w,pold
    
#%% 

################### Projection step ###########################################
        
#-----------------------------------------------------------------------------#
# Compute divergence of velocity field
#-----------------------------------------------------------------------------#

DIV = Dx*u+Dy*v+Dz*w

if spcg == 1:
    C = C.T*C
    [p,flag] = cg(-M,-DIV,pold,pcgtol,pcgmaxit,C)

if sbicg == 1:
    C = Lbicg*Ubicg
    #[p,flag] = bicgstab(M,DIV,pold,bicgtol,bicgmaxit,C)
    [p,flag] = bicgstab(M,DIV,pold,bicgtol,bicgmaxit)

pold = p;

# Compute pressure gradient
px = Dxp*p;
py = Dyp*p;
pz = Dzp*p;

# Correct velocity field (vectors) with pressure gradient
u = u-px.reshape(-1,1);
v = v-py.reshape(-1,1);
w = w-pz.reshape(-1,1);

#%%

if timing == 1:
    start_time = time.time()

for i in range(nsteps):

############ ThreeDChannel_AdjustTimeStep function ############################
    
#-----------------------------------------------------------------------------#
# Here, Courant numbers in three cartesian directions are sensed, and the
# new time step is adjusted accordingly
#-----------------------------------------------------------------------------#
    
    Cox = U*dt/FX[:,:,1:N3-1]
    Coy = V*dt/FY[:,:,1:N3-1]
    Coz = W*dt/FZ[:,:,1:N3-1]
    
    Co = Cox+Coy+Coz
    Comax = np.max(Co)
    
    if tstepopt == 0:
        dt = setdt
    
    if tstepopt == 1:
        dt = dt/(Comax/CoTarget)

     
############ ThreeDChannel_Solver function ####################################
        
#-----------------------------------------------------------------------------#
# Incompressible Navier-Stokes solver
#-----------------------------------------------------------------------------#
        
    # Convert 3D velocity arrays into column vectors
    u = np.reshape(U,[-1,1],order='F')
    v = np.reshape(V,[-1,1],order='F')
    w = np.reshape(W,[-1,1],order='F')
    
    # Store values from previous time step and create arrays for intermediary storage
    uold = u; uc = u
    vold = v; vc = v
    wold = w; wc = w   
    
    # Define the slope vectors (k)
    uk = np.zeros((N1*N2*(N3-2),s))
    vk = np.zeros((N1*N2*(N3-2),s))
    wk = np.zeros((N1*N2*(N3-2),s))


    for ii in range(s):
    
    # In the RK-loop, Uold is the velocity field obtained in the last iteration. 
    # Uc is the velocity field that is collated for the next time step.
    
    # Compute state ii according to the Runge-Kutta formula
        
        du = np.zeros((N1*N2*(N3-2),1))
        dv = np.zeros((N1*N2*(N3-2),1))
        dw = np.zeros((N1*N2*(N3-2),1))
        
        if ii >=1:
            for jj in range(s):
                # Here, the RK formula is used with the given Butcher tableau
                du = du + a[ii,jj]*uk[:,jj].reshape(-1,1);
                dv = dv + a[ii,jj]*vk[:,jj].reshape(-1,1);
                dw = dw + a[ii,jj]*wk[:,jj].reshape(-1,1);
            
            u = uold + dt*du
            v = vold + dt*dv
            w = wold + dt*dw
            
            # Make pressure field estimate based on previously solved pressures
            
            ################### Projection step ###################################
            u,v,w,pold = projection(Dx,Dy,Dz,M,u,v,w,p,pold,C,Lbicg,Ubicg)
            
        # Begin computing Navier-Stokes contributions
        
        # Convection term (skew-symmetric) - cartesian components
        # Note: Neumann BC operator Dzp is used for velocity product differentiation 
        # across the channel wall
        
        CONVx = 0.5*(Dx*(u*u) + Dy*(v*u) + Dzp*(w*u) + u*(Dx*u) + v*(Dy*u) + w*(Dz*u))
        CONVy = 0.5*(Dx*(u*v) + Dy*(v*v) + Dzp*(w*v) + u*(Dx*v) + v*(Dy*v) + w*(Dz*v))
        CONVz = 0.5*(Dx*(u*w) + Dy*(v*w) + Dzp*(w*w) + u*(Dx*w) + v*(Dy*w) + w*(Dz*w))     
        
        # Diffusion term - cartesian components
        DIFFx = nu*(Dxx*u + Dyy*u + Dzz*u)
        DIFFy = nu*(Dxx*v + Dyy*v + Dzz*v)
        DIFFz = nu*(Dxx*w + Dyy*w + Dzz*w)
        
        # Implementation into momentum equation
        uk[:,ii] = (-CONVx + DIFFx + gx).flatten()
        vk[:,ii] = (-CONVy + DIFFy + gy).flatten()
        wk[:,ii] = (-CONVz + DIFFz + gz).flatten()
                     
        # End computing contributions
        # [du dv dw]^T are the k_i slope
        
        # Contribution of step i is added to the collective contribution
        # via coefficients defined by vector b
        
        uc = uc + dt*b[ii]*uk[:,ii].reshape(-1,1);
        vc = vc + dt*b[ii]*vk[:,ii].reshape(-1,1);
        wc = wc + dt*b[ii]*wk[:,ii].reshape(-1,1);
        
        # on final loop
        if ii == s:
            u = uc; v = vc; w = wc

    # carry out final projection
    u,v,w,pold = projection(Dx,Dy,Dz,M,u,v,w,p,pold,C,Lbicg,Ubicg)
    
    U = np.reshape(u,[N1,N2,N3-2],order='F')
    V = np.reshape(v,[N1,N2,N3-2],order='F')
    W = np.reshape(w,[N1,N2,N3-2],order='F')
    
    Cox = U*dt/FX[:,:,1:N3-1]
    Coy = V*dt/FY[:,:,1:N3-1]
    Coz = W*dt/FZ[:,:,1:N3-1]
    
    Co = Cox+Coy+Coz
    Comax = np.max(Co)
    
    print('n = ', i, 'dt = ', dt, 'Comax = ', Comax)

if timing == 1:
    end_time = time.time()

print('Elapsed time is ', end_time - start_time, ' Seconds')

#%%

if runmode == 0:
    np.savez('field0.npz', U=U, V=V, W=W)

if runmode == 1:
    np.savez('field1.npz', U=U, V=V, W=W)
    
if runmode == 2:
    np.savez('field2.npz', U=U, V=V, W=W)           

#%%

############## ThreeDChannel_Analyze_Fluctuations_Compact step ################

#if i%interval == 0:

# statistical boundary layer analysis
y1 = np.linspace(0,N1-1,N1,dtype = int)
x1 = np.linspace(0,N2-1,N2,dtype = int)
z1 = np.linspace(0,int(N3/2)-2,int(N3/2)-1,dtype = int)

y2 = np.linspace(0,N1-1,N1,dtype = int)
x2 = np.linspace(0,N2-1,N2,dtype = int)
z2 = np.linspace(int(N3/2)-1,N3-3,int(N3/2)-1,dtype = int)


#ut1 = (nu*np.abs(U[y1[0]:y1[-1]+1,x1[0]:x1[-1]+1,0])/Z[y1[0]:y1[-1]+1,x1[0]:x1[-1]+1,0])**(0.5)

ut1 = (nu*np.abs(U[0:N1,0:N2,0])/Z[0:N1,0:N2,0])**(0.5) # Shear velocity (lower)
ut1mean = np.mean(ut1)

ut2 = (nu*np.abs(U[0:N1,0:N2,N3-3])/(Height - Z[0:N1,0:N2,N3-3]))**(0.5) # Shear velocity (upper)
ut2mean = np.mean(ut2)

utmean = 0.5*(ut1mean+ut2mean)

uplu1 = U[0:N1,0:N2,0:int(N3/2)-1]/utmean
uplu1mean = np.mean(uplu1, axis = 0)
uplu1mean = np.mean(uplu1mean, axis = 0)

uplu2 = U[0:N1,0:N2,int(N3/2)-1:N3-2]/utmean
uplu2 = np.flip(uplu2,2)
uplu2mean = np.mean(uplu2, axis = 0)
uplu2mean = np.mean(uplu2mean, axis = 0)

vplu1 = W[0:N1,0:N2,0:int(N3/2)-1]/utmean # Here, V and W change places
vplu1mean = np.mean(vplu1, axis = 0)
vplu1mean = np.mean(vplu1mean, axis = 0)

vplu2 = -W[0:N1,0:N2,int(N3/2)-1:N3-2]/utmean # velocity away from wall
vplu2 = np.flip(vplu2,2)
vplu2mean = np.mean(vplu2, axis = 0)
vplu2mean = np.mean(vplu2mean, axis = 0)

vplu2mean = np.flip(vplu2mean,0)

wplu1 = V[0:N1,0:N2,0:int(N3/2)-1]/utmean # boundary distance
wplu1mean = np.mean(wplu1, axis = 0)
wplu1mean = np.mean(wplu1mean, axis = 0)

wplu2 = V[0:N1,0:N2,int(N3/2)-1:N3-2]/utmean
wplu2 = np.flip(wplu2,2)
wplu2mean = np.mean(wplu2, axis = 0)
wplu2mean = np.mean(wplu2mean, axis = 0)

yplu1 = (Z[0:N1,0:N2,0:int(N3/2)-1]/nu)*utmean
yplu1mean = np.mean(yplu1, axis = 0)
yplu1mean = np.mean(yplu1mean, axis = 0)

yplu2 = ((Height - Z[0:N1,0:N2,int(N3/2)-1:N3-2])/nu)*utmean
yplu2 = np.flip(yplu2,2)
yplu2mean = np.mean(yplu2, axis = 0)
yplu2mean = np.mean(yplu2mean, axis = 0)

uplumean = 0.5*(uplu1mean + uplu2mean)
vplumean = 0.5*(vplu1mean + vplu2mean)
wplumean = 0.5*(wplu1mean + wplu2mean)
yplumean = 0.5*(yplu1mean + yplu2mean)

utmean = np.reshape(utmean,[1,1])
uplumean = np.reshape(uplumean,[int(N3/2)-1,1])
vplumean = np.reshape(vplumean,[int(N3/2)-1,1])
wplumean = np.reshape(wplumean,[int(N3/2)-1,1])
yplumean = np.reshape(yplumean,[int(N3/2)-1,1])

# Begin averaging from 1st interval, and fluctuations from "statinit"
if i == interval:
    uutmean = utmean;
    uuplumean = uplumean;
    vvplumean = vplumean;
    wwplumean = wplumean;
    yyplumean = yplumean;
    taveuplumean = uplumean;
    tavevplumean = vplumean;
    tavewplumean = wplumean;
    taveyplumean = yplumean;

if i > interval:
    uutmean = np.concatenate((uutmean, utmean), axis = 1);
    uuplumean = np.concatenate((uuplumean,uplumean), axis = 1);
    vvplumean = np.concatenate((vvplumean,vplumean), axis = 1);
    wwplumean = np.concatenate((wwplumean,wplumean), axis = 1);
    yyplumean = np.concatenate((yyplumean,yplumean), axis = 1);
    taveuutmean = np.mean(uutmean, axis = 1);
    taveuplumean = np.mean(uuplumean, axis = 1);
    tavevplumean = np.mean(vvplumean, axis = 1);
    tavewplumean = np.mean(wwplumean, axis = 1);
    taveyplumean = np.mean(yyplumean, axis = 1);

matuplumean = np.zeros((N1,N2,int(N3/2)-1))
matvplumean = np.zeros((N1,N2,int(N3/2)-1))
matwplumean = np.zeros((N1,N2,int(N3/2)-1))

matuplumean[:,:,:] = taveuplumean
matvplumean[:,:,:] = tavevplumean
matwplumean[:,:,:] = tavewplumean

# Compute fluctuations by using instantaneous values and
# constantly evolving averages. Average fluctuations over samples.
# For diagonal Reynolds stresses, compute RMS of fluctuation.
# For off-diagonal, compute mean of fluctuation stress component
# product.

# u fluctuations
ufluc1 = uplu1 - matuplumean
ufluc2 = uplu2 - matuplumean

ufluc1sq = ufluc1*ufluc1
ufluc2sq = ufluc2*ufluc2
uflucsq = np.sqrt(ufluc1sq + ufluc2sq)

uflucsq = np.mean(uflucsq, axis = 0)
uflucsq = np.mean(uflucsq, axis = 0)

if i == interval*statinit:
    uufluc = uflucsq;
    uplurms = uflucsq**(0.5);
    uplurms = np.reshape(uplurms,[int(N3/2)-1,1]);

if i > interval*statinit:
    uufluc = np.concatenate((uufluc,uflucsq), axis = 1)
    uplurms = np.mean(uufluc, axis = 1)
    uplurms = uplurms**(0.5)
    uplurms = np.reshape(uplurms,[int(N3/2)-1,1])

# v fluctuations
vfluc1 = vplu1 - matvplumean
vfluc2 = vplu2 - matvplumean

vfluc1sq = vfluc1*vfluc1
vfluc2sq = vfluc2*vfluc2
vflucsq = np.sqrt(vfluc1sq + vfluc2sq)

vflucsq = np.mean(vflucsq, axis = 0)
vflucsq = np.mean(vflucsq, axis = 0)

if i == interval*statinit:
    vvfluc = vflucsq;
    vplurms = vflucsq**(0.5);
    vplurms = np.reshape(vplurms,[int(N3/2)-1,1]);

if i > interval*statinit:
    vvfluc = np.concatenate((vvfluc,vflucsq), axis = 1)
    vplurms = np.mean(vvfluc, axis = 1)
    vplurms = vplurms**(0.5)
    vplurms = np.reshape(vplurms,[int(N3/2)-1,1])

# w fluctuations
wfluc1 = wplu1 - matwplumean
wfluc2 = wplu2 - matwplumean

wfluc1sq = wfluc1*wfluc1
wfluc2sq = wfluc2*wfluc2
wflucsq = np.sqrt(wfluc1sq + wfluc2sq)

wflucsq = np.mean(wflucsq, axis = 0)
wflucsq = np.mean(wflucsq, axis = 0)

if i == interval*statinit:
    wwfluc = wflucsq;
    wplurms = wflucsq**(0.5);
    wplurms = np.reshape(wplurms,[int(N3/2)-1,1]);

if i > interval*statinit:
    wwfluc = np.concatenate((wwfluc,wflucsq), axis = 1)
    wplurms = np.mean(wwfluc, axis = 1)
    wplurms = wplurms**(0.5)
    wplurms = np.reshape(wplurms,[int(N3/2)-1,1])

uvfluc1 = uplu1*vplu1 - matuplumean*matvplumean
uvfluc2 = uplu2*vplu2 - matuplumean*matvplumean
uvfluc = 0.5*(uvfluc1 + uvfluc2)

uvfluc = np.mean(uvfluc, axis = 0)
uvfluc = np.mean(uvfluc, axis = 0)

if i == interval*statinit:
    uuvvfluc = uvfluc
    uupluvvplumean = uvfluc
    uupluvvplumean = np.reshape(uupluvvplumean,[int(N3/2)-1,1])
    
if i > interval*statinit:
    uuvvfluc = np.concatenate((uuvvfluc,uvfluc), axis = 1)
    uupluvvplumean = np.mean(uuvvfluc, axis = 1)
    uupluvvplumean = np.reshape(uupluvvplumean,[int(N3/2)-1,1])

uwfluc1 = uplu1*wplu1 - matuplumean*matwplumean
uwfluc2 = uplu2*wplu2 - matuplumean*matwplumean
uwfluc = 0.5*(uwfluc1 + uwfluc2)

uwfluc = np.mean(uwfluc, axis = 0)
uwfluc = np.mean(uwfluc, axis = 0)

if i == interval*statinit:
    uuwwfluc = uwfluc
    uupluwwplumean = uwfluc
    uupluwwplumean = np.reshape(uupluwwplumean,[int(N3/2)-1,1])
    
if i > interval*statinit:
    uuwwfluc = np.concatenate((uuwwfluc,uwfluc), axis = 1)
    uupluwwplumean = np.mean(uuwwfluc, axis= 1)
    uupluwwplumean = np.reshape(uupluwwplumean,[int(N3/2)-1,1])

vwfluc1 = vplu1*wplu1 - matvplumean*matwplumean
vwfluc2 = vplu2*wplu2 - matvplumean*matwplumean
vwfluc = 0.5*(vwfluc1+vwfluc2)

vwfluc = np.mean(vwfluc, axis = 0)
vwfluc = np.mean(vwfluc, axis = 0)

if i == interval*statinit:
    vvwwfluc = vwfluc
    vvpluwwplumean = vwfluc
    vvpluwwplumean = np.reshape(vvpluwwplumean,[int(N3/2)-1,1])

if i > interval*statinit:
    vvwwfluc = np.concatenate((vvwwfluc,vwfluc), axis = 1)
    vvpluwwplumean = np.mean(vvwwfluc, axis = 1)
    vvpluwwplumean = np.reshape(vvpluwwplumean,[int(N3/2)-1,1])

# For post-processing
uplumean = np.reshape(taveuplumean,[int(N3/2)-1,1])







