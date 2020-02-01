#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 09:58:10 2020

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
import datetime

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable

font = {'family' : 'Times New Roman',
        'size'   : 14}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

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

runmode = 2
retain = 1 # Select whether to save results 
resume = 0 # Select whether to load results
retainoperators = 1 # Select whether to save differential operators
resumeoperators = 0 # Select whether to load differential operators
interpolatenew = 1 # Select whether to interpolate results onto a new grid
                    # This requires runmode = 2
validate = 0    # Use when same initial field as matlab is to be used

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

nsteps = 1000 # Provide number of time steps
tstepopt = 1 # Time step option: 0 = fixed, 1 = dynamic
setdt = 4.501E-3 # Fixed time step (requires tstepopt = 0)
CoTarget = 0.5 # Provide target for maximum local Courant number
                # (requires tstepopt = 1)
                
interval = 10 # Provide output interval
intervalplot = 10 #
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

plo11 = 0 # Velocity and fluctuation profiles
plo12 = 0 # Velocity contours
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

#-----------------------------------------------------------------------------#
# This routine finds an arbitrary number of nearest neighbor points 
# of an old grid (X,Y,Z) for each point of a new grid (Xn,Yn,Zn) 
# and performs a weighted interpolation of velocities from the old grid 
# (U,V,W) to the new grid (Un,Vn,Wn)
#-----------------------------------------------------------------------------#

def ThreeDChannel_newgridvelocity(N1,N2,N3,X,Y,Z,U,V,W,Xn,Yn,Zn,pnum):
    
    S = Xn.shape
    ind = np.zeros((3,pnum), dtype='int')
    weight = np.zeros((1,pnum))
    
    Un = np.zeros((N1,N2,N3))
    Vn = np.zeros((N1,N2,N3))
    Wn = np.zeros((N1,N2,N3))
    
    for i in range(S[0]): # Y direction
        for j in range(S[1]): # X direction
            for k in range(S[2]):
                
                dist = (X-Xn[i,j,k])**2 + (Y-Yn[i,j,k])**2 + (Z-Zn[i,j,k])**2
                dist = np.sqrt(dist)
                diststore = np.copy(dist)
                
                for s in range(pnum):
                    minval = np.min(dist)
                    [iy,ix,iz] = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
                    ind[0,s] = iy
                    ind[1,s] = ix
                    ind[2,s] = iz
                    
                    dist[iy,ix,iz] = np.max(dist)
                
                dist = np.copy(diststore)
                
                if minval == 0: # special case of the exact same point
                    Un[i,j,k] = U[ind[0,0], ind[1,0], ind[2,0]]
                    Vn[i,j,k] = V[ind[0,0], ind[1,0], ind[2,0]]
                    Wn[i,j,k] = W[ind[0,0], ind[1,0], ind[2,0]]
                
                else:
                    for s in range(pnum): # Calculate weights
                        weight[0,s] = 1/dist[ind[0,s], ind[1,s], ind[2,s]]
                    
                    # Normalize weights
                    weightsum = np.sum(weight)
                    weight = weight/weightsum
                    
                    # compute velocity on new grid points
                    for s in range(pnum):
                        Un[i,j,k] = Un[i,j,k] + weight[0,s]*U[ind[0,s], ind[1,s], ind[2,s]]
                        Vn[i,j,k] = Vn[i,j,k] + weight[0,s]*V[ind[0,s], ind[1,s], ind[2,s]]
                        Wn[i,j,k] = Wn[i,j,k] + weight[0,s]*W[ind[0,s], ind[1,s], ind[2,s]]
    
    return Un, Vn, Wn



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
    data =  np.load('fields0.npz') #add variables saved from runmode = 0
    U = data['U']
    V = data['V']
    W = data['W']
    UF1 = UF*(np.random.rand(N1,N2,N3-2) - 0.5)
    UF2 = UF*(np.random.rand(N1,N2,N3-2) - 0.5)
    UF3 = UF*(np.random.rand(N1,N2,N3-2) - 0.5)
    U = U + UF1*np.max(U)
    V = V + UF2*np.max(U)
    W = W + UF3*np.max(U)
    
    if validate == 1:
        data = scipy.io.loadmat('initial_field.mat')
        U = data.get('U')
        V = data.get('V')
        W = data.get('W')
        
    if resume == 1:
        print('Loading old velocity field...')
        data =  np.load('fields1.npz') #add variables saved from runmode = 1
        U = data['U']
        V = data['V']
        W = data['W']
        # load('Field1') add saved variables for runmode = 1

if runmode == 2:
    data =  np.load('fields1.npz') #add variables saved from runmode = 0
    U = data['U']
    V = data['V']
    W = data['W']
    Xo = data['X0']
    Yo = data['Y0']
    Zo = data['Z0']
    
    #print('Field1 load script to be added')
    if resume == 1:
        print('Loading old velocity field...')
        data =  np.load('fields2.npz') #add variables saved from runmode = 2
        U = data['U']
        V = data['V']
        W = data['W']
        #print('Field2 load script to be added resume')
        
    if interpolatenew == 1:
        print('Interpolating velocity onto new grid...')
        U, V, W = ThreeDChannel_newgridvelocity(N1,N2,N3-2,Xo,Yo,Zo, \
                                                U,V,W,X,Y,Z,3)
        print('Interpolation complete.\n')

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
    #M = sp.csc_matrix(M)
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
    #M = sp.csc_matrix(M)
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
    #M = sp.csc_matrix(M)
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
                        M[A0[i,j,k],AG.ravel(order='F')[A0[i,j,k]]] = -1/FZ0*FZ0/(FZ0+FZG)
                    else:
                        M[A0[i,j,k],A0[i,j,k]] = 1/FZ0* \
                                                 (FZA/(FZ0+FZA)-FZG/(FZG+FZ0) - \
                                                  FZ0/(FZG+FZ0))
                    
                    if AA.ravel(order='F')[A0[i,j,k]] < N1*N2*(N3-2):
                        M[A0[i,j,k],AA.ravel(order='F')[A0[i,j,k]]] = 1/FZ0*FZ0/(FZ0+FZA)
                    else:
                        M[A0[i,j,k],A0[i,j,k]] = 1/FZ0* \
                                                 (FZA/(FZ0+FZA)-FZG/(FZG+FZ0) + \
                                                  FZ0/(FZA+FZ0))
    
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
    #M = sp.csc_matrix(M)
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
        data = scipy.io.loadmat('Operators0.mat')#, allow_pickle=True)
    if runmode == 1:
        data = scipy.io.loadmat('Operators1.mat')
    if runmode == 2:
        data = scipy.io.loadmat('Operators2.mat')
    
    Dx = data.get('Dx')
    Dy = data.get('Dy')
    Dz = data.get('Dz')
    Dxx = data.get('Dxx')
    Dyy = data.get('Dyy')
    Dzz = data.get('Dzz')
    Dxp = data.get('Dxp')
    Dyp = data.get('Dyp')
    Dzp = data.get('Dzp')
    M = data.get('M')
    
    del data
    
else:    
    print('Generating new linear operators for differentiation... ')
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
    if retainoperators == 1:
        if runmode == 0:
            scipy.io.savemat('Operators0.mat',{'Dx':Dx,'Dy':Dy, 'Dz':Dz,
                                               'Dxx':Dxx, 'Dyy':Dyy, 'Dzz':Dzz,
                                               'Dxp':Dxp, 'Dyp':Dyp, 'Dzp':Dzp, 
                                               'M':M})
            
        if runmode == 1:
            scipy.io.savemat('Operators1.mat',{'Dx':Dx,'Dy':Dy, 'Dz':Dz,
                                               'Dxx':Dxx, 'Dyy':Dyy, 'Dzz':Dzz,
                                               'Dxp':Dxp, 'Dyp':Dyp, 'Dzp':Dzp, 
                                               'M':M})
        
        if runmode == 2:
            scipy.io.savemat('Operators2.mat',{'Dx':Dx,'Dy':Dy, 'Dz':Dz,
                                               'Dxx':Dxx, 'Dyy':Dyy, 'Dzz':Dzz,
                                               'Dxp':Dxp, 'Dyp':Dyp, 'Dzp':Dzp, 
                                               'M':M})
#            np.savez('Operators2.npz', Dx=Dx, Dy=Dy, Dz=Dz, Dxx=Dxx, Dyy=Dyy, Dzz=Dzz,
#                     Dxp=Dxp, Dyp=Dyp, Dzp=Dzp, M=M)

#%%
# check operators wit matlab
#op1 = scipy.io.loadmat('Operators1.mat')
#Dxm = op1.get('Dx'); abc = Dxm-Dx; print(abc)
#Dym = op1.get('Dy'); abc = Dym-Dy; print(abc)
#Dzm = op1.get('Dz'); abc = Dzm-Dz; print(abc)
#Dxxm = op1.get('Dxx'); abc = Dxxm-Dxx; print(abc)
#Dyym = op1.get('Dyy'); abc = Dyym-Dyy; print(abc)
#Dzzm = op1.get('Dzz'); abc = Dzzm-Dzz; print(abc)
#Dxpm = op1.get('Dxp'); abc = Dxpm-Dxp; print(abc)
#Dypm = op1.get('Dyp'); abc = Dypm-Dyp; print(abc)
#Dzpm = op1.get('Dzp'); abc = Dzpm-Dzp; print(abc)
#Mm = op1.get('M'); abc = Mm-M; print(abc)
 
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

def projection(Dx,Dy,Dz,M,u,v,w,p,pold):
#-----------------------------------------------------------------------------#
# Compute divergence of velocity field
#-----------------------------------------------------------------------------#
    
    DIV = Dx*u+Dy*v+Dz*w
    
    if spcg == 1:
        #C = C.T*C
        [p,flag] = cg(-M,-DIV,pold,pcgtol,pcgmaxit,C)
    
    if sbicg == 1:
        #C = Lbicg*Ubicg
        #[p,flag] = bicgstab(M,DIV,pold,bicgtol,bicgmaxit,C)
        [p,flag] = bicgstab(M,DIV,pold,bicgtol,bicgmaxit)
    
    p = p.reshape(-1,1)
    pold = p
    
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

p = p.reshape(-1,1)
pold = p

# Compute pressure gradient
px = Dxp*p;
py = Dyp*p;
pz = Dzp*p;

# Correct velocity field (vectors) with pressure gradient
u = u-px.reshape(-1,1);
v = v-py.reshape(-1,1);
w = w-pz.reshape(-1,1);

#%%
#check = scipy.io.loadmat('check.mat')
#mat = check.get('w') #.reshape(-1,1)
#AABB = mat - w
#print(mat - p)

#%%
t = 0

if timing == 1:
    start_time = time.time()

for i in range(1,nsteps+1):

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
            u,v,w,pold = projection(Dx,Dy,Dz,M,u,v,w,p,pold)#,C,Lbicg,Ubicg)
            
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
        if ii == s-1:
            u = uc; v = vc; w = wc

    # carry out final projection
    u,v,w,pold = projection(Dx,Dy,Dz,M,u,v,w,p,pold)#,C,Lbicg,Ubicg)
    
    U = np.reshape(u,[N1,N2,N3-2],order='F')
    V = np.reshape(v,[N1,N2,N3-2],order='F')
    W = np.reshape(w,[N1,N2,N3-2],order='F')
    
    Cox = U*dt/FX[:,:,1:N3-1]
    Coy = V*dt/FY[:,:,1:N3-1]
    Coz = W*dt/FZ[:,:,1:N3-1]
    
    Co = Cox+Coy+Coz
    Comax = np.max(Co)
    
    print('n = ', i, 'dt = ', dt, 'Comax = ', Comax)
    
#%%

############## ThreeDChannel_RunPostProcessing step ###########################

    if i%interval == 0:
    
############## ThreeDChannel_Analyze_Fluctuations_Compact step ################
        
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
        
        ut2mean = np.mean(ut1) # ???
        
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
            taveuutmean = np.mean(uutmean, axis = 1).reshape([-1,1]);
            taveuplumean = np.mean(uuplumean, axis = 1).reshape([-1,1]);
            tavevplumean = np.mean(vvplumean, axis = 1).reshape([-1,1]);
            tavewplumean = np.mean(wwplumean, axis = 1).reshape([-1,1]);
            taveyplumean = np.mean(yyplumean, axis = 1).reshape([-1,1]);
        
        matuplumean = np.zeros((N1,N2,int(N3/2)-1))
        matvplumean = np.zeros((N1,N2,int(N3/2)-1))
        matwplumean = np.zeros((N1,N2,int(N3/2)-1))
        
        matuplumean[:,:,:] = taveuplumean[:,0]
        matvplumean[:,:,:] = tavevplumean[:,0]
        matwplumean[:,:,:] = tavewplumean[:,0]
        
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
        uflucsq = 0.5*(ufluc1sq + ufluc2sq)
        
        uflucsq = np.mean(uflucsq, axis = 0)
        uflucsq = np.mean(uflucsq, axis = 0)
        uflucsq = np.reshape(uflucsq,[-1,1])
        
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
        vflucsq = 0.5*(vfluc1sq + vfluc2sq)
        
        vflucsq = np.mean(vflucsq, axis = 0)
        vflucsq = np.mean(vflucsq, axis = 0)
        vflucsq = np.reshape(vflucsq,[-1,1])
        
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
        wflucsq = 0.5*(wfluc1sq + wfluc2sq)
        
        wflucsq = np.mean(wflucsq, axis = 0)
        wflucsq = np.mean(wflucsq, axis = 0)
        wflucsq = np.reshape(wflucsq,[-1,1])
        
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
        uvfluc = np.reshape(uvfluc,[-1,1])
        
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
        uwfluc = np.reshape(uwfluc,[-1,1])
        
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
        vwfluc = np.reshape(vwfluc,[-1,1])
        
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
        
        if i > interval*statinit:
            
            uflu = np.concatenate((uplu1, np.flip(uplu2, axis=2)), axis=2)*utmean - \
                   np.concatenate((matuplumean, np.flip(matuplumean, axis=2)), axis=2)
            
            vflu = np.concatenate((vplu1, np.flip(vplu2, axis=2)), axis=2)*utmean - \
                   np.concatenate((matvplumean, np.flip(matvplumean, axis=2)), axis=2)
                  
            wflu = np.concatenate((wplu1, np.flip(wplu2, axis=2)), axis=2)*utmean - \
                   np.concatenate((matwplumean, np.flip(matwplumean, axis=2)), axis=2)
            
            # compute fluctuations derivative
            
            dufdx = Dx*np.reshape(uflu,[N1*N2*(N3-2),1],order='F')
            dvfdx = Dx*np.reshape(vflu,[N1*N2*(N3-2),1],order='F')
            dwfdx = Dx*np.reshape(wflu,[N1*N2*(N3-2),1],order='F')
            dufdy = Dy*np.reshape(uflu,[N1*N2*(N3-2),1],order='F')
            dvfdy = Dy*np.reshape(vflu,[N1*N2*(N3-2),1],order='F')
            dwfdy = Dy*np.reshape(wflu,[N1*N2*(N3-2),1],order='F')
            dufdz = Dz*np.reshape(uflu,[N1*N2*(N3-2),1],order='F')
            dvfdz = Dz*np.reshape(vflu,[N1*N2*(N3-2),1],order='F')
            dwfdz = Dz*np.reshape(wflu,[N1*N2*(N3-2),1],order='F')
        
            AIJAIJ = dufdx**2 + dvfdx**2 + dwfdx**2 + \
                     dufdy**2 + dvfdy**2 + dwfdy**2 + \
                     dufdz**2 + dvfdz**2 + dwfdz**2
            
            AIJBIJ = dufdx**2 + dvfdx*dufdy + dwfdx*dufdz + \
                     dufdy*dvfdx + dvfdy**2 + dwfdy*dvfdz + \
                     dufdz*dwfdx + dvfdz*dwfdy + dwfdz**2
            
            TD = nu*(AIJAIJ + AIJBIJ)
            
            TDcut = np.reshape(TD,[N1,N2,N3-2],order='F')
            TDcut = TDcut[int(N1/2)-1,:,:]
            TDcut = np.reshape(TDcut,[N2,N3-2],order='F')
            TDcut = TDcut.T
        
################# ThreeDChannel_ComputeQCriterion step ########################

#-----------------------------------------------------------------------------#
# This script computes the Q criterion for visualization of turbulent structures
#-----------------------------------------------------------------------------#
            
        dudx = Dx*np.reshape(U,[-1,1],order='F')
        dudy = Dy*np.reshape(U,[-1,1],order='F')
        dudz = Dz*np.reshape(U,[-1,1],order='F')
        
        dvdx = Dx*np.reshape(V,[-1,1],order='F')
        dvdy = Dy*np.reshape(V,[-1,1],order='F')
        dvdz = Dz*np.reshape(V,[-1,1],order='F')
        
        dwdx = Dx*np.reshape(W,[-1,1],order='F')
        dwdy = Dy*np.reshape(W,[-1,1],order='F')
        dwdz = Dz*np.reshape(W,[-1,1],order='F')
        
        S2 =  dudx**2 + (0.5*(dudy+dvdx))**2 + (0.5*(dudz+dwdx))**2 + \
              (0.5*(dvdx+dudy))**2 + dvdy**2 + (0.5*(dvdz+dwdy))**2 + \
              (0.5*(dwdx+dudz))**2 + (0.5*(dwdy+dvdz))**2 + dwdz**2
        
        OM2 = 0 + (0.5*(dudy-dvdx))**2 + (0.5*(dudz-dwdx))**2 + \
              (0.5*(dvdx-dudy))**2 + 0 + (0.5*(dudz-dwdy))**2 + \
              (0.5*(dwdx-dudz))**2 + (0.5*(dwdy-dvdz))**2 + 0
              
        Q = -0.5*(S2-OM2)
        
        Q = np.reshape(Q,[N1,N2,N3-2],order='F')
        
        
################# ThreeDChannel_Process_Fields step ###########################

#-----------------------------------------------------------------------------#
# Field post-processing for visualization
#-----------------------------------------------------------------------------# 
        
        Vmag = np.sqrt(U**2 + V**2 +W**2) 
        
        bbb = U[:,:,0]/(0.5*FZ[:,:,0])
        bbb = np.where(bbb>0, bbb, 0)
        
        yplusall = 0.5*FZ[:,:,0]*np.sqrt(bbb)/np.sqrt(nu)
        
        DUDZ = Dz*np.reshape(U,[-1,1],order='F')
        
        UTAU1 = np.sqrt(nu*np.abs(U[:,:,0])/Z[:,:,0])
        UTAU2 = np.sqrt(nu*np.abs(U[:,:,N3-3])/(Height - Z[:,:,N3-3]))
        RTAU1 = UTAU1*(0.5*Height)/nu
        RTAU2 = UTAU2*(0.5*Height)/nu
        RTAUAVG = 0.5*(RTAU1 + RTAU2)
        RTAUAVG = np.mean(RTAUAVG)
        
        if i == interval:
            rtau = RTAUAVG
        else:
            rtau = np.vstack((rtau,RTAUAVG))
        
        # make slice for 2D interpretations
        
        Uslice = U[int(N1/2)-1,:,:]
        Uslice = np.reshape(Uslice,[N2,N3-2],order='F')
        Uslice = Uslice.T
        
        Uslice2 = U[:,:,yplane-1]
        Uslice2 = np.reshape(Uslice2,[N1,N2],order='F')
        
        Uslice3 = U[:,int(N2/4)-1,:]
        Uslice3 = np.reshape(Uslice3,[N1,N3-2],order='F')
        Uslice3 = Uslice3.T
    
        Uslice4 = U[:,int(3*N2/4)-1,:]
        Uslice4 = np.reshape(Uslice4,[N1,N3-2],order='F')
        Uslice4 = Uslice4.T
    
        Wslice = W[int(N1/2)-1,:,:]
        Wslice = np.reshape(Wslice,[N2,N3-2],order='F')
        Wslice = Wslice.T
    
        RT = (1/np.sqrt(nu))*(2/Height)*yplusall/FZ[:,:,1]    
        
        E = np.sum(Vmag[inx[0]:inx[-1]+1,iny[0]:iny[-1]+1,inz[0]-1:inz[-1]]**2 * \
            FX[inx[0]:inx[-1]+1,iny[0]:iny[-1]+1,inz[0]:inz[-1]+1] * \
            FY[inx[0]:inx[-1]+1,iny[0]:iny[-1]+1,inz[0]:inz[-1]+1] * \
            FZ[inx[0]:inx[-1]+1,iny[0]:iny[-1]+1,inz[0]:inz[-1]+1])
        
        HGX = 0.5*Height*gx
        NDZ = nu*DUDZ[0,0]
        
        MU = np.mean(U)
        
        if i == interval:
            e = E
            T = t
            hgx = HGX
            ndz = NDZ
        else:
            e = np.vstack((e,E))
            T = np.vstack((T,t))
            hgx = np.vstack((hgx, HGX))
            ndz = np.vstack((ndz, NDZ))
            
        XQ = X[:,:,0:int(N3/2)-1]
        YQ = Y[:,:,0:int(N3/2)-1]
        ZQ = Z[:,:,0:int(N3/2)-1]
        QQ = Q[:,:,0:int(N3/2)-1]
        VmagQ = Vmag[:,:,0:int(N3/2)-1]    
        
        
################# ThreeDChannel_Visualize step ################################

#-----------------------------------------------------------------------------#
# Visualization scripts
#-----------------------------------------------------------------------------#  
    if i%intervalplot == 0:
        if frame >= 2:
            # plot 1 (line plots)
            if plo11 == 1 or i == nsteps:
                fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(12,8))
                ax = ax.flat
                
                ax[0].semilogx(yplumean[:,0], uplumean[:,0],'o',markerfacecolor='none', \
                               markeredgecolor='black', label = 'Python')
#                ax[0].semilogx(yplumean_m[:,0], uplumean_m[:,0],'o',markerfacecolor='none', \
#                               markeredgecolor='red', label = 'Matlab')
                ax[0].semilogx(MSRyplus,MSRUmean,'o',markerfacecolor='black', \
                               markeredgecolor='black', label = 'Reference (MKM, 1999)')
                ax[0].set_ylim([0, 30])
                ax[0].set_xlim([0.1, np.max(MSRflucyplus)])
                ax[0].legend()
                
                ax[1].plot(yplumean[:,0], uplurms,'o',markerfacecolor='none', \
                               markeredgecolor='black', label = '$u^{+}$')
                ax[1].plot(yplumean[:,0], vplurms,'o',markerfacecolor='none', \
                               markeredgecolor='red', label = '$v^{+}$')
                ax[1].plot(yplumean[:,0], wplurms,'o',markerfacecolor='none', \
                               markeredgecolor='blue', label = '$w^{+}$')
                
                ax[1].plot(MSRflucyplus,MSRflucuplus,'o',markerfacecolor='black', \
                               markeredgecolor='black', label = '$u^{+}$ (MKM, 1999)')
                ax[1].plot(MSRflucyplus,MSRflucvplus,'o',markerfacecolor='red', \
                               markeredgecolor='red', label = '$v^{+}$ (MKM, 1999)')
                ax[1].plot(MSRflucyplus,MSRflucwplus,'o',markerfacecolor='blue', \
                               markeredgecolor='blue', label = '$w^{+}$ (MKM, 1999)')
                ax[1].set_ylim([0, 4.5])
                ax[1].set_xlim([0, np.max(MSRflucyplus)])
                ax[1].legend()
                
                ax[2].plot(yplumean[:,0], uupluvvplumean,'o',markerfacecolor='none', \
                               markeredgecolor='black', label = '$u^{+}v^{+}$')
                ax[2].plot(MSRflucyplus,MSRflucuvplus,'o',markerfacecolor='black', \
                               markeredgecolor='black', label = '$u^{+}v^{+}$ (MKM, 1999)')
                ax[2].set_ylim([-1, 0])
                ax[2].set_xlim([0, np.max(MSRflucyplus)])
                ax[2].legend()
                
                ax[3].plot(T,rtau, color='blue')
                ax[3].set_xlabel('$t$')
                ax[3].set_ylabel('Re'+r'$_\tau$', color='blue')
                ax[3].tick_params(axis='y', labelcolor='blue')
                ax[3].set_ylim([120, 200])
                
                ax2 = ax[3].twinx()  # instantiate a second axes that shares the same x-axis
                ax2.plot(T,e, color='red')
                ax2.set_xlabel('$t$')
                ax2.set_ylabel('$E$', color='red')
                ax2.tick_params(axis='y', labelcolor='red')
                #ax2.set_ylim([120, 200])
                
                fig.tight_layout()
                plt.show()
                fig.savefig('lineplot.pdf')
            
            #plot 2 (contour plot)
            if plo12 == 1 or i == nsteps:
                fig = plt.figure(tight_layout=True, figsize=(8,7))
                gs = gridspec.GridSpec(3, 2)
                
                ax = fig.add_subplot(gs[0, :])
                cs = ax.contourf(x,z[1:N3-1],Uslice, 20, cmap = 'jet',vmin=0,vmax=USC*Uscale,
                                 levels=np.linspace(0,USC*Uscale,100))
                
                # This is the fix for the white lines between contour levels
                for c in cs.collections:
                    c.set_edgecolor("face")

                cs.set_clim(0, USC*Uscale)
                cb = fig.colorbar(cs, orientation='vertical', fraction=0.15, aspect=5,
                             ticks=np.linspace(0, USC*Uscale, 5),format='%.1f')
                
                #ax.set_aspect(1.0)
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_title('Streamwise velocity, XY-plane')
                
                ax = fig.add_subplot(gs[1, :])
                cs = ax.contourf(x,y,Uslice2, 20, cmap = 'jet',vmin=0,vmax=USC*Uscale,
                                 levels=np.linspace(0,USC*Uscale,100))
                
                for c in cs.collections:
                    c.set_edgecolor("face")
                
                cs.set_clim(0, USC*Uscale)                                
                cb = fig.colorbar(cs, orientation='vertical', fraction=0.15, aspect=5, \
                                  ticks=np.linspace(0, USC*Uscale, 5),format='%.1f')
                
                #ax.set_aspect(1.0)
                ax.set_xlabel('x')
                ax.set_ylabel('z')
                ax.set_title('Streamwise velocity, XZ-plane ' + '${y}^{+} = $'+ \
                             str(np.round(yplumean[yplane],1)))
                
                
                ax = fig.add_subplot(gs[2, 0])
                cs = ax.contourf(y,z[1:N3-1],Uslice3, 20, cmap = 'jet',vmin=0,vmax=USC*Uscale,
                                 levels=np.linspace(0,USC*Uscale,100))
                
                for c in cs.collections:
                    c.set_edgecolor("face")
                                 
#                cs.set_clim(0, USC*Uscale
                
#                cb = fig.colorbar(cs, orientation='vertical', fraction=0.15, aspect=5, \
#                                  ticks=np.linspace(0, USC*Uscale, 5),format='%.1f')
#                cb.formatter.set_powerlimits((0, 0))
#                cb.ax.yaxis.set_offset_position('right')                         
#                cb.update_ticks()
                
                #ax.set_aspect(1.0)
                ax.set_xlabel('z')
                ax.set_ylabel('y')
                ax.set_title('Streamwise velocity, ZY-plane 1')
                
                ax = fig.add_subplot(gs[2, 1])
                cs = ax.contourf(y,z[1:N3-1],Uslice4, 20, cmap = 'jet',vmin=0,vmax=USC*Uscale,
                                 levels=np.linspace(0,USC*Uscale,100))
                
                for c in cs.collections:
                    c.set_edgecolor("face")

#                cs.set_clim(0, USC*Uscale
#                cb = fig.colorbar(cs, orientation='vertical', fraction=0.15, aspect=5, \
#                                  ticks=np.linspace(0, USC*Uscale, 5),format='%.1f')
#                cb.formatter.set_powerlimits((0, 0))
#                cb.ax.yaxis.set_offset_position('right')                         
#                cb.update_ticks()
                #ax.set_aspect(1.0) 
                ax.set_xlabel('z')
                ax.set_ylabel('y')
                ax.set_title('Streamwise velocity, ZY-plane 2')
                
                plt.show()
                fig.savefig('contourplot.pdf')
            
            #if plot3 == 3:
            
            #if plot4 == 4:
            
        frame = frame + 1

        if runmode == 0:
            np.savez('fields0.npz', U=U, V=V, W=W)
        
        if runmode == 1:
            X0 = X # cordiates for interpolation
            Y0 = Y # cordiates for interpolation
            Z0 = Z # cordiates for interpolation
            np.savez('fields1.npz', U=U, V=V, W=W, X0=X0, Y0=Y0, Z0=Z0)
            
        if runmode == 2:
            X0 = X # cordiates for interpolation
            Y0 = Y # cordiates for interpolation
            Z0 = Z # cordiates for interpolation
            np.savez('fields2.npz', U=U, V=V, W=W, X0=X0, Y0=Y0, Z0=Z0)
            
            if frame >=2 and i == nsteps:
                filename = 'Results_%d_%0.2f_%s.npz'% (res*2,Lscale,datetime.date.today())
                np.savez(filename, X=X, Y=Y, Z=Z, 
                        x=x, y=y, z=z, 
                        U=U, V=V, W=W, 
                        Vmag=Vmag, 
                        yplumean=yplumean, uplumean=uplumean, 
                        Length=Length, Width=Width, Height=Height, 
                        uplurms=uplurms, vplurms=vplurms, wplurms=wplurms, 
                        uupluvvplumean=uupluvvplumean, 
                        Uslice=Uslice, Uslice2=Uslice2, Uslice3=Uslice3, Uslice4=Uslice4, 
                        TD=TD, XQ=XQ, YQ=YQ, ZQ=ZQ, QQ=QQ, VmagQ=VmagQ, 
                        rtau=rtau, e=e, T=T, 
                        N1=N1, N2=N2, N3=N3, RTAUAVG=RTAUAVG, utnom=utnom)
                
    t = t + dt         

if timing == 1:
    end_time = time.time()

print('Elapsed time is ', end_time - start_time, ' Seconds')

#%%
#fields1 = scipy.io.loadmat('Fields1.mat')
#U = fields1.get('U')
#V = fields1.get('V')
#W = fields1.get('W')


#plot = scipy.io.loadmat('plot.mat')
#yplumean_m = plot.get('yplumean') #.reshape(-1,1)
#uplumean_m = plot.get('uplumean')
#AABB = mat - w
#print(mat - p)

#prec = scipy.io.loadmat('pre.mat')
#LL = prec.get('LL')
#UU = prec.get('UU') 
#
#
#data =  np.load('Operators1.npz', allow_pickle=True) #add variables saved from runmode = 1
#M = data['M']
#P = spilu(M)
#LLp = P.L
#UUp = P.U