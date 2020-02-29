#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 22:00:15 2020

@author: suraj
"""


import numpy as np
#import cupy as cp
#import cupyx as cpx
from scipy.sparse import spdiags
import scipy.sparse as sp
import scipy.io
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import bicgstab
from scipy.sparse.linalg import spilu
import time
#from numba import jit
#from numba import vectorize
import datetime
#import timeit

from mpi4py import MPI

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
comm = MPI.COMM_WORLD
mpi_size = comm.Get_size()
mpi_rank = comm.Get_rank()

if mpi_rank == 0:
    start_mpi = time.time()

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
res = 288 # Resolution; must be divisible by 2
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

#A = np.zeros((N1*N2*N3,1))
#SA = np.shape(A)
#
##for aa in range(SA[0]):
##    A[aa] = aa
#
#pp = 0
#A = np.zeros((N1,N2,N3),dtype=int)
#
#for k in range(N3):
#    for j in range(N2):
#        for i in range(N1):
#            A[i,j,k] = int(pp)
#            pp = pp +1

pp = np.linspace(0,N1*N2*N3-1,N1*N2*N3,dtype=int)
A = np.reshape(pp,[N1,N2,N3],order='F')
SA = np.shape(A.reshape(-1,1))


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

#for p in range(1,N3-1):
#    z[p] = fz[p-1] + 0.5*(fz[p] - fz[p-1])

z[1:N3-1] = fz[0:N3-2] + 0.5*(fz[1:N3-1] - fz[0:N3-2]) # vectorized

z[N3-1] = fz[N3-2] + 0.5*(fz[N3-2] - fz[N3-3])
z = z/fz[N3-2]*Height; fz = fz/fz[N3-2]*Height;


FZ = np.zeros((N1,N2,N3))

#for p in range(1,N3-1):
#    FZ[:,:,p] = fz[p]-fz[p-1]

FZ[:,:,1:N3-1] = fz[1:N3-1] - fz[0:N3-2] # vectorized

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
    if resume == 1:
        print('Loading old velocity field...')
        data =  np.load('fields1.npz') #add variables saved from runmode = 1
        U = data['U']
        V = data['V']
        W = data['W']
    
    else:
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
        
if runmode == 2:
    if resume == 1:
        print('Loading old velocity field...')
        data =  np.load('fields2.npz') #add variables saved from runmode = 2
        U = data['U']
        V = data['V']
        W = data['W']
    
    else:
        data =  np.load('fields1.npz') #add variables saved from runmode = 0
        U = data['U']
        V = data['V']
        W = data['W']
        Xo = data['X0']
        Yo = data['Y0']
        Zo = data['Z0']
        
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

A0c = A[:,:,inz[0]:inz[-1]+1] - N1*N2
ANc = A[north[0]:north[-1]+1,inx[0]:inx[-1]+1,inz[0]:inz[-1]+1] - N1*N2
ASc = A[south[0]:south[-1]+1,inx[0]:inx[-1]+1,inz[0]:inz[-1]+1] - N1*N2
AEc = A[iny[0]:iny[-1]+1,east[0]:east[-1]+1,inz[0]:inz[-1]+1] - N1*N2
AWc = A[iny[0]:iny[-1]+1,west[0]:west[-1]+1,inz[0]:inz[-1]+1] - N1*N2
AAc = A[iny[0]:iny[-1]+1,inx[0]:inx[-1]+1,air[0]:air[-1]+1] - N1*N2
AGc = A[iny[0]:iny[-1]+1,inx[0]:inx[-1]+1,ground[0]:ground[-1]+1] - N1*N2

u = np.reshape(U,[-1,1],order='F')
v = np.reshape(V,[-1,1],order='F')
w = np.reshape(W,[-1,1],order='F')

pold = np.zeros((N1*N2*(N3-2),1))

#%%
print('size = ', mpi_size)
print('rank = ', mpi_rank, ' ', u.shape)

#%%
   
inb = 2
sx = inx.shape[0]
sy = iny.shape[0]
sz = inz.shape[0]
shp = int(sx*sy*sz*3/mpi_size) # per processor
row = np.zeros(shp, dtype='i')
col = np.zeros(shp, dtype='i')
dat = np.zeros(shp, dtype='i')
p = 0

lbx = int(mpi_rank*sx/mpi_size)
ubx = int((mpi_rank+1)*sx/mpi_size)
lby = int(mpi_rank*sy/mpi_size)
uby = int((mpi_rank+1)*sy/mpi_size)
lbz = int(mpi_rank*sy/mpi_size)
ubz = int((mpi_rank+1)*sy/mpi_size)
 
for k in inz[lbz:ubz]-1:
    for j in inx[lbx:ubx]:
        for i in iny[lby:uby]:
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
                row[p] = A0[i,j,k]
                col[p] = A0[i,j,k]
                dat[p] = 1/FY0*(FYN/(FY0+FYN)-FYS/(FYS+FY0))
                p = p+1
            
            if inb == 2:
                row[p] = A0[i,j,k]
                col[p] = A0[i,j,k]
                dat[p] = 1/FX0*(FXE/(FX0+FXE)-FXW/(FXW+FX0))
                p = p+1

                    
            if inb == 3:
                row[p] = A0[i,j,k]
                col[p] = A0[i,j,k]
                dat[p] = 1/FZ0*(FZA/(FZ0+FZA)-FZG/(FZG+FZ0))
                p = p+1
                
            if inb == 1:
                row[p] = A0[i,j,k]
                col[p] = AN.ravel(order='F')[A0[i,j,k]]
                dat[p] = 1/FY0*FY0/(FY0+FYN)
                p = p+1
                
                row[p] = A0[i,j,k]
                col[p] = AS.ravel(order='F')[A0[i,j,k]]
                dat[p] = -1/FY0*FY0/(FY0+FYS)
                p = p+1
                
            if inb == 2:
                row[p] = A0[i,j,k]
                col[p] = AE.ravel(order='F')[A0[i,j,k]]
                dat[p] = 1/FX0*FX0/(FX0+FXE)
                p = p+1
                
                row[p] = A0[i,j,k]
                col[p] = AW.ravel(order='F')[A0[i,j,k]]
                dat[p] = -1/FX0*FX0/(FX0+FXW)
                p = p+1
                
            # Account for wall Dirichlet (no-slip) condition in the wall-normal direction
            if inb == 3:
                if AG.ravel(order='F')[A0[i,j,k]] >= 0:
                    row[p] = A0[i,j,k]
                    col[p] = AG.ravel(order='F')[A0[i,j,k]]
                    dat[p] = -1/FZ0*FZ0/(FZ0+FZG)
                    p = p+1
                else:
                    row[p] = A0[i,j,k]
                    col[p] = A0[i,j,k]
                    dat[p] = 1/FZ0*(FZA/(FZ0+FZA)-FZG/(FZG+FZ0) + FZ0/(FZG+FZ0))
                    p = p+1
                    
                if AA.ravel(order='F')[A0[i,j,k]] < N1*N2*(N3-2):
                    row[p] = A0[i,j,k]
                    col[p] = AA.ravel(order='F')[A0[i,j,k]]
                    dat[p] = 1/FZ0*FZ0/(FZ0+FZA)
                    p = p+1
                else:
                    row[p] = A0[i,j,k]
                    col[p] = A0[i,j,k]
                    dat[p] = 1/FZ0*(FZA/(FZ0+FZA)-FZG/(FZG+FZ0) - FZ0/(FZA+FZ0))
                    p = p+1

data_recvbuf = None
row_recvbuf = None
col_recvbuf = None

comm.Barrier()

if mpi_rank == 0:  
    sx = inx.shape[0]
    sy = iny.shape[0]
    sz = inz.shape[0]
    shp = int(sx*sy*sz*3/mpi_size) # per processor        
    data_recvbuf = np.zeros([mpi_size, shp], dtype='float64')
    row_recvbuf = np.zeros([mpi_size, shp], dtype='i')
    col_recvbuf = np.zeros([mpi_size, shp], dtype='i')
    
comm.Gather(dat, data_recvbuf, root=0)
comm.Gather(row, row_recvbuf, root=0)
comm.Gather(col, col_recvbuf, root=0)

del dat
del row
del col

if mpi_rank == 0:
    data_recvbuf = np.reshape(data_recvbuf, [-1,])
    row_recvbuf = np.reshape(row_recvbuf, [-1,])
    col_recvbuf = np.reshape(col_recvbuf, [-1,])
    M = sp.csc_matrix(( data_recvbuf, (row_recvbuf,col_recvbuf) ))
    del data_recvbuf
    del row_recvbuf
    del col_recvbuf
    print(M.shape)
    end_mpi = time.time()
    print('parallel = ', end_mpi - start_mpi)
    

#%%
# comm.Disconnect()
    
#%%
start = time.time()
n = 500
for i in range(n):
    for j in range(n):
        for k in range(n):
            a = 1.0

for i in range(n):
    for j in range(n):
        for k in range(n):            
            b = 1.0

for i in range(n):
    for j in range(n):
        for k in range(n):
            c = 1.0

print('time = ', time.time() - start)