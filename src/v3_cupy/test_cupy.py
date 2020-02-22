#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 15:12:48 2020

@author: suraj
"""

import numpy as np
import cupy as cp
import cupyx as cpx
from scipy.sparse import spdiags
import scipy.sparse as sp

#%%
data = np.ones((3,4))
diags = np.array([-1,0,1])
M= spdiags(data, diags, 4, 4)
M = sp.csc_matrix(M)
print(M[0,0])
M[0,0] = 5
print(M[0,0])

#%%

datac = cp.ones((3,4))
diagsc = cp.array(diags)
Mc = cpx.scipy.sparse.spdiags(datac, diagsc, 4, 4)
Mc = cpx.scipy.sparse.csc_matrix(Mc)

Mc = cpx.scipy.sparse.csc_matrix(Mc)
print(Mc.get()[0,0])
Mc[0,0] = 5
print(Mc[0,0])