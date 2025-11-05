#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 11:23:34 2023

Reproducing Eberhard's noise threshold, via minimizing determinant of B in the SM.
eps is efficiency. Reproduces exactly. Fast convergence.

@author: s2897605
"""
import numpy as np

#taking eta=0, i.e. no noise.
def det_B(x, eps):
    T = eps/2*(np.exp(1j*x[0])-1)
    R = np.exp(1j*x[1])-1
    bell_op = eps/2*np.array([[2-eps, 1-eps, 1-eps, T*R-eps],
                              [1-eps, 2-eps, T*R.conjugate()-eps, 1-eps],
                              [1-eps, T.conjugate()*R-eps, 2-eps, 1-eps],
                              [T.conjugate()*R.conjugate()-eps, 1-eps, 1-eps, 2-eps]])
    return np.real(np.linalg.det(bell_op))

from scipy.linalg import eigvalsh

def min_eig_from_B(x,eps):
    T = eps/2*(np.exp(1j*x[0])-1)
    R = np.exp(1j*x[1])-1
    bell_op = eps/2*np.array([[2-eps, 1-eps, 1-eps, T*R-eps],
                              [1-eps, 2-eps, T*R.conjugate()-eps, 1-eps],
                              [1-eps, T.conjugate()*R-eps, 2-eps, 1-eps],
                              [T.conjugate()*R.conjugate()-eps, 1-eps, 1-eps, 2-eps]])
    return np.min(np.real(eigvalsh(bell_op)))

from scipy.optimize import differential_evolution

def min_eig(eps):
    bounds = [(0,np.pi),(0,np.pi)]
    result = differential_evolution(det_B, bounds, args=(eps,), popsize=10)
    return min_eig_from_B(result.x,eps)

eps_vals=np.linspace(2/3,1,1000)
purity_cutoffs = np.empty(len(eps_vals))
for i, eps in enumerate(eps_vals):
    print(i)
    eig_min = min_eig(eps)
    purity_cutoffs[i]= 1/(1-eps*(1-eps/2)/eig_min)

import matplotlib.pyplot as plt
plt.semilogy(eps_vals, purity_cutoffs,'k-')
plt.xlabel(r'$\bar{\epsilon}$')
plt.ylabel(r'$\eta$')

np.savez('eberhard_noise_cutoffs_no_noise', x=eps_vals, y=purity_cutoffs)
