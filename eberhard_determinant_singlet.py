#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 12:07:17 2024

Find white-noise thresholds for Eberhard's inequality, using a 
maximally entangled state.
eps is efficiency.

@author: s2897605
"""
import numpy as np
import qutip as qt

def expectation_val(x, eps):
    T = eps/2*(np.exp(1j*x[0])-1)
    R = np.exp(1j*x[1])-1
    bell_op = eps/2*np.array([[2-eps, 1-eps, 1-eps, T*R-eps],
                              [1-eps, 2-eps, T*R.conjugate()-eps, 1-eps],
                              [1-eps, T.conjugate()*R-eps, 2-eps, 1-eps],
                              [T.conjugate()*R.conjugate()-eps, 1-eps, 1-eps, 2-eps]])
    omega = np.pi/8
    state = 1/np.sqrt(2)*(np.exp(-1j*omega)*qt.tensor(qt.basis(2,0),qt.basis(2,0)) + np.exp(1j*omega)*qt.tensor(qt.basis(2,1),qt.basis(2,1)))
    dm = qt.ket2dm(state).full()
    return np.real(np.trace(bell_op@dm))

from scipy.optimize import differential_evolution

def min_expectation(eps):
    bounds = [(-np.pi,np.pi),(-np.pi,np.pi)]
    result = differential_evolution(expectation_val, bounds, args=(eps,), disp=False, popsize=10)
    return result.fun

eps_vals=np.linspace(1,2*(np.sqrt(2)-1),1000)
purity_cutoffs = np.empty(len(eps_vals))
for i, eps in enumerate(eps_vals):
    print(i)
    res_cutoff = min_expectation(eps)
    purity_cutoffs[i]=1/(1-eps*(1-eps/2)/res_cutoff)


import matplotlib.pyplot as plt

plt.semilogy(eps_vals, purity_cutoffs,'-')
plt.xlabel(r'$\bar{\epsilon}$')
plt.ylabel(r'$\eta$')

np.savez('eberhard_maxent_noise_cutoffs_noiseless', x=eps_vals, y=purity_cutoffs)
