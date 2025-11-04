#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 12:07:17 2024

Find white-noise thresholds for Eberhard's inequality, using a 
maximally entangled state.
eta is efficiency.

@author: s2897605
"""
import numpy as np
import qutip as qt

def expectation_val(x, eta):
    A = eta/2*(np.exp(2*1j*x[0])-1)
    B = np.exp(2*1j*x[1])-1
    bell_op = eta/2*np.array([[2-eta, 1-eta, 1-eta, A.conjugate()*B.conjugate()-eta],
                              [1-eta, 2-eta, A*B.conjugate()-eta, 1-eta],
                              [1-eta, A.conjugate()*B-eta, 2-eta, 1-eta],
                              [A*B-eta, 1-eta, 1-eta, 2-eta]])
    omega = np.pi/8
    state = 1/np.sqrt(2)*(np.exp(-1j*omega)*qt.tensor(qt.basis(2,0),qt.basis(2,0)) + np.exp(1j*omega)*qt.tensor(qt.basis(2,1),qt.basis(2,1)))
    dm = qt.ket2dm(state).full()
    return np.real(np.trace(bell_op@dm))

from scipy.optimize import differential_evolution

def min_expectation(eta):
    bounds = [(-np.pi,np.pi),(-np.pi,np.pi)]
    result = differential_evolution(expectation_val, bounds, args=(eta,), disp=False, popsize=10)
    return result.fun

eta_vals=np.linspace(1,2*(np.sqrt(2)-1),1000)
purity_cutoffs = np.empty(len(eta_vals))
for i, eta in enumerate(eta_vals):
    print(i)
    res_cutoff = min_expectation(eta)
    purity_cutoffs[i]=1/(1-eta*(1-eta/2)/res_cutoff)


import matplotlib.pyplot as plt

plt.semilogy(eta_vals, purity_cutoffs,'-')
plt.xlabel(r'$\epsilon$')
plt.ylabel(r'$\xi$')

np.savez('eberhard_maxent_noise_cutoffs_noiseless', x=eta_vals, y=purity_cutoffs)
