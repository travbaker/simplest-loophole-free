#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 11:23:34 2023

Try reproduce Eberhard's noise threshold, via minimizing determinant of B.
eta is efficiency. Reproduces exactly. Fast convergence.

@author: s2897605
"""
import numpy as np

def paper_calculation(eta,zeta,alphadiff,betadiff):
    A = eta/2*(np.exp(2*1j*alphadiff)-1)
    B = np.exp(2*1j*betadiff)-1
    xi=4*zeta/eta
    bell_op = eta/2*np.array([[2-eta+xi, 1-eta, 1-eta, A.conjugate()*B.conjugate()-eta],
                              [1-eta, 2-eta+xi, A*B.conjugate()-eta, 1-eta],
                              [1-eta, A.conjugate()*B-eta, 2-eta+xi, 1-eta],
                              [A*B-eta, 1-eta, 1-eta, 2-eta+xi]])
    return np.real(np.linalg.det(bell_op))

#taking zeta=0, i.e. no noise.
def det_B(x, eta):
    A = eta/2*(np.exp(2*1j*x[0])-1)
    B = np.exp(2*1j*x[1])-1
    bell_op = eta/2*np.array([[2-eta, 1-eta, 1-eta, A.conjugate()*B.conjugate()-eta],
                              [1-eta, 2-eta, A*B.conjugate()-eta, 1-eta],
                              [1-eta, A.conjugate()*B-eta, 2-eta, 1-eta],
                              [A*B-eta, 1-eta, 1-eta, 2-eta]])
    return np.real(np.linalg.det(bell_op))

from scipy.linalg import eigvalsh

def min_eig_from_B(x,eta):
    A = eta/2*(np.exp(2*1j*x[0])-1)
    B = np.exp(2*1j*x[1])-1
    bell_op = eta/2*np.array([[2-eta, 1-eta, 1-eta, A.conjugate()*B.conjugate()-eta],
                              [1-eta, 2-eta, A*B.conjugate()-eta, 1-eta],
                              [1-eta, A.conjugate()*B-eta, 2-eta, 1-eta],
                              [A*B-eta, 1-eta, 1-eta, 2-eta]])
    return np.min(np.real(eigvalsh(bell_op)))

from scipy.optimize import differential_evolution

def min_eig(eta):
    bounds = [(0,np.pi),(0,np.pi)]
    result = differential_evolution(det_B, bounds, args=(eta,), popsize=10)
    return min_eig_from_B(result.x,eta)

eta_vals=np.linspace(2/3,1,1000)
purity_cutoffs = np.empty(len(eta_vals))
for i, eta in enumerate(eta_vals):
    print(i)
    eig_min = min_eig(eta)
    purity_cutoffs[i]= 1/(1-eta*(1-eta/2)/eig_min)

import matplotlib.pyplot as plt
plt.semilogy(eta_vals, 1-purity_cutoffs,'k-')
plt.xlabel(r'$\epsilon$')
plt.ylabel(r'$\xi$')

np.savez('eberhard_noise_cutoffs_no_noise', x=eta_vals, y=purity_cutoffs)
