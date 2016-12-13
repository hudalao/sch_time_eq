#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_sch_eq
----------------------------------

Tests for `sch_eq` module.
"""

import sys
import unittest

import numpy as np
import math as mt
import cmath
import multiprocessing as mp
from functools import partial

from sch_time_eq.function import  Hamiltonian_momentum_basis,  x_sequential, rearrange_matrix, \
        x_solver, wave_function_propagation_sequential, wave_function_propagation_parallel 


class TestSch_eq(unittest.TestCase):
    def test_ax_b_implementation(self):
        #testing if Ax=b solver got the same result as the numpy.solve function
        L1 = np.array([[1.0,0.,0.,0.],[0.,5.0,0.,0.],[0.,5.0,-1.0,0.],[0.,0.3,0.,7.0]])
        b1 = np.array([3.0,-3.0,-11.0,-5.0])

        result = x_sequential(L1, b1)
        ref = np.linalg.solve(L1, b1)
       
        result = np.around(result, decimals = 2)
        ref = np.around(ref, decimals = 2)

        for ii in range(ref.size):
            self.assertEqual(result[ii], ref[ii])

    def test_ax_b_implementation_parallel(self):
        L1 = np.array([[1.0,0.0,0.0,0.0],[0.0,5.0,0.0,0.0],[0.0,5.0,-1.0,0.0],[0.0,0.3,0.0,7.0]])
        b1 = np.array([3.0,-3.0,-11.0,-5.0])

        ref = np.linalg.solve(L1, b1)
        
        size1 = L1.shape[0]
        ii1 = range(size1)
        partial_rearrange = partial(rearrange_matrix,size=size1,L=L1,b=b1 )
        if __name__ == '__main__':
            pool = mp.Pool()
            result = pool.map(partial_rearrange, ii1)
            L2 = result[0][0]
            b2 = result[0][1]
            partial_solver = partial(x_solver, L=L2,b=b2)
            nn1 = range(size1 - 1, -1, -1)
            x = pool.map(partial_solver, nn1)
            pool.close()
            x_rev = x[::-1]
            x_rev = np.around(x_rev, decimals = 2)
            ref = np.around(ref, decimals = 2)
            for ii in range(ref.size):
                self.assertEqual(x_rev[ii], ref[ii])

    def test_harmonic_oscillator(self):
        #the ODE describing the harmonic oscillator has the same for as our ODE,becaue we know the eigenvalues(energy) of the harmonic oscillator is E = h_bar * frequence * (N + 1/2), where N is the occupation number of phonons, the values of N is from 0 to infinity
        
        #using Hartree atomic units, h_bar = 1
        #define c as h_bar ** 2 /2m
        m = 1
        omega = 1
        c = 1 / 2 / m 
        domain = 15
        N = 70
        
        #the corresponding potential
        x  = np.linspace(-1 * domain / 2, domain / 2, N)
        potential = m * omega ** 2 * x ** 2 / 2
        
        #from analytic solution, the first five energy are
        E = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        
        h1 = Hamiltonian_momentum_basis(c, potential, domain, N)
        eigenvalues = np.linalg.eigvals(h1)
        eigenvalues_real_sort = np.sort(eigenvalues.real)
        #selecting first five energy from our calculated results
        #only keep one digit after the decimal
        E_calc = np.around(eigenvalues_real_sort[:5], decimals = 1)
        for ii in range(5):
            self.assertEqual(E[ii], E_calc[ii])

    def test_wave_function_propagation(self):
        #here, still using harmonic oscillator potential
        #using Hartree atomic units, h_bar = 1
        #define c as h_bar ** 2 /2m
        #
        m = 1
        omega = 1
        c = 1 / 2 / m 
        domain = 12
        N = 50
        total_time = 2
        time_interval = 0.01
      
        #the corresponding potential
        x  = np.linspace(-1 * domain / 2, domain / 2, N)
        potential = m * omega ** 2 * x ** 2 / 2
        H = Hamiltonian_momentum_basis(c, potential, domain, N)
        w, v = np.linalg.eig(H)
        w = w.real
        #sorting the eigenvalues and conrresponding eigenvectors
        idx = w.argsort()
        w = w[idx]
        v = v[:,idx]
        ground_wave = v[:,0]  #in momentum basis
        print(w[0])
        #here, we can pick the first eigenfunction corresponding to the ground energy as our initial wavefunction,
        #in that senario, according to time depedent schrodinger equation, the solution is phi(t) = exp(-iE0t/h_bar) * phi0
        #here, phi0 is the first eigenfunction and E0 is the ground state energy
        #thus, we can use this to test if our function works
        #when it is known the ground energy is 0.5, when t = 2, phi(t=2) = exp(-i) * phi0
        ref = cmath.exp(-1j) * ground_wave

        initial_wave = ground_wave
        wave = wave_function_propagation_sequential(H, initial_wave, total_time, time_interval)
        
        wave1 = np.round(wave.real, decimals = 2)
        ref1 = np.round(ref.real, decimals = 2)
        
        #there is some error between calculated result with the expected result around 0.01 which is
        #acceptable in this test
        #check if the error between two results is within 0.01
        for ii in range(N):
            #return True if the error between two results is within 0.01
            check = mt.isclose(wave1[ii], ref1[ii], rel_tol=0.01)
        self.assertEqual(check,True)

