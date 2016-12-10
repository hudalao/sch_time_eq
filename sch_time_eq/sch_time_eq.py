import sys
import os
#make sure the program can be executable from test file
dir_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(dir_root)


import numpy as np
import matplotlib.pyplot as plt
import math as mt
import numpy.polynomial.legendre as legen
import cmath
from function import wave_fourier_basis, Hamiltonian_momentum_basis, Legendre_polynomial_basis, reconstruct_wave, \
                    Hamiltonian_Legendre_polynomial

#input includes:
#c the constant
#N the size of the basis set
#V the potential energy V(x) ps: the size of V(x) should be same as the size of the basis set
#V_const the constant potential energy 
#domain is the range of V(x)
#the choice of basis set function: 1 ---> the fourier basis     2 ---> the legendre polynomial basis
#ps: the fourier basis must take V as a function of x, which means the V input must be a array, and the legendre polynomial basis  can only take the constant V. Be careful when you use different basis method

def output(c, V, V_const, N, wave_func, choice, domain):
    if choice == 1:
        matrix = Hamiltonian_momentum_basis(c, V, domain, N)
        wave_fourier = wave_fourier_basis(wave_func, domain, N)
        result = np.dot(matrix, wave_fourier)
        return result

    elif choice == 2:
        return Legendre_polynomial_basis(c, V_const, domain, N, wave_func) 
    
#    
def ground_wave_function(c, V, V_const, domain, N, choice):
    if choice == 1:
        matrix = Hamiltonian_momentum_basis(c, V, domain, N)
        w, v = np.linalg.eig(matrix)
        w = w.real
        #sorting the eigenvalues and conrresponding eigenvectors
        idx = w.argsort()
        w = w[idx]
        v = v[:,idx]
        ground_wave = reconstruct_wave(v[:,0], domain, N)
        
        return ground_wave
    elif choice == 2:
        matrix = Hamiltonian_Legendre_polynomial(c, V_const, domain, N)
        w, v = np.linalg.eig(matrix)
        w = w.real
        #sorting the eigenvalues and conrresponding eigenvectors
        idx = w.argsort()
        w = w[idx]
        v = v[:,idx]
        #transform to the wave function
        x = np.linspace(-domain / 2, domain / 2, N)
        ground_wave = legen.legval(x, v[: , 0]) 
        
        return ground_wave



