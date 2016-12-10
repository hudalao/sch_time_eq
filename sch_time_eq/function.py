import sys
import os
#make sure the program can be executable from test file
dir_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(dir_root)


import numpy as np
import matplotlib.pyplot as plt
import math as mt
import cmath
import numpy.linalg as nlin
import multiprocessing as mp

#domain is the range of x and V(x)
#c the constant
#N the size of the basis set
#V the potential energy V(x) ps: the size of V(x) should be same as the size of the basis set
#V_const the constant potential energy 
#the choice of basis set function: 1 ---> the fourier basis 2 ---> the legendre polynomial basis
#ps: the fourier basis can take function V of x, but the legendre polynomial basis  can only take the constant V. Be careful when you use different basis method
#ps: in this question, we set h_bar to 1

#with input wave function, calculate its coefficient under the fourier basis
def wave_fourier_basis(wave_func, domain, N):
    x = np.linspace(-domain / 2, domain / 2, N)
    n = np.linspace(-N / 2 + 1, N / 2, N)
    exp_coeff = 1j * 2 * np.pi * n / domain
    delta_x = domain / (N - 1)
    a = np.zeros(N, dtype = complex)
    for ii in range(1, N):
        for kk in range(N):
            add = wave_func[kk] * cmath.exp( -1 * exp_coeff[ii] * x[kk] )
            a[ii] = a[ii] + add
    a = a / N
    return a

#reconstruct the original function for testing purpose
def reconstruct_wave(wave_fourier_coeff, domain, N):
    x = np.linspace(-domain / 2, domain / 2, N)
    n = np.linspace(-N / 2 + 1, N / 2, N)
    exp_coeff = 1j * 2 * np.pi * n / domain
    delta_p = 2 * np.pi / domain
    wave = np.zeros(N, dtype = complex)
    for kk in range(N):
        for ii in range(N):
            add = wave_fourier_coeff[ii] * \
                    cmath.exp( exp_coeff[ii] * x[kk] )
            wave[kk] = wave[kk] + add
    
    wave = wave * delta_p
    
    return wave
    
#here, we use the momentum basis which is a fourier basis set, which means we reprsent the whole (-c Lap + V) as matrix with the momentum basis
#potential here refers to V in the equation shown above
#the reson using this method is that we can obtain the eigenvalues and eigenvectors directly by diaglize this matrix
def Hamiltonian_momentum_basis(c, potential, domain, N):
    x = np.linspace(-domain / 2, domain / 2, N)
    n = np.linspace(-N / 2 + 1, N / 2, N)
    exp_coeff = 1j * 2 * np.pi * n / domain
    delta_x = domain / (N - 1)
    #potential term
    V = np.zeros((N, N), dtype = complex)

    for ii in range(N):
        for jj in range(N):
            for kk in range(N):
                brax_ketp = cmath.exp( exp_coeff[jj] * x[kk] )
                brap_ketx = cmath.exp( -1 * exp_coeff[ii] * x[kk] )
                add = brap_ketx * potential[kk] * brax_ketp * delta_x
                V[ii][jj] = V[ii][jj] + add
    #kinetic term
    K = np.zeros((N, N), dtype = complex)

    K_coeff = c * 4 * np.pi ** 2 / domain ** 2 
    for ii in range(N):
        K[ii][ii] = n[ii] ** 2 * N * delta_x

    K = K_coeff * K
    
    #it is known that HC = HSC, because S here is a identity matrix with elements 
    # equals to period, we can just divide the H by period value
    H = (K + V) / domain

    return H

#Here, in order to comparing the difference of efficiency between sequential and parallel algorithm, we will implement the matrix inversion algorithm by ourselves using Gaussian Elimination
def matrix_inversion_sequential(L):
    
    size = L.shape[0]
    #using Gaussian Elimination to calculate out the inverse matrix of L
    #in our case, the matrix L must have the inverse matrix becuase it is a symmetric matrix
    I = np.identity(size)
    for ii in range(size):
        scale = L[ii,ii]
        for jj in range(size):
            L[ii,jj] = L[ii,jj] / scale
            I[ii,jj] = I[ii,jj] / scale
        if ii < size - 1:
            for row in range(ii + 1, size):
                factor = L[row, ii]
                for col in range(1, size):
                    L[row,col]=L[row,col] - factor * L[ii,col]
                    I[row,col]=I[row,col] - factor * I[ii,col]

    #back substitution
    for ref in range(size - 1, 0, -1):
        for row in range(ref - 1, 0, -1):
            factor = L[row, ref]
            for col in range(row - size, size):
                L[row,col]=L[row,col] - factor * L[ref,col]
                I[row,col]=I[row,col] - factor * I[ref,col]
    
    return L, I

def matrix_inversion_parallel(L):

    #using Gaussian Elimination to calculate out the inverse matrix of L
    #in our case, the matrix L must have the inverse matrix becuase it is a symmetric matrix
    def matrix_inversion_func(param):
        size = L.shape[0]
        I = np.identity(size)
        scale = L[ii,ii]
        for jj in range(size):
            L[ii,jj] = L[ii,jj] / scale
            I[ii,jj] = I[ii,jj] / scale
        if ii < size - 1:
            for row in range(ii + 1, size):
                factor = L[row, ii]
                for col in range(1, size):
                    L[row,col]=L[row,col] - factor * L[ii,col]
                    I[row,col]=I[row,col] - factor * I[ii,col]
        return L, I 
    
    ii = range(size)
    pool = mp.Pool()
    results = pool.map(matrix_inversion_func, ii)
    
    
    #back substitution
    def back_substitution(results):
        size = L.shape[0]
        I = np.identity(size)
        for row in range(ref - 1, 0, -1):
            factor = L[row, ref]
            for col in range(row - size, size):
                L[row,col]=L[row,col] - factor * L[ref,col]
                I[row,col]=I[row,col] - factor * I[ref,col]
        return L, I 
    
    ref = range(size - 1, 0, -1)
    identity, matrix_inverse = pool.map(back_substitution, ref)
    
def wave_function_propagation_sequential(H, initial_wave, total_time, time_interval):
    
    t_points = total_time / time_interval + 1
    
    size = initial_wave.size

    #using Crank-Nicolson methond 
    #assuming Hamiltonian is time-independent in this case
    wave = initial_wave
    
    L = np.identity(size) + 1j * time_interval / 2 * H
    b_coef = np.identity(size) - 1j * time_interval / 2 * H
    

    for t in range(2:point)
        b = b_coef * wave
        wave = L_inverse * b
    
    return wave

def wave_function_propagation_parallel
