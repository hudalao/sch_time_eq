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
def x_sequential(L, b):
    
    size = L.shape[0]
    #using Gaussian Elimination to figure out the upper triaugular matrix of L
    #in our case, the matrix L must have the inverse matrix becuase it is not a singular matrix
    #make sure the L[ii][ii] != 0
    #ii refers to column
    for ii in range(size):
        nonzeroRow = ii
        run = True
        if L[ii][ii] != 0:
            run = False
        while run == True:
            nonzeroRow = nonzeroRow + 1
            if L[nonzeroRow][ii] != 0:
                run = False
                
        #swap these two rows
        for kk in range(size):
            tmp = L[nonzeroRow][kk]
            L[nonzeroRow][kk] = L[ii][kk]
            L[ii][kk] = tmp
        #swap b
        tmp_b = b[nonzeroRow]
        b[nonzeroRow] = b[ii]
        b[ii] = tmp_b
        
        #make the elements below the L[ii][ii] to be zero
        for ll in range(ii + 1, size):
            factor = -L[ll][ii] / L[ii][ii]
            for mm in range(ii, size):
                if mm == ii:
                    L[ll][mm] = 0
                else:
                    L[ll][mm] += factor * L[ii][mm]
            b[ll] += factor * b[ii]

    #for ref in range(size - 1, 0, -1):
    #    for row in range(ref - 1, 0, -1):
    #        factor = L[row, ref]
    #        for col in range(row - size, size):
    #            L[row,col]=L[row,col] - factor * L[ref,col]
    #            I[row,col]=I[row,col] - factor * I[ref,col]
 
    #solving Ax=b 
    x = [0 for i in range(size)]
    for nn in range(size - 1, -1, -1):
        x[nn] = b[nn] / L[nn][nn]
        for pp in range(nn-1, -1, -1):
            b[pp] -= L[pp][nn] * x[nn]
    
    return x
    
    
#using Gaussian Elimination to calculate out the inverse matrix of L
#in our case, the matrix L must have the inverse matrix becuase it is not a singular matrix
#make sure at every run, L[ii][ii] != 0
def rearrange_matrix(ii, size, L, b):
    maxRow = ii
    maxVal = abs(L[ii][ii])
    for jj in range(ii + 1, size):
        if L[jj][ii] > maxVal:
            maxVal = L[jj][ii]
            maxRow = jj
    #swap these two rows
    for kk in range(ii, size):
        tmp = L[maxRow][kk]
        L[maxRow][kk] = L[ii][kk]
        L[ii][kk] = tmp
    #swap b
    tmp_b = b[maxRow]
    b[maxRow] = b[ii]
    b[ii] = tmp_b
    #make the elements below the L[ii][ii] to be zero
    for ll in range(ii + 1, size):
        factor = -L[ll][ii] / L[ii][ii]
        for mm in range(ii, size):
            if mm == ii:
                L[ll][mm] = 0
            else:
                L[ll][mm] += factor * L[ii][mm]
        b[ll] += factor * b[ii]
    return L, b
#solving Ax=b
#this function is hard to be made parallel by pool tool
def x_solver(nn, L, b):
    #x = [0 for i in range(size)]
    #for nn in range(size - 1, -1, -1):
    x = b[nn] / L[nn][nn]
    for pp in range(nn - 1, -1, -1):
        b[pp] -= L[pp][nn] * x
    return x
    

def wave_function_propagation_sequential(H, initial_wave, total_time, time_interval):
    
    t_points = total_time / time_interval + 1
    
    size = initial_wave.size

    #using Crank-Nicolson methond 
    #assuming Hamiltonian is time-independent in this case
    wave = initial_wave

    L = np.identity(size) + 1j * time_interval / 2 * H
    b_coef = np.identity(size) - 1j * time_interval / 2 * H

    for t in range(1,int(t_points)):
        b = np.dot(b_coef, wave)
        wave = x_sequential(L, b)        
        #wave = np.linalg.solve(L,b)
    return wave

def wave_function_propagation_parallel():
    
    t_points = total_time / time_interval + 1
    
    size = initial_wave.size

    #using Crank-Nicolson methond 
    #assuming Hamiltonian is time-independent in this case
    wave = initial_wave
    
    L1 = np.identity(size) + 1j * time_interval / 2 * H
    size1 = L1.shape[0]
    x1 = [0 for i in range(size1)]
    ii1 = range(size1)
    nn1 = range(size1 - 1, -1, -1)
    b_coef = np.identity(size) - 1j * time_interval / 2 * H

    for t in range(2,point):
        b1 = b_coef * wave

        partial_rearrange = partial(rearrange_matrix,size=size1,L=L1,b=b1 )
        if __name__ == '__main__':
            pool = mp.Pool()
            result = pool.map(partial_rearrange, ii1)
            L2 = result[0][0]
            b2 = result[0][1]
            
            wave = x_solver(L2, b2,size1)
            
            #partial_solver = partial(x_solver, x=x1,L=L2,b=b2) 
            #wave = pool.map(partial_solver, nn1)
            
            pool.close() 
            
    return wave

