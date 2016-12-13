from function import x_sequential, x_solver,rearrange_matrix, Hamiltonian_momentum_basis, wave_function_propagation_sequential 
import numpy as np
import multiprocessing as mp
from functools import partial

L1 = np.array([[1.0,0.0,0.0,.0],[.0,5.0,.0,.0],[.0,5.0,-1.0,.0],[.0,0.3,.0,7.0]])
b1 = np.array([3.0,-3.0,-11.0,-5.0])

jb = x_sequential(L1, b1)
print(jb)
size1 = L1.shape[0]
x1 = [0 for i in range(size1)]
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
    #x = x_solver(L2,b2,size1)
    x_rev = x[::-1]
    #x_rev = np.fliplr(x) 

print(x_rev)


answer = np.linalg.solve(L1,b1)
print(answer)

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
print(w[0],w[1],w[2],w[3])
#here, we can pick the first eigenfunction corresponding to the ground energy as our initial wavefunction,
#in that senario, according to time depedent schrodinger equation, the solution is phi(t) = exp(iEt/h_bar) * phi0
#here, phi0 is the first eigenfunction
#thus, we can use this to test if our function works
initial_wave = ground_wave
wave = wave_function_propagation_sequential(H, initial_wave, total_time, time_interval)

import cmath
ref = cmath.exp(-1j) * initial_wave
print(wave)
print(ref)
wave = np.asarray(wave)
import matplotlib.pyplot as plt
plt.plot(range(N),wave.real, range(N), ref.real)
plt.show()
