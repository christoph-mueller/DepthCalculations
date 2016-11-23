# -*- coding: utf-8 -*-

import numpy as np


def nu_from_tau(x):
    '''calculates the frequency nu for given values of tau'''
    nu = []
    tau = np.array(x)
    nu = 1/(2*tau)
    return nu


def S_from_data(tau, y, gamma_e, N):
    '''calculates the spectral density S for given coherence data
       N is the number of pulses'''
    S = []
    tau = np.array(tau)
    y = np.array(y)
    chi = -np.log((2*y)-1)
    S = 1/(gamma_e*gamma_e) * (np.pi*np.pi)/(4*N*tau) * chi
    return S


def calculate_depth_simple(B, rho):
    '''gets B in T, returns d in nm'''
    mu_0 = 1.256e-6
    mu_p = 2*2.79*5.0507866*1e-27
    d = (rho * mu_0 * mu_0 * mu_p * mu_p * 5 / (1536 * np.pi * B * B))**(1./3)
    d = d*1e9
    return d
    
    
def get_data(filename, data_columns):
    #TODO: needs to be implemented in a good and robust way
    data = np.loadtxt(filename)
    x = data[:,0]
    y1 = data[:,1]
    if data_columns == 3:
        y2 = data[:,2]
        return x, y1, y2
    return x, y1