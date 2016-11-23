# -*- coding: utf-8 -*-

import numpy as np


def do_normalisation(y_raw, rabi_amplitude, rabi_offset):
    '''data normalization'''
    y_norm = 1-(rabi_offset + rabi_amplitude - y_raw)/(2*rabi_amplitude)
    return y_norm


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


def calculate_depth_simple(mu_0, h, B, gamma_n, rho):
    rho *= 1e27
    '''gets rho in 1/nm^3 and B in T, returns d in nm'''
    d = (rho * mu_0 * mu_0 * h * h * gamma_n * gamma_n * 5 / (1536 * np.pi * B * B))**(1./3)
    d = d*1e9
    return d


def calculate_density_simple(mu_0, h, B, gamma_n, d):
    '''gets d in nm and B in T, returns rho in 1/nm^3'''
    d /= 1e9
    rho = d**3 * 1536 * np.pi * B * B / (5 * mu_0 * mu_0 * h * h * gamma_n * gamma_n)
    rho /= 1e27
    return rho
    
    
def get_data(filename, data_columns):
    '''importing data with two or three columns'''
    data = np.loadtxt(filename)
    x = data[:,0]
    y1 = data[:,1]
    if data_columns == 3:
        y2 = data[:,2]
        return x, y1, y2
    return x, y1