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
    
    
def get_data(filename):
    #TODO: needs to be implemented in a good and robust way
    data = np.loadtxt(filename)
    x = data[:,0]
    y = data[:,1]
    return x, y
    

    def load_file(self, filepath):
        """ loads the data from a file that has a 2 column structure
            ignores the first linestoignore lines of this file """
        x, y, xx, yy = self.read_data(filepath)
        return x, y, xx, yy
        
    
    def read_data(self, filepath):
        """importing data from a file"""
        isAlternating = False
        x = []
        y = []
        xx = []
        yy = []
        inp = open (filepath,'rb')
        check = inp.readline()  
        check_new = check.split()
        if len(check_new) > 5:
            isAlternating = True            
        inp.seek(0)       #moves the cursor back to the beginning of the file
        i = 0
        for line in inp:
            i += 1
            if i > self.linestoignore:
                numbers = map(float, line.split())
                x.append(numbers[0])          # now x is in microseconds
                y.append(numbers[1]*1e3)                    
                if isAlternating:
                    if check_new[3] == 'Fit':
                        xx.append(numbers[3])     # now xx is in microseconds
                        yy.append(numbers[4]*1e3)                        
                    else:
                        xx.append(numbers[2])     # now yy is in microseconds
                        yy.append(numbers[3]*1e3)
        inp.close()
        #checken, ob  len(x) = len(xx)
        return x, y, xx, yy