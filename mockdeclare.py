# -*- coding: utf-8 -*-
"""
Created on Mon May 29 11:46:27 2017

@author: sohom
"""

import scipy as sp
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import integrate
import colorsys
import time

thetab = 5  
Amean = 0.31
alphamean = 0.70
gammamean = 0.08
Aerr = 0.02
alphaerr = 0.07
gammaerr = 0.17
thetascat = 0.1
z = 2
omega_=1.0
omegam_ = 0.3
omegar_ = 0.0
omegab_ = 0.04734
omegav_ = 1.0 - omegam_ 
w0_ =-1.0
w1_=0.0 
ckms = 299792.458 
num_cluster = 1000
N=20
h_ = 0.65

Tmin = 1.2
Tmax = 8.5
Tscat = 0.1

Tlin = np.linspace(Tmin,Tmax,num_cluster)

A = np.random.normal(Amean,Aerr)
alpha = np.random.normal(alphamean,alphaerr)
gamma = np.random.normal(gammamean,gammaerr)

def Ez(z): 
	x = 1.0+z
	return (math.sqrt(x*x*((1.0 - omega_) + x*(omegam_ + x*omegar_)) + omegav_*pow(x,3.0*(1.0+w0_+w1_))*math.exp(3.0*w1_*(1.0/x-1.0)) ))

def Fz(z): 
	return (1.0/Ez(z))

def Hz(z):
	return (100.0*h_*Ez(z))
 
def DC(z):
	temp = integrate.romberg(Fz, 0, z)
	if (z< 1.e-4):	return (0.0)
	else:	return ((ckms/(100*h_))*temp)
 
def DA(z):
     return (DC(z)/(1.0+z))
     
def RI(T,z):
    R = A*T**alpha*(1+z)**gamma
    return R

HSV_tuples = [(x*1.0/N, 1.0, 1.0) for x in range(N)]
RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
k=0


