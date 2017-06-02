# -*- coding: utf-8 -*-
"""
Created on Mon May 29 19:21:35 2017

@author: sohom
"""

import numpy as np


Tmin=1.0
Tmax=10.0

T=[]
y=[]
 
for i in range(500):
    T.extend([np.log(np.random.uniform(Tmin,Tmax))])
    y.extend([T[i]**5])
#plt.plot(T,y,'o')

#plt.yscale('log')
plt.hist(T)