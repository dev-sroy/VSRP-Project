# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 12:46:47 2017

@author: sohom
"""

import mockdeclare

for z in np.arange(0.05,1.05,0.1):
    thetaI = []
    T=[]
    thetaIlin = []
    for i in range(num_cluster):
        Tx = np.random.uniform(Tmin,Tmax)
        thetaI.extend([np.random.normal(RI(Tx,z),thetascat)/DA(z)])
        T.extend([Tx])
    plt.plot(T,thetaI,'o',markersize=2,c=RGB_tuples[k])
    for i in range(num_cluster):    
        thetaIlin.extend([RI(Tlin[i],z)/DA(z)])
    plt.plot(Tlin,thetaIlin,c=RGB_tuples[k],label='z = '+str(z))
    k=k+1
    
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$T_X$(in keV)')
plt.ylabel(r'$\theta_I$')
plt.legend(loc='center left', bbox_to_anchor=(1,0.5))
plt.show()