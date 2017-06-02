# This file makes the correct dndz, dndM and sigmaM plots
# A separate 'constants.py' file is attached from which values of constants are imported.

import random
import constants
import math
import numpy as np
import scipy as sp
from scipy import integrate
from scipy.integrate import quad
from scipy.integrate import simps
from scipy.integrate import odeint
import scipy.optimize
from scipy import interpolate 
import matplotlib.pyplot as plt
import sys

# Input parameters (same as dndz.c)
omegam_ = 0.3	
omegar_ = 0.0
omegab_ = 0.04734	# Default 0.04?
omegav_ = 1.0 - omegam_ #Assumed a flat universe
omega_ = 1.0
w0_ = -1.0
w1_ = 0.0	# Not added till now
h_ = 0.65
specindex_ = 1.0
sigma8_ = 1.0	
A_ = 10.29		
alpha_ = 1.7 	
gamma_ = 0.64 
dNorm = 1.0 
dNorm2 = 1.0
sNorm = 1.0
sNorm2 = 1.0

#--- Functional form of w w(a) = w0 + w1(1-a) ---*/
def WDE(a):
	return (w0_ + w1_*(1.0-a))


#--- distance/volume formulas from "Distance measures in cosmology", David W. Hogg, astro-ph/9905116 ---*/

#--- Calculates Hubble ratio E(z), (14) ---
def Ez(z): 
	x = 1.0+z
	return (math.sqrt(x*x*((1.0 - omega_) + x*(omegam_ + x*omegar_)) + omegav_*pow(x,3.0*(1.0+w0_+w1_))*math.exp(3.0*w1_*(1.0/x-1.0)) ))


#--- F(z)=1/E(z) ---*/
def Fz(z): 
	return (1.0/Ez(z))


#--- Hubble constant at redshift z ---*/
def Hz(z):
	return (100.0*h_*Ez(z))


#--- Comoving distance (line-of-sight), (15) ---*/
def DC(z):
	temp = integrate.romberg(Fz, 0, z)
	if (z< 1.e-4):	return (0.0)
	else:	return ((constants.ckms/(100*h_))*temp)


#--- Comoving distance (tranverse), (16) ---*/
def DM(z):
	if(omega_ == 1.0):	
		dm =  DC(z)
	elif(omega_ < 1.0):	
		dm =  (constants.ckms/(100.0*h_))*(1.0/math.sqrt(1.0-omega_))*math.sinh(math.sqrt(1.0-omega_)*DC(z)*(100.0*h_)/constants.ckms)
	elif(omega_ > 1.0):	
		dm =  (constants.ckms/(100.0*h_))*(1.0/math.sqrt(omega_-1.0))*math.sin(math.sqrt(omega_-1.0)*DC(z)*(100.0*h_)/constants.ckms)
	else:	
		dm =  -1.0	# should never reach this point */
	return (dm)


#--- Angular diameter distance, (17) ---*/
def DA(z):
	return (DM(z)/(1.0+z))


#--- Luminosity distance, (21) ---*/
def DL(z):
	return (DM(z)*(1.0+z))

#--- Comoving volume element dV/dz/dOmega, with Omega in square degrees, (28)  #Why degrees? ---*/
def covol(z):
	dm = DM(z)
	return ((constants.PI*constants.PI/32400.0)*(constants.ckms/(100.0*h_))*dm*dm/Ez(z))
# 	return ((constants.ckms/(100.0*h_))*dm*dm/Ez(z)) #Radians


#--- Omegam(z)/omegam_ = (1+z)^3/E(z)^2---*/ 
def Omegamz(z):
	x = 1.0+z
	y = Ez(z)
	return (omegam_*x*x*x/y/y)
	

#--- Returns derivatives for differential equation ---*/		
def derivs(y,a):
	z = (1.0/a)-1.0
	omegamz = Omegamz(z)
	return [y[1], ((-1.5/a)*((1.0-WDE(a)*(1.0-omegamz))*y[1] - (omegamz/a)*y[0]))]

#--- Numerically solves perturbation differential equation (as given by Battye and Weller, Phys. Rev. D 68 083506), must call Dnorm first to ensure proper normalization ---*/
def Dsolve(z):
	a = 1.0/(1.0+z)
	x1 = 0.01
	da = [x1, a]	
	ystart = [x1, 1.0]
	y = odeint(derivs,ystart,da,h0=0.001, hmin=0.0, rtol=None, atol=1.0e-3)
	ans = y[-1][0]
	return (ans/dNorm)
#Doubt :Derivs and DSolve (use of odeint) 
#Numerical Recipes in C	
#--- Sets global variable dNorm = D(a=1), used to normalize D(a) ---*/
# This subroutine is only called once for each set of parameters (should be called immediately after setparam(parameters) */
def Dnorm():
	global dNorm
	dNorm = 1.0 
	dNorm2 = Dsolve(0.0)
	dNorm = dNorm2
Dnorm()

#--- Spherical top hat window function ---*/
def window(x):
	return (3.0*(math.sin(x)/x - math.cos(x))/x/x)


def Tr_BBKS(k):
	q = k/omegam_/h_/h_*math.exp(omegab_*(1.0 + math.sqrt(2.0*h_)/omegam_))	# shape parameter for non-zero bayron density 
	return (math.log(1.0 + 2.34*q)/(2.34*q))*pow(1.0 + q*(3.89 + q*(259.21 + q*(163.667323 + q*2027.16958081))),-0.25) # BBKS transfer function


#--- Computes the integrand of sigmaR using the BBKS fitting form with an exponential correction for omega_baryon ---*/
def sigmaint(k):
	T = Tr_BBKS(k)				# BBKS transfer function 
	powerspec = pow(k,specindex_)*T*T	# linear power spectrum P(k) 
	deltasq = 4.0*constants.PI*k*k*k*powerspec
	W = window(k*R_)
	return (W*W*deltasq/k)


#--- Returns sigmaR, not normalized must call Snorm first to normalize to sigma8 ---*/
def sigmaR(R):
    global R_
    R_ = R
    temp = quad(sigmaint, 0.0, sp.inf)[0]
    return (math.sqrt(temp)/sNorm)
    


#--- Normalizes sigmaR such that sigma(R=8/h Mpc) = sigma8 ---*/
def Snorm():
	global sNorm
	sNorm = 1.0
	sNorm2 = sigmaR(8.0/h_)/sigma8_
	sNorm = sNorm2
Snorm()	


#--- Returns sigma(M) ---*/
def sigmaM(M):
	# rho0(M_solar/Mpc^2) = 2.775179934e11*omegam*h^2 */
	R_M = pow(0.75*M/constants.PI/omegam_/h_/h_/constants.rhoconstant,1.0/3.0)	# radius corresponding to mass M */
	return (sigmaR(R_M))

#--- Uses spline to calculate second derivatives of sigmaM for splint, stored in global vector y2 ---*/
def sigmaMspline():
	
	global tck
	xa = np.arange(constants.Nspl, dtype=np.float)
	ya = xa.copy()
	print "spline start"
	for i in range(0, constants.Nspl):
		xa[i] = pow(10.0, (12. + (4.2*i)/constants.Nspl))/h_
		ya[i] = sigmaM(xa[i])
	tck = interpolate.splrep(xa, ya, s=0)
	print "spline formed"
sigmaMspline()

#--- Calculates comoving number density (dn/dM)dM
#J Estimated Initial step-size ignored
#J Sheth and Tormen mass function
def dndM(M):
	DE = Dsolve(z_)	
	sM = interpolate.splev(M, tck, der=0)

#	Mlim_scatter = 3.e-1*ml_ 
	
	# IMP : 1) NEVERS put Mlim_scatter = 0.0, 
	#		2) If scatter is higher increase accuracy in qromo.c 
	
#	selection_fn = 0.5*(1. + math.erf((M-ml_)/Mlim_scatter))	
	
#Haiman dn/dM function
	checking = math.fabs((0.315*constants.rhoconstant*omegam_*h_*h_/M)*(1.0/sM)*(interpolate.splev(M, tck, der=1))*math.exp(-pow(math.fabs(0.61-math.log(sM*DE)),3.8)))

#	return selection_fn*checking 
	return checking


#--- given mass m_inp enclosed in r_x with overdensity Delta_inp at redshift = z this returns ---*/
#    mass enclosed in r_y with overdensity Delta_out, assuming a NFW density profile.            */
def mass_convert(m_inp, Delta_inp, Delta_out, z):
	rho_c = constants.rhoconstant*h_*h_
	
	
#	variable conc following Dalal etal, although no "scatter" is taken, scatter can be taken similarly
#	like in Mlim .*/
         
	conc = 6.0
	deltac_bryan_norman = 18.*constants.PI*constants.PI + 82.*(Omegamz(z)-1.) - 39.*pow((Omegamz(z)-1.),2.)
	global deltac_nfw
	deltac_nfw = deltac_bryan_norman*pow(conc,3.0)/3.0	
	deltac_nfw = deltac_nfw/(math.log(1.0+conc)-conc/(1.0+conc)) 
	

#	/* the only z-dependence */
	if(z>1.0e-3):	rho_cz = pow((Ez(z)/Ez(0.0)),2.0)*rho_c
	else:	rho_cz = rho_c.copy()

	r_inp = pow(3.0*m_inp/(4.0*constants.PI*rho_cz*Delta_inp),1.0/3.0)
	
	global Delta_nfw
	Delta_nfw = Delta_inp
	if( (r_solver(1.0)*r_solver(10.0))<0 ):	
		tmp = scipy.optimize.bisect(r_solver, 1.0, 10.0, xtol=1e-10)
	else:
		print " r_solver root finding error"
		exit()
	l_inp = tmp.copy()

	Delta_nfw = Delta_out
	if( (r_solver(1.0)*r_solver(10.0))<0 ):		
		tmp = scipy.optimize.bisect(r_solver, 1.0, 10.0, xtol=1e-10)
	else:
		print " r_solver root finding error"
		exit()
	l_out = tmp.copy()

	r_out = r_inp*l_out/l_inp
                                                                                                                                                                  
	return (4.0*constants.PI*rho_cz*pow(r_out,3.0)*Delta_out/3.0)


def r_solver(r):
	tmp = 3.0*deltac_nfw*(math.log(1.0+r)-r/(1.0+r))
	tmp /= pow(r,3.0)
	return (tmp-Delta_nfw)


def M_lim(z,  flux_sz, sz_freq, MTscat, Ascat, alphascat):

#	/* adds random scatter to A and alpha */
	if (MTscat):
		A_temp = A_*(1.0 + Ascat*np.random.normal(0.0, 1.0))
		alpha_temp = alpha_*(1.0 + alphascat*np.random.normal(0.0, 1.0))
	else:
		A_temp = A_.copy()
		alpha_temp = alpha_.copy()

#	/*-------------------------------------*/
#	/* For Optical survey :  ans is mass in M200 from Yee & Ellingson */

	ans = pow(10., A_temp)*pow(flux_sz, alpha_temp)*pow((1.+z),gamma_)
					
#	/* converts mass from M_200 to M_jenkins */
	ans = mass_convert(ans,200.0,180.0*Omegamz(z),z)
	
#	/*hardwired mass cutoff! Note: these cutoff's are also hardwired in MCMC.c, so that new parameters are generated
#	if Mlim goes out of range...this saves computational cost*/
	if (ans < 1.0e14):	ans = 1.0e14  #this fixed lim cannot be more than that at MCMC.c
	if (ans > 9.9e15):	ans = 9.9e15 #this fixed lim cannot be less than that at MCMC.c
		
	return (ans)


#--- Returns dN/dz/dOmega (per square degree)  ----------------------------***************
def dndz(z):
	
	global z_, ml_
	z_ = z
#	ml = M_lim(z_,flux_,freq_,MTscat_,Ascat_,alphascat_)
	lowmass = 1.e13
	ml = lowmass	##!
		#IMP:  1) when using scatter, lowmass has to be lower than Mlim 
		#  	2)Also lowmass mut be higher than lowest limit on MSplie ~ 1.5e12 
		#	3) If low mass is smaller than 1e13m check to see if accuracy needs to incraesed */
		
				
	if(ml < lowmass):
		print "lower lowmass in dndz integral"
		exit()
        
	ml_ = ml
	temp = quad(dndM, 1.e14, 1.2e16)[0]
	return (temp*covol(z_))
	

# Below there are 3 sets of codes just for plotting sigmaM, dndM & dndz individually
#Plotting sigmaM -----------------------------------------
plt.figure()
xa = np.arange(constants.Nspl, dtype=np.float)
ya = xa.copy()
for i in range(0, constants.Nspl):
	xa[i] = pow(10.0, (12. + (4.2*i)/constants.Nspl))/h_
	ya[i] = sigmaM(xa[i])
points = 200
xb = np.arange(points, dtype=np.float)
yb = xb.copy()
yc = xb.copy()
for i in range(0, points):
	xb[i] = pow(10.0, (12. + (4.2*i)/points))/h_
	yb[i] = interpolate.splev(xb[i], tck, der=0)
plt.plot(xa, ya, 'x', xb, yb, 'r')
plt.legend(['Actual','Interpolated'], loc=1)
plt.grid()
plt.xlabel('M')
plt.ylabel('$\sigma_M$')
plt.savefig("sigmaM.pdf")
#END of plotting sigmaM --------------------------------------

#dndM plots-----------------------------------------
z_ = 0.0
points = 20
plt.figure()
xa = np.arange(points, dtype=np.float)
ya = xa.copy()
yb = xa.copy()
for i in range(0, points):
	xa[i] = pow(10.0, (12.8 + (3.5*i)/points))
	ya[i] = dndM(xa[i])
plt.plot(xa, ya, 'r')
plt.legend(['$z = 0.0$'], loc=1)
plt.grid()
plt.xlabel('M')
plt.ylabel('$dndM$')
plt.xscale('log')
plt.yscale('log')
plt.savefig("dndM.pdf")
#END dndM plots--------------------------------------


##dndz plots---------------------------
plt.figure()
xa = np.arange(0.05, 3.01, 0.1) 
ya = xa.copy()
Dnorm()
Snorm()
sigmaMspline()
for i in range(len(xa)):
	ya[i] = (dndz(xa[i])/12.)	
plt.plot(xa, ya, 'b')
plt.legend([ '$w_0 = -1.0$'], loc=1)
plt.grid()
plt.xlabel('$z$')
plt.ylabel('$dN/dz/12$')
plt.savefig("plot.pdf")
#For saving the file
#np.savetxt('dndm.dat', np.c_[xa,ya,yb])
## dndz plots----------------------------


# Only some parts of the following code are converted from C to python
# This code wasn't used so didn't convert
# #--- Gives total dndz over (z,z+delta)---*/
# def DN( z, delta, area, flux, freq, MTscat, Ascat, alphascat):
# 	global flux_, freq_, MTscat_, Ascat_, alphascat_ 
# 	flux_ = flux.copy()
# 	freq_ = freq.copy()
# 	MTscat_ = MTscat.copy()
# 	Ascat_ = Ascat.copy()
# 	alphascat_ = alphascat.copy()
# 		
# #		/*only for RCS, can neglect integral is data is beyond z=0.2 */
# 	return (area*delta*dndz(z+delta/2.0))	
# #		/*return (area*dn_qromb(dndz,z,z+delta))*/ 
# 
# 
# #--- gives dn/dz in area, rounded to an integer ---*/
# # a math.sqrt(N) poisson error can be added, as well as scatter in the mass-flux relation #D-scatter?
# def DNint( z, delta, area, flux, freq, MTscat, Ascat, alphascat, DNscat):  
# 	dn = DN(z,delta,area,flux,freq,MTscat,Ascat,alphascat)
# 	
# 	if(DNscat):
# 		dn += math.sqrt(dn)*np.random.normal(0.0, 1.0)
# 	
# 	return (rint(dn))

	
	
