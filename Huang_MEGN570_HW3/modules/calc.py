"""Basic calculations"""

import numpy as np

Avogadro = 6.022214e23


def elyte_net_charge(X_O,X_OH,n_Ce3):
	"""
	Calculate net charge of electrolyte
	
	Parameters:
	-----------
	X_O: mole fraction O
	X_OH: mole fraction OH
	n_Ce3: formula units Ce3+ (k in formulas). Limits: (0,0.7)
	"""
	cat_charge = 2 + n_Ce3*3 + (0.7-n_Ce3)*4 + 0.1*4 + 0.2*3
	O_charge = 3*(X_O*(-2) + X_OH*(-1))
	
	return cat_charge + O_charge

def ca_net_charge(X_O,X_OH,n_Co2,n_Fe2):
	"""
	Calculate net charge of cathode
	
	Parameters:
	-----------
	X_O: mole fraction O
	X_OH: mole fraction OH
	n_Co2: formula units Ce2+ (k in formulas). Remainder effectively assigned to 4+. Limits: (0,0.4)
	n_Fe2: formula units Fe2+ (l in formulas). Remainder effectively assigned to 4+. Limits: (0,0.4)
	"""
	cat_charge = 2 + n_Co2*2 + (0.4-n_Co2)*4 + n_Fe2*2 + (0.4-n_Fe2)*4 + 0.1*4 + 0.1*3
	O_charge = 3*(X_O*(-2) + X_OH*(-1))
	
	return cat_charge + O_charge

def charge_bal_X_elyte(X_OH,n_Ce3):
	"""
	Determine elyte (BCZYYb) bulk species fractions assuming charge balance constraints
	
	Parameters:
	-----------
	X_OH: fraction OH (z)
	n_Ce3: formula units Ce3+ (k in formulas). Limits: (0,0.7)
	
	Returns X_O (x), X_Vac (y)
	"""
	#formula units
	n_O = 2.9 - (n_Ce3 + X_OH*3)/2
	n_Vac = 0.1 + (n_Ce3 - X_OH*3)/2
	
	return n_O/3, n_Vac/3 #divide by 3 for mole fractions
	
def charge_bal_dXdt_elyte(dXdt_OH,dndt_Ce3):
	"""
	Determine elyte (BCZYYb) bulk species fractions derivatives assuming charge balance constraints
	
	Parameters:
	-----------
	dXdt_OH: d/dt of X_OH (z)
	dndt_Ce3: d/dt of formula units Ce3+ (k in formulas). Limits: (0,0.7)
	
	Returns dXdt_O (x), dXdt_Vac (y)
	"""
	#formula units
	dndt_O = (dndt_Ce3 + dXdt_OH*3)/2
	dndt_Vac = (dndt_Ce3 - dXdt_OH*3)/2
	
	return dndt_O/3, dndt_Vac/3 #divide by 3 for mole fractions
	
def charge_bal_X_ca(X_OH, n_Co2, n_Fe2):
	"""
	Determine BCFZY bulk species fractions assuming charge balance constraints
	
	Parameters:
	-----------
	X_OH: fraction OH (z)
	n_Co2: formula units Ce2+ (k in formulas). Remainder effectively assigned to 4+. Limits: (0,0.4)
	n_Fe2: formula units Fe2+ (l in formulas). Remainder effectively assigned to 4+. Limits: (0,0.4)
	
	Returns X_O (x), X_Vac (y)
	"""
	#formula units
	n_O = 2.95 - n_Co2 - n_Fe2 - X_OH*3/2
	n_Vac = 0.05 + n_Co2 + n_Fe2 - X_OH*3/2
	
	return n_O/3, n_Vac/3 #divide by 3 for mole fractions
	
def charge_bal_dXdt_ca(dXdt_OH, dndt_Co2, dndt_Fe2):
	"""
	Determine BCFZY bulk species fractions assuming charge balance constraints
	
	Parameters:
	-----------
	dXdt_OH: d/dt of X_OH (z)
	dndt_Co2: d/dt of formula units Ce2+ (k in formulas). Remainder effectively assigned to 4+. Limits: (0,0.4)
	dndt_Fe2: d/dt of formula units Fe2+ (l in formulas). Remainder effectively assigned to 4+. Limits: (0,0.4)
	
	Returns dXdt_O (x), dXdt_Vac (y)
	"""
	#formula units
	dndt_O =  -dndt_Co2 - dndt_Fe2 - dXdt_OH*3/2
	dndt_Vac = dndt_Co2 + dndt_Fe2 - dXdt_OH*3/2
	
	return dndt_O/3, dndt_Vac/3 #divide by 3 for mole fractions
	
	
def BCFZY_gas_area_density(e_ca,r_ca,f_contact):
	"""
	BCFZY-gas interface area per unit volume
	
	Parameters:
	-----------
	e_ca: cathode volume fraction
	r_ca: BCFZY particle radius
	f_contact: particle contact factor
	"""
	
	return 3*e_ca*f_contact/r_ca
	
def BCFZY_elyte_area_density(delta_z):
	"""
	BCFZY-elyte interface area per unit volume
	Only applies to control volumes at BCFZY-elyte interface
	
	Parameters:
	-----------
	delta_z: control volume thickness
	"""
	
	return 1/delta_z
	
def O_site_density(phase,MW,n_O=3):
	"""
	O-site density (kmol/m3) of phase
	
	Parameters:
	-----------
	phase: cantera phase object
	MW: molar mass of phase (kg/kmol)
	n_O: formula units of O
	"""
	
	return n_O*phase.density/MW
	

	