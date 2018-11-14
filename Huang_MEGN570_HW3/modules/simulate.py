"""Functions and classes for running simulations"""

import numpy as np
import cantera as ct
from wrappers import IndexedVector, CtInterface
from calc import BCFZY_gas_area_density, BCFZY_elyte_area_density, charge_bal_X_elyte, charge_bal_X_ca, charge_bal_dXdt_elyte, charge_bal_dXdt_ca, O_site_density
from analysis import plot_sol
from scipy.integrate import solve_ivp

def derivative(t,SV,SV_indices,params,objs,charge_balance,static,scale=1):
	"""
	Derivative function for solution vector
	
	Parameters:
	----------
	t: time
	SV: initial solution vector
	SV_indices: dict of solution vector indices for IndexedVector creation
	params: dict of parameters
	objs: dict of cantera and wrapper objects
	charge_balance: if True, impose charge balance and track only OH. If false, track all O-site species
	static: list of quantities to hold constant. Any key(s) in SV_indices may be passed
	scale: factor by which to scale i_dl
	
	Returns derivative vector
	"""
	#idea: always use full SV
	#if something is to be held static, simply set dSVdt=0
	#for charge balance, derive dXdt from BCFZY_frac and elyte_frac equations
	#not the most efficient for running, but provides most consistent format
	
	#initialize derivative vector
	dSVdt = np.zeros_like(SV)
	
	#create indexed vectors
	SV = IndexedVector(SV,SV_indices)
	dSVdt = IndexedVector(dSVdt,SV_indices)
	
	#unpack objs
	BCFZY_bulk = objs['BCFZY_bulk']
	elyte = objs['elyte']
	gas_ca = objs['gas_ca']
	wBCFZY_elyte = objs['wBCFZY_elyte']
	wBCFZY_gas = objs['wBCFZY_gas']

	#set BCFZY and elyte potentials from SV
	Phi_BCFZY = SV.get_val('Phi_BCFZY')
	BCFZY_bulk.electric_potential = Phi_BCFZY
	wBCFZY_gas.obj.electric_potential = Phi_BCFZY
	elyte.electric_potential = 0
	
	#set bulk mole fractions from input SV
	#X_OH_ca = SV.get_val('X_ca') #in the future this may be a vector of X_OH and X_Vac
	#X_OH_el = SV.get_val('X_elyte')
	#X_O_ca, X_Vac_ca = BCFZY_frac(X_OH_ca, params['n_Co2'], params['n_Fe2'])
	#X_O_el, X_Vac_el = elyte_frac(X_OH_el, params['n_Ce3'])
	BCFZY_bulk.X = SV.get_val('X_ca')#[X_O_ca, X_OH_ca, X_Vac_ca]
	elyte.X = SV.get_val('X_elyte')#[X_O_el, X_OH_el, X_Vac_el]
	
	#set cathode surface coverages from SV
	#theta_O_ca, theta_OH_ca = SV.get_val('theta_ca')
	#theta_Vac_ca = 1 - (theta_O_ca + theta_OH_ca)
	#print([theta_O_ca, theta_OH_ca, theta_Vac_ca])
	wBCFZY_gas.obj.coverages = SV.get_val('theta_ca')#[theta_O_ca, theta_OH_ca, theta_Vac_ca]
	wBCFZY_gas.obj.X = SV.get_val('theta_ca')#[theta_O_ca, theta_OH_ca, theta_Vac_ca]
	
	#set gas species mass fractions from SV
	rho_gas = SV.get_val('rho_gas')
	gas_density = np.sum(rho_gas)
	Y_gas = rho_gas/gas_density
	#Y_gas = np.concatenate((Y_gas, [1 - np.sum(Y_gas)])) 
	gas_ca.TD = [gas_ca.T, gas_density] #gas_ca.density is not writable
	gas_ca.Y = Y_gas
	
	#BCFZY-gas surface production rates
	BCFZY_gas_ppr = wBCFZY_gas.phase_production_rates
	BCFZY_gas_spr = wBCFZY_gas.species_production_rates
	
	#BCFZY-elyte interface production rates
	BCFZY_elyte_ppr = wBCFZY_elyte.phase_production_rates
	BCFZY_elyte_spr = wBCFZY_elyte.species_production_rates
	
	#No surface species defined for this interface - no thetas to differentiate
	#dSVdt.set_val("index for BCFZY_elyte thetas",BCFZY_elyte_ppr['BCFZY_elyte'][0:SV.len()]*params['gamma_BCFZY_elyte_inv'])
	
	#derivative of BCFZY-gas surface coverages
	dSVdt.set_val('theta_ca', BCFZY_gas_ppr['BCFZY_gas'][0:SV.len('theta_ca')]*params['gamma_BCFZY_gas_inv'])
	
	#derivative of bulk O-site mole fractions
	dXdt_ca = (params['A_ca_gas']*BCFZY_gas_ppr['BCFZY_bulk'] + params['A_cael_ca']*BCFZY_elyte_ppr['BCFZY_bulk'])/(params['e_ca']*params['C_Osite_ca'])
	dXdt_elyte = params['A_cael_el']*BCFZY_elyte_ppr['elyte']/(params['e_el']*params['C_Osite_el'])
	#if applying charge balance, calculate derivatives for O and Vac from dXdt_OH
	if charge_balance==True:
		idx_O_ca = wBCFZY_gas.phase_species['BCFZY_bulk'].index('O(ca_b)')
		idx_OH_ca = wBCFZY_gas.phase_species['BCFZY_bulk'].index('OH(ca_b)')
		idx_Vac_ca = wBCFZY_gas.phase_species['BCFZY_bulk'].index('Vac(ca_b)')
		idx_O_elyte = wBCFZY_elyte.phase_species['elyte'].index('O(elyte_b)')
		idx_OH_elyte = wBCFZY_elyte.phase_species['elyte'].index('OH(elyte_b)')
		idx_Vac_elyte = wBCFZY_elyte.phase_species['elyte'].index('Vac(elyte_b)')
		dXdt_ca[[idx_O_ca,idx_Vac_ca]] = charge_bal_dXdt_ca(dXdt_ca[idx_OH_ca],dndt_Co2=0,dndt_Fe2=0) #assume nCo2 and nFe2 constant
		dXdt_elyte[[idx_O_elyte,idx_Vac_elyte]] = charge_bal_dXdt_elyte(dXdt_elyte[idx_OH_elyte],dndt_Ce3=0) #assume nCe3 constant
		
	dSVdt.set_val('X_ca',dXdt_ca[0:SV.len('X_ca')])
	dSVdt.set_val('X_elyte',dXdt_elyte[0:SV.len('X_elyte')])
	
	#derivative of bulk gas phase species densities
	drho_dt = params['A_ca_gas']*BCFZY_gas_ppr['gas_ca']*params['gas_MW']/(1 - params['e_ca'])
	dSVdt.set_val('rho_gas',drho_dt[0:SV.len('rho_gas')])
	
	#derivative of Phi_BCFZY
	#positive i_Far = negative charge to electrode
	i_Far = -ct.faraday*BCFZY_elyte_spr['OH(ca_b)']# + BCFZY_gas_spr['OH(ca_b)'])  #don't think this should be included - no charge transfer at BCFZY_gas interface
	i_dl = params['i_ext'] - i_Far*params['A_int']
	dSVdt.set_val('Phi_BCFZY', scale*i_dl/(params['C_dl']*params['A_int'])) #d Phi/dt = I_dl/C_dl
	
	#set derivatives for static values to zero
	for key in static:
		dSVdt.set_val(key,0)
	
	#pull vector out after setting all values
	dSVdt = dSVdt.vector
	
	return dSVdt
	
class SimpleIntegrator():
	"""Simple integrator for testing"""
		
	def step(self, fun, t, t_step, y0, **kwargs):
		dydt = fun(t,y0,**kwargs)
		delta = t_step*dydt
		return y0 + delta
	
	def run(self, fun, t_span, t_step, y0, **kwargs):
		times = np.arange(t_span[0],t_span[1],t_step)
		y = np.zeros((len(y0),len(times)))
		y[:,0] = y0
		for i,t in enumerate(times[1:]):
			y[:,i+1] = self.step(fun, t, t_step, y[:,i], **kwargs)
		sol = {'y':y,'t':times}
		return sol
		
class Simulation():
	"""Class for initializing, running, and plotting simulations"""
	
	def __init__(self, derivative_func=derivative):
		"""
		Initialization
		
		Parameters:
		-----------
		derivative_func: solution vector derivative function
		"""
		self.derivative_func = derivative
		#gas molar masses
		self.gas_MW = {'O2':2*15.999, 'H2O': 2*1.008 + 15.999, 'N2': 2*14.07,'AR': 39.948}
		self.runs = {}
		
	def load_phases(self,ctifile):
		"""
		Import phases and create interface wrappers
		
		Parameters:
		-----------
		ctifile: CTI file from which to import
		"""
		self.ctifile = ctifile
		gas_ca = ct.Solution(ctifile,'gas_ca')
		BCFZY_bulk = ct.Solution(ctifile,'BCFZY_bulk')
		transition_metal = ct.Solution(ctifile,'transition_metal')
		elyte = ct.Solution(ctifile,'elyte')

		BCFZY_gas_phases = [gas_ca, BCFZY_bulk, transition_metal]
		BCFZY_gas = ct.Interface(ctifile,'BCFZY_gas',BCFZY_gas_phases)

		BCFZY_elyte_phases = [elyte, BCFZY_bulk]
		BCFZY_elyte = ct.Interface(ctifile,'BCFZY_elyte',BCFZY_elyte_phases)

		#phase and interface dicts
		phase_objs = [gas_ca, BCFZY_bulk, elyte]
		phase_names = [p.name for p in phase_objs]
		self.phases = dict(zip(phase_names,phase_objs))
		self.interfaces = {'BCFZY_gas':BCFZY_gas,'BCFZY_elyte':BCFZY_elyte}
		
	def set_sv_indices(self):
		"Set solution vector names and indices"
		#solution vector names
		self.sv_names = [#electric potentials
			'Phi_cc',
			'Phi_BCFZY',
			'Phi_elyte',
			#gas species densities - rho_k = Y_k*rho_gas
			'rho_O2',
			'rho_H2O',
			'rho_N2',
			'rho_Ar',
			#surface coverages in cathode
			'theta_O_ca',
			'theta_OH_ca',
			'theta_Vac_ca',
			#elyte species mole fractions
			'X_O_el',
			'X_OH_el', #only track protons for now - assume Ce oxidation state fixed
			'X_Vac_el',
			#cathode species mole fractions
			'X_O_ca',
			'X_OH_ca', #only track protons for now - assume TM oxidation state fixed
			'X_Vac_ca'
		   ]

		#set indices for IndexedVector
		self.SV_indices = {}
		self.SV_indices['Phi_cc'] = (self.sv_names.index('Phi_cc'), self.sv_names.index('Phi_cc') + 1)
		self.SV_indices['Phi_BCFZY'] = (self.sv_names.index('Phi_BCFZY'), self.sv_names.index('Phi_BCFZY') + 1)
		self.SV_indices['Phi_elyte'] = (self.sv_names.index('Phi_elyte'), self.sv_names.index('Phi_elyte') + 1)
		self.SV_indices['rho_gas'] = (self.sv_names.index('rho_O2'), self.sv_names.index('rho_Ar') + 1)
		self.SV_indices['theta_ca'] = (self.sv_names.index('theta_O_ca'), self.sv_names.index('theta_Vac_ca') + 1)
		self.SV_indices['X_elyte'] = (self.sv_names.index('X_O_el'), self.sv_names.index('X_Vac_el') + 1)
		self.SV_indices['X_ca'] = (self.sv_names.index('X_O_ca'), self.sv_names.index('X_Vac_ca') + 1)
	
	def set_default_params(self):
		"Set default parameters"
		self.user_params = {}
		
		Phi_BCFZY_init = 0
		T = 773 #temperature (K)
		e_ca = 0.6 #cathode volume fraction (1-porosity)
		e_el = 1 #elyte volume fraction (fully dense)
		r_ca = 100e-9 #BCFZY particle radius (m)
		t_ca = 40e-6 #cathode thickness (m)
		t_el = 15e-6 #elyte thickness (m)
		f_contact = 0.9 #particle contact factor (dimensionless)
		C_dl = 1e-6
		i_ext = 0
		A_int = 1 #cathode-elyte interface area per unit area
		#assume fixed n_Ce3, n_Co, n_Fe initially
		n_Ce3 = 0.7 #fully reduced - max vacancies
		n_Co2 = 0.3 #avg Co ox state = +2.5
		n_Fe2 = 0.1 #avg Fe ox state = +3.5
		
		#user-defined parameters
		self.user_params['Phi_BCFZY_init'] = Phi_BCFZY_init
		self.user_params['T'] = T
		self.user_params['e_ca'] = e_ca
		self.user_params['e_el'] = e_el
		self.user_params['r_ca'] = r_ca
		self.user_params['t_ca'] = t_ca
		self.user_params['t_el'] = t_el
		self.user_params['f_contact'] = f_contact
		self.user_params['C_dl'] = C_dl
		self.user_params['i_ext'] = i_ext
		self.user_params['A_int'] = A_int
		self.user_params['n_Ce3'] = n_Ce3 
		self.user_params['n_Co2'] = n_Co2 
		self.user_params['n_Fe2'] = n_Fe2 
		
		#set calculated parameters
		self.recalc_params()
		
	def recalc_params(self):
		"""
		Recalculate calculated parameters
		Called at the beginning of run() to ensure parameters are updated
		"""
		self.calc_params = {}
		
		self.calc_params['A_ca_gas'] = BCFZY_gas_area_density(self.user_params['e_ca'],self.user_params['r_ca'],self.user_params['f_contact']) #cathode-gas interface area per unit volume
		self.calc_params['A_cael_ca'] = BCFZY_elyte_area_density(self.user_params['t_ca']) #cathode-elyte interface area per unit volume in cathode (only at interface)
		self.calc_params['A_cael_el'] = BCFZY_elyte_area_density(self.user_params['t_el']) #cathode-elyte interface area per unit volume in elyte (only at interface)
		self.calc_params['C_Osite_ca'] = O_site_density(self.phases['BCFZY_bulk'],MW=249.25) #O-site density in BCFZY (kmol/m3)
		self.calc_params['C_Osite_el'] = O_site_density(self.phases['elyte'],MW=318.73) #O-site density in BCZYYb (kmol/m3)
		self.calc_params['gamma_BCFZY_gas_inv'] = 1/self.interfaces['BCFZY_gas'].site_density
		#self.calc_params['gamma_BCFZY_elyte_inv'] = 1/BCFZY_elyte.site_density #don't need this- no surface species
		#vector of gas molar masses - ensure order matches phase order
		self.calc_params['gas_MW'] = [self.gas_MW[s.name] for s in self.phases['gas_ca'].species()] 
		
		#update sim params 
		self.sim_params = self.user_params.copy()
		self.sim_params.update(self.calc_params)
	
	def initialize(self,ctifile,set_params=True):
		"""
		Initialize default parameters and load phases
		
		Parameters:
		-----------
		ctifile: CTI file from which to import phases and interfaces
		set_params: if True, set default parameters. If False, not parameters will be set
		"""
		#load phases
		self.load_phases(ctifile)
		
		#equilibrate coverages
		#self.interfaces['BCFZY_gas'].advance_coverages(100000)
		#self.interfaces['BCFZY_elyte'].advance_coverages(100000)
		
		#set parameters
		if set_params is True:
			self.set_default_params()
		
		#simulation objects
		self.sim_objs = self.phases.copy()
		
		#interface wrappers
		wBCFZY_gas = CtInterface(self.interfaces['BCFZY_gas'])
		wBCFZY_elyte = CtInterface(self.interfaces['BCFZY_elyte'])
		self.sim_objs['wBCFZY_gas'] = wBCFZY_gas
		self.sim_objs['wBCFZY_elyte'] = wBCFZY_elyte
		
		#solution vector names & indices
		self.set_sv_indices()

	def init_SV(self, charge_balance):
		"""
		Get initial solution vector
		Determined from sim_params and current state of phases and interfaces
		
		Parameters:
		-----------
		charge_balance: if True, impose charge balance
		
		Returns initial solution vector
		"""
		SV_0 = np.zeros_like(self.sv_names,dtype=float)

		#IndexedVector for easier value setting
		SV_0 = IndexedVector(SV_0,self.SV_indices)
		SV_0.set_val('Phi_BCFZY',self.sim_params['Phi_BCFZY_init']) #initial cathode potential
		
		#if imposing charge balance, set initial O and Vac fractions from X_OH
		if charge_balance==True:
			#this is ugly but seems safest to ensure correct positions
			#maybe write set_species_X method for CtInterface
			wBCFZY_elyte = self.sim_objs['wBCFZY_elyte']
			idx_O_ca = wBCFZY_elyte.phase_species['BCFZY_bulk'].index('O(ca_b)')
			idx_OH_ca = wBCFZY_elyte.phase_species['BCFZY_bulk'].index('OH(ca_b)')
			idx_Vac_ca = wBCFZY_elyte.phase_species['BCFZY_bulk'].index('Vac(ca_b)')
			idx_O_elyte = wBCFZY_elyte.phase_species['elyte'].index('O(elyte_b)')
			idx_OH_elyte = wBCFZY_elyte.phase_species['elyte'].index('OH(elyte_b)')
			idx_Vac_elyte = wBCFZY_elyte.phase_species['elyte'].index('Vac(elyte_b)')
			X_OH_ca = self.phases['BCFZY_bulk'].X[idx_OH_ca]
			X_ca = self.phases['BCFZY_bulk'].X.copy()
			X_ca[[idx_O_ca,idx_Vac_ca]] = charge_bal_X_ca(X_OH_ca, self.sim_params['n_Co2'],self.sim_params['n_Fe2'])
			self.phases['BCFZY_bulk'].X = X_ca
			X_OH_elyte = self.phases['elyte'].X[idx_OH_elyte]
			X_elyte = self.phases['elyte'].X.copy()
			X_elyte[[idx_O_elyte,idx_Vac_elyte]] = charge_bal_X_elyte(X_OH_elyte,self.sim_params['n_Ce3'])
			self.phases['elyte'].X = X_elyte
			
		SV_0.set_val('X_elyte', self.phases['elyte'].X) 
		SV_0.set_val('X_ca', self.phases['BCFZY_bulk'].X) 
		SV_0.set_val('theta_ca',self.interfaces['BCFZY_gas'].coverages) 
		SV_0.set_val('rho_gas', self.phases['gas_ca'].density*self.phases['gas_ca'].Y[0:SV_0.len('rho_gas')]) 

		#pull vector back out of IndexedVector
		SV_0 = SV_0.vector
		
		return SV_0
	
	def run(self, t_span, charge_balance, static=[], SV_0=None, scale=1, method='BDF', save_as=None,**integrator_kw):
		"""
		run simulation over t_span
		
		Parameters:
		-----------
		t_span: time span over which to run [start, end]
		charge_balance: if True, impose charge balance and track only OH. If false, track all O-site species
		static: list of quantities to hold constant. Any key(s) in SV_indices may be passed
		SV_0: initial solution vector. If None, SV_0 will be determined from sim_params and current state of phases and interfaces
		scale: factor by which to scale i_dl
		method: solve_ivp method
		save_as: if string is passed, run results and settings will be stored in runs dict with save_as as key. If None, run results will not be saved
		integrator_kw: kwargs to pass to `solve_ivp`
		
		Returns solve_ivp result
		"""
		#recalculate sim_params in case any user-defined params have been changed
		self.recalc_params()
		#if no starting SV provided, initialize default
		if SV_0 is None:
			SV_0 = self.init_SV(charge_balance)
		sol = solve_ivp(lambda t, y: self.derivative_func(t, y, self.SV_indices, self.sim_params,self.sim_objs, charge_balance, static, scale=scale),
				t_span, SV_0, method=method, **integrator_kw)
		
		#save run results if specified
		if save_as is not None:
			run_data = {'sol':sol,'params':self.sim_params,'sv_names':self.sv_names,'charge_balance':charge_balance,'static':static,'cti_file':self.ctifile}
			self.runs[save_as] = run_data
			
		return sol
	
	def plot_runs(self,run_ids=None,axes=None, labels=None, **kw):
		"""
		Plot time evolution of saved run data
		
		Parameters:
		-----------
		run_ids: dict keys of runs to plot. If None, all saved runs will be plotted
		axes: axes on which to plot. If None, new axes will be created
		labels: list of labels for runs. If None, dict keys will be used as labels
		kw: kwargs to pass to `plot_sol`
		
		Returns axes
		"""
		#if no runs specified, plot all saved runs
		if run_ids is None:
			run_ids = self.runs.keys()
		#if no labels given, use run_ids
		if labels is None:
			labels = run_ids
		for run_id,label in zip(run_ids,labels):
			run_data = self.runs[run_id]
			axes = plot_sol(run_data['sol'],run_data['sv_names'],run_data['params'],axes=axes,label=label, **kw)
		return axes
		
	def clear_runs(self):
		"Clear saved run data"
		self.runs = {}