import numpy as np
import cantera as ct
from wrappers import IndexedVector
from calc import BCFZY_frac, elyte_frac

def derivative(t,SV,SV_indices,params,objs,scale=1):
	"""Derivative function for solution vector"""
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
	X_OH_ca = SV.get_val('X_ca') #in the future this may be a vector of X_OH and X_Vac
	X_OH_el = SV.get_val('X_elyte')
	X_O_ca, X_Vac_ca = BCFZY_frac(X_OH_ca, params['n_Co2'], params['n_Fe2'])
	X_O_el, X_Vac_el = elyte_frac(X_OH_el, params['n_Ce3'])
	BCFZY_bulk.X = [X_O_ca, X_OH_ca, X_Vac_ca]
	elyte.X = [X_O_el, X_OH_el, X_Vac_el]
	
	#set cathode surface coverages from SV
	#theta_O_ca, theta_OH_ca = SV.get_val('theta_ca')
	#theta_Vac_ca = 1 - (theta_O_ca + theta_OH_ca)
	#print([theta_O_ca, theta_OH_ca, theta_Vac_ca])
	wBCFZY_gas.obj.coverages = SV.get_val('theta_ca')#[theta_O_ca, theta_OH_ca, theta_Vac_ca]
	wBCFZY_gas.obj.X = SV.get_val('theta_ca')#[theta_O_ca, theta_OH_ca, theta_Vac_ca]
	
	#set gas species mass fractions from SV
	rho_gas = SV.get_val('rho_gas')
	Y_gas = rho_gas/gas_ca.density
	#Y_gas = np.concatenate((Y_gas, [1 - np.sum(Y_gas)])) 
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
	dXca_dt = (params['A_ca_gas']*BCFZY_gas_ppr['BCFZY_bulk'] + params['A_cael_ca']*BCFZY_elyte_ppr['BCFZY_bulk'])/(params['e_ca']*params['C_Osite_ca'])
	dXel_dt = params['A_cael_el']*BCFZY_elyte_ppr['elyte']/(params['e_el']*params['C_Osite_el'])
	#only tracking OH - should come up with a better way to index this
	#maybe switch to tracking O instead of OH since O is the first species in the list
	#dSVdt.set_val('X_ca',dXca_dt[1])
	#dSVdt.set_val('X_elyte',dXel_dt[1])
	
	#derivative of bulk gas phase species densities
	drho_dt = params['A_ca_gas']*BCFZY_gas_ppr['gas_ca']*params['gas_MW']/(1 - params['e_ca'])
	#dSVdt.set_val('rho_gas',drho_dt[0:SV.len('rho_gas')])
	
	#derivative of Phi_BCFZY
	#positive i_Far = negative charge to electrode
	i_Far = -ct.faraday*BCFZY_elyte_spr['OH(ca_b)']# + BCFZY_gas_spr['OH(ca_b)'])  #don't think this should be included - no charge transfer at BCFZY_gas interface
	i_dl = params['i_ext'] - i_Far*params['A_int']
	dSVdt.set_val('Phi_BCFZY', scale*i_dl/(params['C_dl']*params['A_int'])) #d Phi/dt = I_dl/C_dl
	
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
        self.derivative_func = derivative
        self.gas_MW = {'O2':2*15.999, 'H2O': 2*1.008 + 15.999, 'N2': 2*14.07,'AR': 39.948}
        self.runs = {}
        
    def load_phases(self,ctifile):
        "Import phases"
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
    
    def set_default_params(self):
        "set default parameters"
        self.params = {}
        
        Phi_ca_init = 0
        T = 773 #temperature (K)
        e_ca = 0.6 #cathode volume fraction (1-porosity)
        e_el = 1 #elyte volume fraction (fully dense)
        r_ca = 100e-9 #BCFZY particle radius (m)
        t_ca = 4000e-6 #cathode thickness (m)
        t_el = 4000e-6 #elyte thickness (m)
        f_contact = 0.9 #particle contact factor (dimensionless)
        C_dl = 1e-6
        i_ext = 0
        A_int = 1 #cathode-elyte interface area per unit area
        #assume fixed n_Ce3, n_Co, n_Fe initially
        n_Ce3 = 0.7 #fully reduced - max vacancies
        n_Co2 = 0.3 #avg Co ox state = +2.5
        n_Fe2 = 0.1 #avg Fe ox state = +3.5
        
        self.params['Phi_ca_init'] = Phi_ca_init
        self.params['T'] = T
        self.params['e_ca'] = e_ca
        self.params['e_el'] = e_el
        self.params['C_dl'] = C_dl
        self.params['i_ext'] = i_ext
        self.params['n_Ce3'] = n_Ce3 
        self.params['n_Co2'] = n_Co2 
        self.params['n_Fe2'] = n_Fe2 
        self.params['A_int'] = A_int
        self.params['A_ca_gas'] = BCFZY_gas_area_density(e_ca,r_ca,f_contact) #cathode-gas interface area per unit volume
        self.params['A_cael_ca'] = BCFZY_elyte_area_density(t_ca) #cathode-elyte interface area per unit volume in cathode (only at interface)
        self.params['A_cael_el'] = BCFZY_elyte_area_density(t_el) #cathode-elyte interface area per unit volume in elyte (only at interface)
        self.params['C_Osite_ca'] = O_site_density(self.phases['BCFZY_bulk'],MW=249.25) #O-site density in BCFZY (kmol/m3)
        self.params['C_Osite_el'] = O_site_density(self.phases['elyte'],MW=318.73) #O-site density in BCZYYb (kmol/m3)
        self.params['gamma_BCFZY_gas_inv'] = 1/self.interfaces['BCFZY_gas'].site_density
        #self.params['gamma_BCFZY_elyte_inv'] = 1/BCFZY_elyte.site_density #don't need this- no surface species
        #vector of gas molar masses - ensure order matches phase order
        self.params['gas_MW'] = [self.gas_MW[s.name] for s in gas_ca.species()] 
        #total gas density
        self.params['rho_gas_tot'] = gas_ca.density
        
    def initialize(self,ctifile,set_params=True):
        "Initialize default parameters and load phases"
        #load phases
        self.load_phases(ctifile)
        
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
            'X_OH_el', #only track protons for now - assume Ce oxidation state fixed
            #cathode species mole fractions
            'X_OH_ca' #only track protons for now - assume TM oxidation state fixed
           ]

        #set indices for IndexedVector
        self.SV_indices = {}
        self.SV_indices['rho_gas'] = (sv_names.index('rho_O2'), sv_names.index('rho_Ar') + 1)
        self.SV_indices['theta_ca'] = (sv_names.index('theta_O_ca'), sv_names.index('theta_Vac_ca') + 1)
        self.SV_indices['X_elyte'] = (sv_names.index('X_OH_el'), sv_names.index('X_OH_el') + 1)
        self.SV_indices['X_ca'] = (sv_names.index('X_OH_ca'), sv_names.index('X_OH_ca') + 1)
        self.SV_indices['Phi_cc'] = (sv_names.index('Phi_cc'), sv_names.index('Phi_cc') + 1)
        self.SV_indices['Phi_BCFZY'] = (sv_names.index('Phi_BCFZY'), sv_names.index('Phi_BCFZY') + 1)
        self.SV_indices['Phi_elyte'] = (sv_names.index('Phi_elyte'), sv_names.index('Phi_elyte') + 1)

    def init_SV(self):
        "get initial solution vector"
        SV_0 = np.zeros_like(self.sv_names,dtype=float)

        #IndexedVector for easier value setting
        SV_0 = IndexedVector(SV_0,self.SV_indices)
        SV_0.set_val('Phi_BCFZY',self.params['Phi_ca_init']) #initial cathode potential
        SV_0.set_val('X_elyte', 0.09) #match cti file
        SV_0.set_val('X_ca', 0.09) #match cti file
        SV_0.set_val('theta_ca',[0.76,0.09,0.15]) #match cti file X
        SV_0.set_val('rho_gas', self.phases['gas_ca'].density*self.phases['gas_ca'].Y[0:SV_0.len('rho_gas')]) #match cti file

        #pull vector back out of IndexedVector
        SV_0 = SV_0.vector
        
        return SV_0
    
    def run(self, t_span, SV_0=None, scale=1, method='BDF', save_as=None,**integrator_kw):
        if SV_0 is None:
            SV_0 = self.init_SV()
        sol = solve_ivp(lambda t, y: self.derivative_func(t, y, self.SV_indices, self.params,self.sim_objs, scale=scale),
                t_span, SV_0, method=method, **integrator_kw)
        
        if save_as is not None:
            run_data = {'sol':sol,'params':self.params,'sv_names':self.sv_names,'cti_file':self.ctifile}
            self.runs[save_as] = run_data
            
        return sol
    
    def plot_run(self,run_id,exclude=[]):
        "plot saved run data"
        run_data = self.runs[run_id]
        plot_sol(run_data['sol'],run_data['sv_names'],run_data['params'],exclude=exclude)
        
    def clear_runs(self):
        "clear saved run data"
        self.runs = {}