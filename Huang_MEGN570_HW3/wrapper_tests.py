import cantera as ct
import numpy as np
import os
import time

import sys
module_path = './modules'
sys.path.append(module_path)

from wrappers import IndexedVector, CtInterface


cti_path = './cti_files'
ctifile = os.path.join(cti_path,'pcfc_10252018.cti')

#Check performance of IndexedVector lookups
print("IndexedVector performance test\n===============================")
n_runs = int(1e5)

sv_names = [#electric potentials
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

SV_0 = np.zeros_like(sv_names,dtype=float)		   

#set SV indices
rho_gas_indices = (sv_names.index('rho_O2'), sv_names.index('rho_N2') + 1)
theta_ca_indices = (sv_names.index('theta_O_ca'), sv_names.index('theta_OH_ca') + 1)
X_elyte_indices = (sv_names.index('X_OH_el'), sv_names.index('X_OH_el') + 1)
X_ca_indices = (sv_names.index('X_OH_ca'), sv_names.index('X_OH_ca') + 1)

SV_indices = dict(zip(['rho_gas','theta_ca','X_elyte','X_ca'], [rho_gas_indices,theta_ca_indices, X_elyte_indices, X_ca_indices]))
SV_indices['Phi_cc'] = (sv_names.index('Phi_cc'), sv_names.index('Phi_cc') + 1)
SV_indices['Phi_BCFZY'] = (sv_names.index('Phi_BCFZY'), sv_names.index('Phi_BCFZY') + 1)
SV_indices['Phi_elyte'] = (sv_names.index('Phi_elyte'), sv_names.index('Phi_elyte') + 1)

print("Class instantiation\n--------------------------------")
start = time.time()
for i in range(n_runs):
    iv = IndexedVector(SV_0,SV_indices)
elapsed = round(time.time() - start,4)
cost = 1e6*elapsed/n_runs
print(f"Instantiation cost: {cost} s per million executions")

print("\nGetting values\n--------------------------------")
print("Single value:")
start = time.time()
for i in range(n_runs):
    SV_0[0]
elapsed_idx = round(time.time() - start,4)
print(f"Direct indexing: {elapsed_idx} s")

start = time.time()
for i in range(n_runs):
    iv.get_val('Phi_cc')
elapsed_class = round(time.time() - start,4)
print(f"Using custom class: {elapsed_class} s")

diff = 1e6*(elapsed_class - elapsed_idx)/n_runs
print(f"Difference: {diff} s per million executions")

print("\nArray of values:")
start = time.time()
for i in range(n_runs):
    SV_0[theta_ca_indices[0]:theta_ca_indices[1]]
elapsed_idx = round(time.time() - start,4)
print(f"Direct indexing: {elapsed_idx} s")

start = time.time()
for i in range(n_runs):
    iv.get_val('theta_ca')
elapsed_class = round(time.time() - start,4)
print(f"Using custom class: {elapsed_class} s")

diff = 1e6*(elapsed_class - elapsed_idx)/n_runs
print(f"Difference: {diff} s per million executions")

print("\nSetting values\n--------------------------------")
print("Single value:")
start = time.time()
for i in range(n_runs):
    SV_0[0] = 0
elapsed_idx = round(time.time() - start,4)
print(f"Direct indexing: {elapsed_idx} s")

start = time.time()
for i in range(n_runs):
    iv.set_val('Phi_cc',0) 
elapsed_class = round(time.time() - start,4)
print(f"Using custom class: {elapsed_class} s")

diff = 1e6*(elapsed_class - elapsed_idx)/n_runs
print(f"Difference: {diff} s per million executions")

print("\nArray of values:")
start = time.time()
for i in range(n_runs):
    SV_0[theta_ca_indices[0]:theta_ca_indices[1]] = [1,2]
elapsed_idx = round(time.time() - start,4)
print(f"Direct indexing: {elapsed_idx} s")

start = time.time()
for i in range(n_runs):
    iv.set_val('theta_ca',[1,2]) 
elapsed_class = round(time.time() - start,4)
print(f"Using custom class: {elapsed_class} s")

diff = 1e6*(elapsed_class - elapsed_idx)/n_runs
print(f"Difference: {diff} s per million executions")


#Check performance of CtInterface lookups
print("\nCtInterface performance test\n===============================")
n_runs = int(1e5)

gas_ca = ct.Solution(ctifile,'gas_ca')
BCFZY_bulk = ct.Solution(ctifile,'BCFZY_bulk')
transition_metal = ct.Solution(ctifile,'transition_metal')
BCFZY_gas_phases = [gas_ca, BCFZY_bulk, transition_metal]
BCFZY_gas = ct.Interface(ctifile,'BCFZY_gas',BCFZY_gas_phases)
test = CtInterface(BCFZY_gas)

#phase production rates
print("\nPhase production rate lookup\n--------------------------------")
start = time.time()
for i in range(n_runs):
    BCFZY_gas.net_production_rates[0:4]
elapsed_idx = round(time.time() - start,4)
print(f"Direct indexing: {elapsed_idx} s")

start = time.time()
for i in range(n_runs):
    test.phase_production_rates['gas_ca']
elapsed_class = round(time.time() - start,4)
print(f"Using custom class: {elapsed_class} s")

diff = 1e6*(elapsed_class - elapsed_idx)/n_runs
print(f"Difference: {diff} s per million executions")


#species production rates
print("\nSpecies production rate lookup\n--------------------------------")
start = time.time()
for i in range(n_runs):
    BCFZY_gas.net_production_rates[5]
elapsed_idx = round(time.time() - start,4)
print(f"Direct indexing: {elapsed_idx} s")

start = time.time()
for i in range(n_runs):
    test.species_production_rates['OH(ca_b)']
elapsed_class = round(time.time() - start,4)
print(f"Using custom class: {elapsed_class} s")

diff = 1e6*(elapsed_class - elapsed_idx)/n_runs
print(f"Difference: {diff} s per million executions")