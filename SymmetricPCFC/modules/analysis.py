import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from calc import BCFZY_frac, elyte_frac


def plot_sol(sol,sv_names, params, exclude=[],axes=None):
	"""Plot time evolution of key quantities from solution vector"""
	y = dict(zip(sv_names,sol['y']))

	if axes is None:
		figs = []
		axes = np.array([])
	else:
		#keep track of which axis is current
		axis_counter = 0
	
	#plot cathode potential
	if 'Phi_BCFZY' not in exclude:
		if axes is None:
			fig1, ax1 = plt.subplots()
			ax1.set_ylabel('Cathode Potential (V)')
			#ax1.set_xlabel('Time (s)')
			ax1.set_title('Cathode Potential',size=14)
			#ax1.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useOffset=False))
			figs.append(fig1)
			axes = np.concatenate((axes,[ax1]))
		else:
			ax1 = axes[axis_counter]
			axis_counter += 1

		ax1.semilogx(sol['t'],y['Phi_BCFZY'],label='$\Phi_{BCFZY}$')
					
	#plot bulk mole fractions
	#elyte bulk
	if 'X_elyte' not in exclude:
		if axes is None:
			fig2, axes2 = plt.subplots(1,3,figsize=(12,3))
			for ax in axes2:
				ax.set_ylabel('$X_k$')
			fig2.suptitle('Elyte bulk mole fractions',size=14,y=1.05)
			figs.append(fig2)
			axes = np.concatenate((axes,axes2))
		else:
			axes2 = axes[axis_counter:axis_counter+3]
			axis_counter += 3

		X_O_el, X_Vac_el = elyte_frac(y['X_OH_el'], params['n_Ce3'])

		axes2[0].semilogx(sol['t'],X_O_el,'k', label='O(el)')
		axes2[1].semilogx(sol['t'],y['X_OH_el'], ':k', label='OH(el)')
		axes2[2].semilogx(sol['t'],X_Vac_el, '--k', label='Vac(el)')
		
	#cathode bulk
	if 'X_ca' not in exclude:
		if axes is None:
			fig2b, axes2b = plt.subplots(1,3,figsize=(12,3))
			for ax in axes2b:
				ax.set_ylabel('$X_k$')
			fig2b.suptitle('Cathode bulk mole fractions',size=14,y=1.05)
			figs.append(fig2b)
			axes = np.concatenate((axes,axes2b))
		else:
			axes2b = axes[axis_counter:axis_counter+3]
			axis_counter += 3
			
		X_O_ca, X_Vac_ca = BCFZY_frac(y['X_OH_ca'], params['n_Co2'], params['n_Fe2'])

		axes2b[0].semilogx(sol['t'],X_O_ca,'k', label='O(ca)')
		axes2b[1].semilogx(sol['t'],y['X_OH_ca'], ':k', label='OH(ca)')
		axes2b[2].semilogx(sol['t'],X_Vac_ca, '--k', label='Vac(ca)')
		
	#plot surface coverage fractions
	if 'theta_ca' not in exclude:
		if axes is None:
			fig3, axes3 = plt.subplots(1,3,figsize=(12,3))
			for ax in axes3:
				ax.set_ylabel('$\Theta_k$')
			fig3.suptitle('Cathode surface coverage fractions',size=14,y=1.05)
			figs.append(fig3)
			axes = np.concatenate((axes,axes3))
		else:
			axes3 = axes[axis_counter:axis_counter+3]
			axis_counter += 3
			
		theta_Vac_ca = 1 - (y['theta_O_ca'] + y['theta_OH_ca'])

		axes3[0].semilogx(sol['t'],y['theta_O_ca'], 'k', label='O(ca_s)')
		axes3[1].semilogx(sol['t'],y['theta_OH_ca'],':k', label='OH(ca_s)')
		axes3[2].semilogx(sol['t'],theta_Vac_ca, 'k--',label='Vac(ca_s)')
		
	#plot gas species mass fractions
	if 'Y_gas' not in exclude:
		if axes is None:
			fig4, axes4 = plt.subplots(2,2,figsize=(8,6))
			for ax in axes4.ravel():
				ax.set_ylabel('$Y_{k, gas}$')
			fig4.suptitle('Gas mass fractions',size=14,y=1.05)
			figs.append(fig4)
			axes = np.concatenate((axes,axes4.ravel())
		else:
			axes4 = axes[axis_counter:axis_counter+4].reshape((2,2))
			axis_counter += 4 
			
		axes4[0,0].semilogx(sol['t'],y['rho_O2']/params['rho_gas_tot'],'r',label='O2')
		axes4[0,1].semilogx(sol['t'],y['rho_H2O']/params['rho_gas_tot'],'b',label='H2O')
		axes4[1,0].semilogx(sol['t'],y['rho_N2']/params['rho_gas_tot'],'green',label='N2')
		# Y_Ar = 1 - np.sum((y['rho_O2'],y['rho_N2'],y['rho_H2O']),axis=0)/params['rho_gas_tot']
		# axes4[1].semilogx(sol['t'],Y_Ar,label='Ar')
		axes4[1,1].semilogx(sol['t'],y['rho_Ar']/params['rho_gas_tot'],'k',label='Ar')
		

	for ax in axes:
		ax.set_xlabel('Time (s)')
		ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useOffset=False))
		ax.legend()
		
	if axes is not None:
		for fig in figs:
			fig.tight_layout()
		
	return axes
	