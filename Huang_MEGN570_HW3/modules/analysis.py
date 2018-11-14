"""Functions for analyzing simulation results (mostly plotting)"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from calc import elyte_net_charge, ca_net_charge


def plot_call(x,y,func, ax, **kwargs):
	"""
	Call to matplotlib plotting function
	Convenience function for use in plot_sol
	
	Parameters:
	-----------
	x: x values
	y: y values
	func: matplotlib plotting function to call. Options: 'plot','semilogx','semilogy','loglog','scatter'
	ax: axis on which to plot
	kwargs: kwargs to pass to func
	"""
	
	if func=='plot':
		ax.plot(x,y,**kwargs)
	elif func=='semilogx':
		ax.semilogx(x,y,**kwargs)
	elif func=='semilogy':
		ax.semilogy(x,y,**kwargs)
	elif func=='loglog':
		ax.loglog(x,y,**kwargs)
	elif func=='scatter':
		ax.scatter(x,y,**kwargs)
	else:
		raise KeyError(f'No plotting function matching name {func}')

def plot_sol(sol,sv_names, params, exclude=[],axes=None, plot_func='semilogx',label=''):
	"""
	Plot time evolution of key quantities from solution vector
	
	Parameters:
	-----------
	sol: solve_ivp `solution` result 
	sv_names: vector or list of solution vector names
	params: dict of parameters
	exclude: list of quantities to exclude from plot. Options: 'Phi_BCFZY','X_elyte','X_ca','theta_ca','Y_gas','net_charge'
	axes: axes on which to plot. If None, new axes will be created
	plot_func: matplotlib plotting function to call. Options: 'plot','semilogx','semilogy','loglog'
	label: label for legends
	
	Returns axes
	"""
	y = dict(zip(sv_names,sol['y']))

	figs = []
	if axes is None:
		new_axes = np.array([])
	else:
		#keep track of which axis is current
		axis_counter = 0
	
	#plot cathode potential
	if 'Phi_BCFZY' not in exclude:
		if axes is None:
			fig1, ax1 = plt.subplots(figsize=(8,4))
			ax1.set_ylabel('Cathode Potential (V)')
			#ax1.set_xlabel('Time (s)')
			ax1.set_title('Cathode Potential',size=14)
			#ax1.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useOffset=False))
			figs.append(fig1)
			new_axes = np.concatenate((new_axes,[ax1]))
		else:
			ax1 = axes[axis_counter]
			axis_counter += 1
			figs.append(plt.gcf())

		plot_call(sol['t'],y['Phi_BCFZY'],plot_func,ax1,label=label)
		
		ax1.legend(loc='upper left', bbox_to_anchor=(1,1))
	
	O_species = ['O','OH','Vac']
	lstyles = ['-','--',':','-.']
	#plot bulk mole fractions
	#elyte bulk
	if 'X_elyte' not in exclude:
		if axes is None:
			fig2, axes2 = plt.subplots(1,3,figsize=(12,3))
			for ax, spec in zip(axes2,O_species):
				ax.set_ylabel('$X_k$')
				ax.set_title(spec)
			fig2.suptitle('Electrolyte bulk mole fractions',size=14,y=1.05)
			figs.append(fig2)
			new_axes = np.concatenate((new_axes,axes2))
		else:
			axes2 = axes[axis_counter:axis_counter+3]
			axis_counter += 3
			figs.append(plt.gcf())

		#X_O_el, X_Vac_el = elyte_frac(y['X_OH_el'], params['n_Ce3'])
		for ax, spec, ls in zip(axes2,O_species, lstyles):
			key = 'X_' + spec + '_el'
			plot_call(sol['t'],y[key],plot_func, ax, ls=ls, label=label)
		
		axes2[2].legend(loc='upper left',bbox_to_anchor=(1,1))
		
	#cathode bulk
	if 'X_ca' not in exclude:
		if axes is None:
			fig2b, axes2b = plt.subplots(1,3,figsize=(12,3))
			for ax,spec in zip(axes2b,O_species):
				ax.set_ylabel('$X_k$')
				ax.set_title(spec)
			fig2b.suptitle('Cathode bulk mole fractions',size=14,y=1.05)
			figs.append(fig2b)
			new_axes = np.concatenate((new_axes,axes2b))
		else:
			axes2b = axes[axis_counter:axis_counter+3]
			axis_counter += 3
			figs.append(plt.gcf())
			
		#X_O_ca, X_Vac_ca = BCFZY_frac(y['X_OH_ca'], params['n_Co2'], params['n_Fe2'])
		
		for ax, spec, ls in zip(axes2b,O_species, lstyles):
			key = 'X_' + spec + '_ca'
			plot_call(sol['t'],y[key],plot_func, ax, ls=ls, label=label)
		
		axes2b[2].legend(loc='upper left',bbox_to_anchor=(1,1))
		
	#plot surface coverage fractions
	if 'theta_ca' not in exclude:
		if axes is None:
			fig3, axes3 = plt.subplots(1,3,figsize=(12,3))
			for ax,spec in zip(axes3,O_species):
				ax.set_ylabel('$\Theta_k$')
				ax.set_title(spec)
			fig3.suptitle('Cathode surface coverage fractions',size=14,y=1.05)
			figs.append(fig3)
			new_axes = np.concatenate((new_axes,axes3))
		else:
			axes3 = axes[axis_counter:axis_counter+3]
			axis_counter += 3
			figs.append(plt.gcf())
			
		#theta_Vac_ca = 1 - (y['theta_O_ca'] + y['theta_OH_ca'])

		for ax, spec, ls in zip(axes3,O_species,lstyles):
			key = 'theta_' + spec + '_ca'
			plot_call(sol['t'],y[key],plot_func, ax, ls=ls, label=label)
		
		axes3[2].legend(loc='upper left',bbox_to_anchor=(1,1))
		
	#plot gas species mass fractions
	gas_species = ['O2','H2O','N2','Ar']
	if 'Y_gas' not in exclude:
		if axes is None:
			fig4, axes4 = plt.subplots(2,2,figsize=(10,6))
			for ax,gas in zip(axes4.ravel(),gas_species):
				ax.set_ylabel('$Y_k$')
				ax.set_title(gas)
			fig4.suptitle('Gas mass fractions',size=14,y=1.05)
			figs.append(fig4)
			new_axes = np.concatenate((new_axes,axes4.ravel()))
		else:
			axes4 = axes[axis_counter:axis_counter+4].reshape((2,2))
			axis_counter += 4
			figs.append(plt.gcf())
		
		rho_gas_tot = y['rho_O2'] + y['rho_H2O'] + y['rho_N2'] + y['rho_Ar']
		for ax,gas,ls in zip(axes4.ravel(),gas_species,lstyles):
			key = 'rho_' + gas
			plot_call(sol['t'],y[key]/rho_gas_tot, plot_func, ax, ls=ls,label=label)
		
		axes4[0,1].legend(loc='upper left', bbox_to_anchor=(1,1))
	
	#plot cathode and elyte bulk net charges
	bulks = ['Cathode',' Electrolyte']
	if 'net_charge' not in exclude:
		if axes is None:
			fig5, axes5 = plt.subplots(1,2,figsize=(8,3))
			for ax, bulk in zip(axes5,bulks):
				ax.set_ylabel('Net charge')
				ax.set_title(bulk)
			fig5.suptitle('Bulk Net Charge',size=14,y=1.05)
			figs.append(fig5)
			new_axes = np.concatenate((new_axes,axes5))
		else:
			axes5 = axes[axis_counter:axis_counter+2]
			axis_counter += 2
			figs.append(plt.gcf())
	
		bulk_chg_ca = ca_net_charge(y['X_O_ca'],y['X_OH_ca'],params['n_Co2'],params['n_Fe2'])
		bulk_chg_el = elyte_net_charge(y['X_O_el'],y['X_OH_el'],params['n_Ce3'])
		bulk_chg = dict(zip(bulks,[bulk_chg_ca,bulk_chg_el]))
		
		for ax, bulk,ls in zip(axes5,bulks,lstyles):
			plot_call(sol['t'],bulk_chg[bulk], plot_func, ax, ls=ls, label=label)
		
		axes5[1].legend(loc='upper left', bbox_to_anchor=(1,1))
		
	#formatting for all axes
	if axes is None:
		axes = new_axes
	
	xfmt = mpl.ticker.ScalarFormatter()
	xfmt.set_powerlimits((-2,2))
	
	for ax in axes:
		ax.set_xlabel('Time (s)')
		ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useOffset=False))
		if plot_func in ('plot','semilogy'):
			ax.xaxis.set_major_formatter(xfmt)
		
	for fig in figs:
		fig.tight_layout()
		
	return axes
	