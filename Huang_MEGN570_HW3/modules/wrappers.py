"""Wrapper classes for convenient lookups"""

import cantera as ct
import numpy as np

class IndexedVector():
	"""
	Convenience wrapper for numpy vectors
	Provides indexing functions
	"""
	
	def __init__(self,vector,indices):
		self.vector = vector
		self.indices = indices
		
	def get_val(self,key):
		"get value of key"
		index = self.indices[key]
		return self.vector[index[0]:index[1]]
	
	def set_val(self,key,value):
		"set value of key"
		index = self.indices[key]
		self.vector[index[0]:index[1]] = value
		
	def len(self,key):
		"get length of key vector"
		index = self.indices[key]
		return index[1] - index[0]

class CtInterface():
	"""
	Convenience wrapper for cantera interface objects
	Provides indexing functions and labeled production rates 
	"""
	
	def __init__(self,cantera_obj, quiet=True):
		self.obj = cantera_obj
		self.quiet = quiet
		#extract phases from cantera object
		self.phase_indices = {k:v for k,v in self.obj._phase_indices.items() if str(type(k))[8:25] == "cantera.composite"}
		#get ordered list of phases
		phase_sort = sorted(self.phase_indices.items(),key=lambda x: x[1])
		self.phases = [kv[0] for kv in phase_sort] #
		self.phase_names = [p.name for p in self.phases]
		self._set_species()
		self._set_species_indices()
		self._set_phase_species_indices()
		self._set_phase_species()
		
	def _set_species(self):
		"""
		Setter for species and species_names
		Called by __init__
		"""
		
		self._species = []
		for p in self.phases:
			self._species += p.species()
		self._species = np.array(self._species)
		self._species_names = [s.name for s in self._species]
		#make sure setter only runs at initialization
		if self.quiet==False:
			print('set species')	
			
	def _get_species(self):
		"Getter for species"
		return self._species
	
	def _get_species_names(self):
		"Getter for species_names"
		return self._species_names
	
	species = property(_get_species,_set_species, doc="Ordered list of species objects")
	species_names = property(_get_species_names,_set_species, doc="Ordered list of species names")
	
	def _set_species_indices(self):
		"""
		Setter for species_indices
		Called by __init__
		"""
		self._species_indices = dict(zip(self.species,np.arange(len(self.species))))
		#make sure setter only runs at initialization
		if self.quiet==False:
			print('set species indices')
		
	def _get_species_indices(self):
		"Getter for species_indices"
		return self._species_indices
	
	species_indices = property(_get_species_indices,_set_species_indices, doc="dict of indices for species (species_name:index)")
	
	def _set_phase_species_indices(self):
		"""
		Setter for phase_species_indices, phase_species_ranges, phase_species_start_indices, and phase_species_end_indices
		Called by __init__
		"""
		phase_counts = [p.n_species for p in self.phases]
		end_idx = np.cumsum(phase_counts)
		start_idx = np.concatenate(([0],np.cumsum(phase_counts)[:-1]))
		tuples = [(s,e) for s,e in zip(start_idx, end_idx)]
		ranges = [np.arange(s,e) for s,e in zip(start_idx, end_idx)]

		self._phase_species_start_indices = dict(zip(self.phase_names,start_idx))
		self._phase_species_end_indices = dict(zip(self.phase_names,end_idx))
		self._phase_species_indices = dict(zip(self.phase_names,tuples))
		self._phase_species_ranges = dict(zip(self.phase_names,ranges))
		#make sure setter only runs at initialization
		if self.quiet==False:
			print('set phase species indices')
		
	def _get_phase_species_start_indices(self):
		"Getter for phase_species_start_indices"
		return self._phase_species_start_indices
	
	def _get_phase_species_end_indices(self):
		"Getter for phase_species_end_indices"
		return self._phase_species_end_indices
	
	def _get_phase_species_indices(self):
		"Getter for phase_species_indices"
		return self._phase_species_indices
		
	def _get_phase_species_ranges(self):
		"Getter for phase_species_ranges"
		return self._phase_species_ranges
	
	phase_species_start_indices = property(_get_phase_species_start_indices,_set_phase_species_indices,doc="dict of start indexes for species belonging to each phase (phase_name:start_idx)")
	phase_species_end_indices = property(_get_phase_species_end_indices,_set_phase_species_indices,doc="dict of end indexes for species belonging to each phase (phase_name:end_idx)")
	phase_species_indices = property(_get_phase_species_indices,_set_phase_species_indices,doc="dict of index tuples for species belonging to each phase (phase_name:(start_idx, end_idx))")
	phase_species_ranges = property(_get_phase_species_ranges,_set_phase_species_indices,doc="dict of list of indices for species belonging to each phase (phase_name:[indices])")
	
	def _set_phase_species(self):
		"""
		Setter for phase_species
		Called by __init__
		"""
		self._phase_species = {phase:self.species_names[s:e] for phase, (s,e) in self.phase_species_indices.items()}
		if self.quiet==False:
			print('set phase species')
			
	def _get_phase_species(self):
		"Getter for phase_species"
		return self._phase_species
	
	phase_species = property(_get_phase_species, _set_phase_species,doc="dict of species belonging to each phase (phase_name:[species_names])")
	
	@property
	def phase_production_rates(self):
		"""
		dict of production rate vectors for each phase (phase_name:[prod_rates])
		recalculated at each call
		"""
		sdot = self.obj.net_production_rates
		sdot_dict = {phase:sdot[s:e] for phase, (s,e) in self.phase_species_indices.items()}
		return sdot_dict
	
	@property
	def species_production_rates(self):
		"""
		dict of production rate scalars for each species (species_name:prod_rate)
		recalculated at each call
		"""
		sdot = self.obj.net_production_rates
		sdot_dict = dict(zip(self.species_names,sdot))
		return sdot_dict
	