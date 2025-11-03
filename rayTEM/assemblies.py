# try:
#	 import cupy as xp
#	 flag_gpu = True
#	 from cupy.typing import ArrayLike
# except:
#	 import numpy as xp
#	 flag_gpu = False
#	 from numpy.typing import ArrayLike
import numpy as xp
flag_gpu = False
from numpy.typing import ArrayLike
import pickle

from pandas import DataFrame

from IPython.display import display # # TWP 2025/08/27 - adding import, required if not running inside IPython (e.g. outside of jupyter)

# # TWP 2025/08/27 - varying indices for matrices is hectic. 
# let's settle on a convention for things like: [whichZ,whichRay,[x,xθ,y,yθ,ϕ,E]]
# here I have modified the "z" convention to "ϕ", since this is primary interest 
# (what is the phase of the electron as a function of its distance travelled)
# as opposed to "z" which is our distance down the column 
# AND, I have swapped whichZ and whichRay. Why? feels weird to pass npts x 6 and get the
# new axis *inserted* as npts x nzs x 6. plus, whichZ out front makes looping easier

class MicroscopeSection:
	""" 
	TODO: Document

	To Do
	-----
	TODO: Remove pring_fancy.
		Revert back to __repr__ returning a str and add a print_fancy function.
	"""
	def __init__(self, name:str='',
				 elements:ArrayLike=None, 
				 position:float=0.,
				 ndim:int=2,
				 print_fancy:bool=True) -> object:
		self.name = name
		self.elements = elements
		self.position = position
		self.ndim = ndim
		self.print_fancy = print_fancy

		self.length = 0#xp.sum([e.length for e in self.elements])
		
		for ele in elements:
			ele.position = self.position + self.length
			self.length += ele.length
	
	def __repr__(self) -> str:
		if self.elements is None:
			return ''
		else:
			columns=['name', 'kind', 'length', 'position', 'strength', 'calibration']
			reps = [[e.name, e.kind, e.length, e.position, e.strength, e.calibration] for e in self.elements]
			
			if  self.print_fancy:
				display(DataFrame(reps, columns=columns))
				return ''
			else:
				return '\n'.join(['\t'.join([f"{key}: {value}, " for key,value in zip(columns,e)])for e in reps])
	"""
	def conform_ray_dim(self, r0:ArrayLike):
		""Recast the input arrays so they conform to 2*ndim+2.

		Parameters
		----------
		r0 : ArrayLike
			List of rays with possible initial conditions (x, θx, y, θy, E).
			For 1D the (y, θy) coordinates are excluded.

		Returns
		-------
		ndarray
			Recast array.

		Raises
		------
		ValueError
			If the array can not be recase due to an incorrect length of rays.

		To do
		-----
		#TODO: Have this as an external function or in a "Ray" class
		""
		#print(r0.shape,self.ndim,r0.shape[-1],self.ndim*2+1,r0.shape[-1]==self.ndim*2+1)
		if r0.shape[-1] == self.ndim*2+2:
			return r0
		elif r0.shape[-1] == self.ndim*2+1:
			# return xp.insert(r0, [1], xp.zeros(r0.shape[0]))
			return xp.insert(r0,-1,0,axis=1) # TWP 25/08/27 looks like we're trying to add a missing column, but what are the columns supposed to be? based on Lens, clearly 0,1,2,3 are x,angle,y,angle,but what are the last two? comment above says E is the 5th, so what's the 6th?
#		 elif r0.shape[-1] == self.ndim*2:
			return xp.pad(r0, ((0,0), (0,2)), constant_values=0)
		else:
			raise ValueError(f'The last shape of the rays has size {r0.shape[-1]}, which can not be understood as ndim*2+(z, E), ndim*2+(E), or ndim*2')
	"""

	def propagate_ray(self, r0:ArrayLike=None,
					   z:None|int|float|ArrayLike=None, 
					   ):
		"""
		To do
		-----
		#TODO: Allow for an array to be passed to z.
		"""
		#r0 = self.conform_ray_dim(r0)
		#if isinstance(z,arraylike):
		#	z_sub=z
		#ri = self.elements[0].propagate_ray(r0, z=z)
		"""
		#ri = xp.append(r0[:,None,:], ri, axis=1)
		for i, ele in enumerate(self.elements[1:]):
			i=i+1
			ele_ri = ele.propagate_ray(ri[:,-1], z=z)
			ele_ri[...,-2] += ele.position

			#for a infinitly thin element asign the last ray as the transofrmed array.
			# TWP 2025/08/27 - i think below is an off-by-one error. if MY length is zero (element[i], since we started enumerating at the 1nth element, then incremented i+=1 already). I think criteria for creating a new row is "if i'm the first element, or my thickness is nonzero)?
			if self.elements[i-1].length == 0: #TODO: Also check if the last z==z0
				ri = xp.append(ri[:,:-1], ele_ri, axis=1)
			else:
				ri = xp.append(ri, ele_ri, axis=1)
		"""
		if r0 is None:
			r0 = self.elements[0].rays()
		ri=[r0]
		for i,ele in enumerate(self.elements):
			ele_ri = ele.propagate_ray(ri[-1], z=z)
			#ele_ri[...,-2] += ele.position # TWP 2025/08/27 - do not add distance. drift already should update z
			#print(ele_ri.shape,r0.shape)
			if ele.length != 0:
				ri.append(ele_ri[:,:])
			else:
				ri[-1]=ele_ri[:,:]
		return xp.asarray(ri) # xp.swapaxes(xp.asarray(ri),0,1)

		#Include the initial ray. #TODO: Add conditional if source is included
		ri = xp.append(r0[:,None,:], ri, axis=1)
		return ri

	def propagate(self, input:ArrayLike=None, zs:None|float|int|ArrayLike=None,
				   output_structure:str='per layer') -> ArrayLike:
		"""propagate the input through the microscope section.

		Parameters
		----------
		input : ArrayLike
			Initial array to transform.
		zs : None | float | int | ArrayLike, optional
			Scaled propogation positions, by default None
			The positions (or created ones) are scaled from 0-1, with 0 being the start of the lens and 1 the total length.
			If None,	  a signle tranformation at the length of the element is performed.
			If float,	 a scaled position.
			If int,	   an array of size z from 0-1 is created.
			If ArrayLike, the input array is used as is.
		output_structure : str
			How to return the output, by default 'per layer'
			'per layer', list with propogation in each element.
			'collapsed', single array.
			'last',	  the last transformation during propocation.

		Returns
		-------
		ArrayLike
			Matricies during propogation.
		"""
		if input is None:
			input = xp.zeros((self.ndim*2,1))
			input[0] = 1
		output = [xp.asarray([input])]
		
		#lzs = self.get_scaled_z(zs, allow_array=True)
		
		for e in self.elements:
			output.append(e.propagate(output[-1][-1], zs=zs))

		if output_structure == 'per layer': return output
		elif output_structure == 'collapsed': return xp.vstack(output)
		elif output_structure == 'last': return output[-1]
		else: ValueError('An improper `output_structure` was requested.')

		return output

	def wobble(self,r0,elementIndex,func,kwargName,valRange,numSteps):
		vals=xp.linspace(valRange[0],valRange[1],numSteps)
		results=[]
		for v in vals:
			self.elements[elementIndex]=func(**{kwargName:v})
			rf=self.propagate_ray(r0)
			results.append(rf[-1,:,:]) # indices are: point in scope, which ray, which value (x,xt,y,yt...)
		return results

	@property
	def labels(self):
		return { e.name:e.position for e in self.elements if e.name is not None }

	def save(self,filename):
		with open(filename+".pkl",'wb') as f:
			pickle.dump(self,f)
def load(filename):
	with open(filename+".pkl",'rb') as f:
		obj = pickle.load(f)
	return obj 

		