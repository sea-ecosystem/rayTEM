from __future__ import annotations

from typing import Sequence, Literal
from numpy.typing import ArrayLike

import numpy as xp
flag_gpu = False
import traceback,inspect
from warnings import warn
from abc import abstractmethod

from .seashells import SEASerializable

from copy import deepcopy

# CONVENTION: Rays are defined by positions laterally (x,y), angles (xt,yt, "t" for theta θ or tilt), position down column (z), intensities (I, e.g. when an aperture masks the beam and the overall intensity is reduced), and energy E
# rays at a given position are 2D: a list up septuplets (grab the 'x' column to grab each ray's x position for example).
# rays throughout the microscope are 3D: a list of the above.
# currently, the columns are ordered: [x,xθ,y,yθ,I,ϕ,E]
# but with the columnByName function used universally, additional columns can be added without every Element needing to be updated, and columns can be reordered arbitrarily.

convention = ["x","xt","y","yt","z","E","I","R"]
# given a keyword, return the column associated. r0[:,columnByName('x')] should return every ray's x position
def columnByName(name):
	return convention.index(name)
# given a transfer_matrix defined by a subset of columns (e.g. a 2x2 lens for focusing in x only) "inflate" out to 7x7 based on convention set by columnByName. [ x₂ θ₂ ] = [2x2] @ [ x₁ θ₁ ] (https://en.wikipedia.org/wiki/Ray_transfer_matrix_analysis) would become [ x₂ xθ₂ y₂ yθ₂ ....] = [7x7] @ [ x₁ xθ₁ y₁ yθ₁....]
def fix_mat_dims(m,columnNames):
	new=xp.eye(len(convention))
	for i,n1 in enumerate(columnNames):
		for j,n2 in enumerate(columnNames):
			new[columnByName(n1),columnByName(n2)]=m[i,j]
	return new
# similar to fix_mat_dims, but for rays
def fix_ray_dims(rays,columnNames):
	new=xp.zeros((len(rays),len(convention)))
	for i,name in enumerate(columnNames):
		new[:,columnByName(name)]=rays[:,i]
	return new

class Element(SEASerializable):
	def __init__(self, name:str='', kind:str=None ) -> SEASerializable:
		"""General microscope element class. Only the basic/required attributes (name and kind) are populated, as additional attributed can be defined at the inheriting class level. e.g. a Lens has a "strength", but a Drift section does not.
		Inheriting classes are required to define a transfer_matrix (enforced via abstractmethod), and *may* define a custom propagate_ray function if the standard "[ x₂ xθ₂ y₂ yθ₂ ....] = [7x7] @ [ x₁ xθ₁ y₁ yθ₁....]" is not applicable

		Parameters
		----------
		name : str, optional
			Name given to the lens, by default ''
		kind : str, optional
			Type of element, by default None
		"""
		self.name = name
		self.kind = kind

	#####################################
    # region: Dunders

	def __repr__(self):
		return self.tabulate()

	def tabulate(self,header=True,columns=[ "name", "kind", "position", "length", "strength", "calibration", "axis" ]) -> str:
		rep = { k:getattr(self,k) for k in self.__dict__ if k in columns }
		#rep = {'name':self.name,
		#	   'kind':self.kind,
		#	   }
		h = [] ; s = []
		for k,v in rep.items():
			h.append(k+" "*(8-len(k)))
			if v is None:
				v="[None]"
			if isinstance(v,float):
				v = xp.round(v,7)
			v=str(v) ; v=v+" "*(8-len(v)) ; v=v[:8]
			s.append(v)
		if header:
			return " ".join(h)+"\n"+" ".join(s)
		return " ".join(s)
	
	def __str__(self):
		if self.name is None or self.name=='': name = 'Unnamed'
		else: name = self.name
		if self.kind is None: kind = 'Unkown'
		else: kind = self.kind
		return f'{name} ({kind} Element)'

	def copy(self):
		return deepcopy(self)
		dic = self.__dict__
		allowed_kwargs = inspect.signature(type(self)).parameters.keys() # infer allowed kwargs from function itself, and filter down to only those.
		dic = { k:v for k,v in dic.items() if k in allowed_kwargs } # e.g., Source doesn't accept "length" even though it
		return type(self)(**dic)

	# e.position should be read-only! user should not set position of an element within a section, they should use s.move(...)
	@property
	def position(self):
		return self._position
	#@position.setter			# commented out: "position" attribute should be read-only! this setter only exists to ensure pytest tests as expected (i.e., failing a test when "position" is writeable)
	#def position(self,val):
	#	self._position = val

	# endregion
	#####################################

	@abstractmethod # abstractmethod means a class which inherits Element will be required to define this function
	def transfer_matrix(self) -> xp.ndarray:
		r"""Transfer matrix for ray propogation: https://en.wikipedia.org/wiki/Ray_transfer_matrix_analysis
		This will typically be defined in terms of ray position x,y, and ray angles xt,yt.
		inheriting class only needs to define with the relevant parameters, then inflate using fix_mat_dims
		"""
		pass

	def propagate_ray(self, r0:xp.ndarray,
					  z:float=None, z0:float=0) -> xp.ndarray:
		"""propagate an array through an element.

		Parameters
		----------
		r0 : xp.ndarray
			List of rays with possible initial conditions (x, θx, y, θy, E).
		z : None | int | float | xp.ndarray, optional
			Positions in the element to propagate to by default None
		z0 : None | float, optional
			Initial position of the element, by default 0

		Returns
		-------
		xp.ndarray
			List of propagated rays with initial condition (x, θx, y, θy, z, E)
		"""
		m = self.transfer_matrix()
		rf = xp.einsum('mn,in->im', m, r0) # matrix multiplication for a "list of vectors"
		# additive terms: z_new = z_old+length, rotation_new = rotation_old+R
		rf[:,columnByName("z")] += self.length
		rf[:,columnByName("R")] += getattr(self,"rotation",0)
		rf[:,columnByName("x")] += getattr(self,"shift_x",0)
		rf[:,columnByName("y")] += getattr(self,"shift_y",0)
		rf[:,columnByName("xt")] += getattr(self,"tilt_x",0)
		rf[:,columnByName("yt")] += getattr(self,"tilt_y",0)

		#print("propagate_ray",self.name,"new rotation",rf[-1,columnByName("R")])
		return rf

class Source(Element):
	"""Source element class. Source element can be put in a MicroscopeSection and then propagating "through" the section will mean the starting rays r0 are generated by the Source (instead of requiring the user pass in starting rays)

		Parameters
		----------
		name : str, optional
			Name of the Source, by default ''
		size : tuple, optional
			size in x and y: rays will be emitted from a square grid of points
		np_xy : tuple, optional
			number of grid points in x and y
		angle : tuple, optional
			maximum angle in x and y: rays will be emitted at multiple angles from each grid point
		na_xy : tuple, optional
			number of angles for emitted rays in x and y
		position : float, optional
			The position of the element along the z-axis, by default 0
		"""

	def __init__(self, name:str=None,
			size:tuple=(2e-3,2e-3), # size in x and y (square grid)
			np_xy:tuple=(3,3),		# number of grid points in x and y. (0,0) --> point-source. (1,1) --> single ray at x,y=size
			angle:tuple=(1,1),		# angles in x,y (ranges of xt yt)
			na_xy:tuple=(3,3),		# number of angles. (0,0) --> parallel rays. (1,1) --> ray at xt,yt=angle only
			position:float=None) -> SEASerializable:
		super().__init__(name=name, kind='Source')

		self.size = size
		self.np_xy = np_xy
		self.angle = angle
		self.na_xy = na_xy
		self._position = position
		self.length = 0
		self.strength = 0
		self.calibration = None

	# Source term, initialize rays at sweep of angles and positions
	def rays(self):
		#print("SOURCE GENERATING RAYS",self.np_xy,self.size,self.na_xy,self.angle)
		xs=xp.zeros(1) ; ys=xp.zeros(1) # central ray only by default if np_xy is zero
		if self.np_xy[0]:
			xs=xp.linspace(-self.size[0],self.size[0],self.np_xy[0])
		if self.np_xy[1]:
			ys=xp.linspace(-self.size[1],self.size[1],self.np_xy[1])
		xts=xp.zeros(1) ; yts=xp.zeros(1) # zero-angle ray only by default if na_xy is zero
		if abs(self.angle[0])>0 and self.na_xy[0]:
			xts=xp.linspace(-self.angle[0],self.angle[0],self.na_xy[0])
		if abs(self.angle[1])>0 and self.na_xy[1]:
			yts=xp.linspace(-self.angle[1],self.angle[1],self.na_xy[1])
		shape=(len(xs),len(ys),len(xts),len(yts))
		array=xp.zeros((len(xs)*len(ys)*len(xts)*len(yts),4))
		array[:,0]=(xs[:,None,None,None]*xp.ones(shape)).flat
		array[:,1]=(ys[None,:,None,None]*xp.ones(shape)).flat
		array[:,2]=(xts[None,None,:,None]*xp.ones(shape)).flat
		array[:,3]=(yts[None,None,None,:]*xp.ones(shape)).flat
		array=fix_ray_dims(array,["x","y","xt","yt"])
		array[:,columnByName("I")]=xp.ones(shape).flat
		return array

	# dummy propagation in case someone tries to propagate through since this is technically an element
	def propagate_ray(self, r0:xp.ndarray, **kwargs) -> xp.ndarray:
		return r0

class Aperture(Element):
	"""Aperture element class. An aperture serves to crop the beam, and the total beam intensity is reduced dependent on the area of the beam and the area of aperture.

		Parameters
		----------
		name : str, optional
			Name of the Aperture, by default ''
		radius : float, optional
			radius of the round aperture
		calibration : float, optional
			linear scaling factor applied to the radius. e.g. nominally 2mm aperture but fitting tells us it is actually 1.95
		position : float, optional
			The position of the element along the z-axis, by default 0
		"""

	def __init__(self, name:str='', radius:float=0., calibration:float=None, position:float=None) -> SEASerializable:
		super().__init__(name=name, kind='Aperture')
		self._position = position
		self.radius = radius
		self.calibration = calibration

	#def transfer_matrix(self) -> xp.ndarray:
	#	r"""Transfer matrix for ray propogation.
	#	"""
	#	#print("WARNING: APERTURE",self.name,"TRANSFER MATRIX CALLED")
	#	#m = xp.eye(6)#[...,None]*xp.ones_like(s)
	#	m = xp.eye(4) # drift tube updates x from xθ and y from yθ
	#	return fix_mat_dims(m,["x","xt","y","yt"])

	# TWO WAYS TO IMPLEMENT AN APERTURE:
	# 1) set the intensity of any rays "outside" the aperture to zero. this is fine for plotting and we can capture beam current by looking at how many rays are zeroed out. *BUT*, this will be problematic during fitting, as rays which "pop" into and out of view will yield an intensity vs [whatever] function with step edges.
	# def propagate_ray(self, r0:xp.ndarray,
	#				  z:float=None, z0:float=0) -> xp.ndarray:
	#	rf=xp.zeros(r0.shape)+r0
	#	radii=xp.sqrt( r0[:,columnByName("x")]**2 + r0[:,columnByName("y")]**2 )
	#	rf[radii>self.radius,columnByName("I")]=0
	#	return rf
	# 2) aperture can rescale all rays based on the outer ray's position, or the area covered by the rays. we can thus calculate reductions in beam current based on the aperture's reduction in intensity (area cropped out). we're effectively pretending the originating rays were less divergent or something, which is actually sort of what we see IRL; you can't tell the divergence of the beam from the gun because the VOA masks out a bunch of it. This will only work for one aperture in the system though (otherwise second aperture undoes the scaling of the first one? or should we only allow the aperture to scale-down, so if the first aperture scales down, second scales down further (second is smaller), or first scale down, second leaves it alone (second is larger, we'd be able to see our first aperture in the CCD for example). and how do we handle apertures of different shapes??
	def propagate_ray(self, r0:xp.ndarray,
					  z:float=None, z0:float=0) -> xp.ndarray:
		xmax = xp.amax(r0[:,columnByName("x")])
		ymax = xp.amax(r0[:,columnByName("y")])
		scale_x = 1 if xmax<self.radius else self.radius/xmax
		scale_y = 1 if ymax<self.radius else self.radius/ymax
		#print("Aperture",self.name,"radius",self.radius,"scale x,y",scale_x,scale_y,"(",xmax,ymax,")")
		rf=xp.zeros(r0.shape)+r0
		rf[:,columnByName("x")]*=scale_x
		rf[:,columnByName("xt")]*=scale_x
		rf[:,columnByName("y")]*=scale_y
		rf[:,columnByName("yt")]*=scale_y
		rf[:,columnByName("I")]*=scale_x*scale_y
		return rf

class Drift(Element):
	"""Drift element class for free-space propagation.

		Parameters
		----------
		name : str, optional
			Name of Drift segment, useful for creating named planes which do not affect beam propagation (e.g., a "sample plane")
		length : float, optional
			length for free-space propagation
		calibration : float, optional
			linear scaling applied to length. e.g. nominally we believe the next lens is at 10 mm, but maybe fitting says 9.9
		position : float, optional
			The position of the element along the z-axis, by default None

		References
		----------
		https://en.wikipedia.org/wiki/Ray_transfer_matrix_analysis#Free_space_example
		"""

	def __init__(self, name:str='', length:float=0., calibration:float=None, position:float=None) -> SEASerializable:

		super().__init__(name=name,kind='Drift')
		self._position = position
		self.length = length
		self.calibration = calibration

	def transfer_matrix(self) -> xp.ndarray:

		m = xp.eye(4) # drift tube updates x from xθ and y from yθ

		s = self.length

		if self.calibration is not None:
			s *= self.calibration

		if s != 0:
			m[0,1] = s
			m[2,3] = s
		elif self.length == 0:
			pass

		return fix_mat_dims(m,["x","xt","y","yt"])

class Quadrapole(Element):
	def __init__(self, name:str='', 
				 position:float=None, length:float=0.,
				 strength:float=0, calibration:float=None) -> SEASerializable:

		"""Quadripole.

		Parameters
		----------
		name : str, optional
			Name given to the lens, by default ''
		position : float, optional
			The position of the element along the z-axis, by default 0
		length : int, optional
			Length of the element, by default 0
		strength : float, optional
			Defined as the field strength (related to inverse focal length,
			see equations in brown1983), by default 0
		calibration : float, optional
			Currnet calibration of the lens in units of ???/A, by default None
		label : bool, optional
			If the element should be labeled when plotted, by default False
		print_fancy : bool, optional
			If a fancy table should be used when printed, by default True
		"""
		
		if length == 0: kind = 'Thin quad'
		else:		   kind = 'Quad'

		super().__init__(name=name,kind=kind)
		self._position = position
		self.length = length
		self.strength = strength
		self.calibration = calibration

	def transfer_matrix(self) -> xp.ndarray:
		r"""Transfer matrix for ray propogation.
		
		The homogenous equaiton of motion approximation leads to a linear solution of $u"+k(s)u=0$ given as $u(s)=C(s)u_0+S(s)u_0', where s is the distance traveled (~z for small u').
		For K>0 $C=cos(\sqrt{Ks})$ and $S=\frac{1}{\sqrt{K}} sin(\sqrt{Ks})$ and for K<0 $C=cosh(\sqrt{|K|s})$ and $S=\frac{1}{\sqrt{|K|}} sinh(\sqrt{|K|s})$.

		To Do
		-----
		"""
		
		#m = xp.eye(6)#[...,None]*xp.ones_like(s) # TWP 2025/08/27 - adding ones_like expression so m is 6x6x1, otherwise eigsum in propagate will fail
		#m = xp.eye(4) # quadrupole updates xθ from x and yθ from y

		K=self.strength
		if self.calibration is not None:
			# linear scaling from mA (lens current) to lens strength?
			if isinstance(self.calibration,(int,float)):
				c = self.calibration
				K *= c
			else:
				c,p = self.calibration
				K = K**p * c

		if K==0:
			return fix_mat_dims(xp.eye(4),["x","xt","y","yt"])

		if self.length==0:
			print("using special")
			X=xp.asarray([[ 1 , 0 ],
					     [ -(K**2) , 1 ]])
			Y=xp.asarray([[ 1 , 0 ],
						 [  (K**2) , 1 ]])
			if K>0: # testing with REMOVED_PRIVATE_INSTRUMENT_TREE/PRIVATE_INSTRUMENT_v2/DQCM.py, sign flip is needed for len=0.08,cal=1 vs len=0,cal=sqrt(0.08)
				X,Y=Y,X
			#print("X",fix_mat_dims(X,["x","xt"]))
			#print("Y",fix_mat_dims(X,["y","yt"]))
		else:
			kL=abs(K*self.length) ; L=self.length
			C=xp.cos(kL) ; S=xp.sin(kL)
			#print("K,L,C,S",K,L,C,S)
			X=xp.asarray([[  C  , 1/K*S ],  # Brown1983 page 46, note the similarity to
						[-K*S ,   C   ]]) # https://en.wikipedia.org/wiki/Ray_transfer_matrix_analysis#Example:_Thin_lens if L=0
			Y=xp.asarray([[  C  , 1/K*S ],  # calculate x,xt and y,yt 2x2s separately, then matmul
						[ K*S ,   C   ]])
			# Small angle approximation: C=1, S=K*L
			#X=xp.asarray([[  1 , L ],
			#			[-K*K*L, 1   ]])
			#Y=xp.asarray([[  1 , L ],
			#			[ K*K*L,  1  ]])

		m=xp.matmul( fix_mat_dims(X,["x","xt"]) , fix_mat_dims(Y,["y","yt"]) )
		#print("QUAD",m,self.strength,K,self.calibration,self.length)
		return m
		

class Dipole(Element):
	def __init__(self, name:str='',
				 position:float=None, length:float=0.,
				 strength:float=0, calibration:float=None,
				 axis: Literal['x','y'] | float | Sequence='x'
				 ) -> SEASerializable:
		"""Dipole.

		Parameters
		----------
		name : str, optional
			Name given to the dipole, by default ''.
		position : float, optional
			The position of the element along the z-axis, by default None.
		length : float, optional
			Length of the element, by default 0.
		strength : float, optional
			Angular kick applied by the dipole, by default 0.
		calibration : float or tuple, optional
			Calibration applied to ``strength``. Numeric values apply a
			linear scale; tuple values are interpreted as ``(scale, power)``,
			matching ``Quadrapole`` behavior.
		axis : {'x', 'y', float, Sequence}, optional
			Transverse axis receiving the kick. Can be 'x', 'y', a float angle in radians, or a sequence [x, y], by default 'x'.

		Raises
		------
		UserWarning
			If ``axis`` is not ``'x'``, ``'y'``, a float, or a sequence.
		"""
		if length == 0: kind = 'Thin dipole'
		else:		   kind = 'Dipole'

		super().__init__(name=name,kind=kind)
		self._position = position
		self.length = length
		self.strength = strength
		self.calibration = calibration
		
		if axis.lower() == 'x':
			self.phi = 0
		elif axis.lower() == 'y':
			self.phi = xp.pi/2
		elif isinstance(axis, float):
			if axis > 0 and axis <= 2*np.pi:
				self.phi = axis
			else:
				self.phi = xp.remainder(axis + xp.pi, 2 * xp.pi) - xp.pi
		elif isinstance(axis, Sequence):
			self.phi = xp.arctan2(axis[1],axis[0])
		else:
			raise UserWarning(f'A float. sequence, "x", or "y" are valid `axis` values but a value of {axis} was provided which is a {type(axis)}.')

	def transfer_matrix(self) -> ArrayLike:
		r"""Transfer matrix for ray propogation.

		Notes
		-----
		The current ray vector has no dedicated homogeneous coordinate, so
		the dipole's constant steering term is carried by the ``I`` column.
		This matches rays from ``Source``, where ``I`` is initialized to 1.

		Returns
		-------
		xp.ndarray
			Transfer matrix with drift and dipole steering terms.
		"""
		K = self.strength

		# Apply a calibration
		if self.calibration is not None:
			if isinstance(self.calibration,(int,float)):
				c = self.calibration
				K *= c
			else:
				c,p = self.calibration
				K = K**p * c
		
		# Project the strength
		Kx = K * xp.cos(self.phi)
		Ky = K * xp.sin(self.phi)

		if self.length == 0:
			self.tilt_x = Kx
			self.tilt_y = Ky
		else:
			self.tilt_x = Kx * self.length
			self.tilt_y = Ky * self.length

		return fix_mat_dims(xp.eye(4),["x","xt","y","yt"])

		#m = xp.zeros((7,8))
		#xp.fill_diagonal(m, 1, wrap=False)
		#
		#if self.length == 0:
		#	m[2, 7] = Kx
		#	m[3, 7] = Ky
		#else:
		#	L = self.length
		#
		#	# Drift terms
		#	m[0, 2] = L   # x <- ux
		#	m[1, 3] = L   # y <- uy
		#	m[4, 4] = 1.0 # z stays identity (already set)
		#
		#	# z advance
		#	m[4, 7] = L
		#
		#	# Angular kicks
		#	m[2, 7] = Kx * L
		#	m[3, 7] = Ky * L
		#
		#	# Position offsets (affine)
		#	m[0, 7] = 0.5 * Kx * L**2
		#	m[1, 7] = 0.5 * Ky * L**2
		#
		#return m
	
	#def propagate_ray(self, r0:xp.ndarray,
	#				  z:float=None, z0:float=0) -> xp.ndarray:
	#	"""propagate an array through an element.
	#
	#	Parameters
	#	----------
	#	r0 : xp.ndarray
	#		List of rays with possible initial conditions (x, θx, y, θy, E).
	#	z : None | int | float | xp.ndarray, optional
	#		Positions in the element to propagate to by default None
	#	z0 : None | float, optional
	#		Initial position of the element, by default 0
	#
	#	Returns
	#	-------
	#	xp.ndarray
	#		List of propagated rays with initial condition (x, θx, y, θy, z, E)
	#	"""
	#	m = self.transfer_matrix()
	#	#print(f'm: {m.shape}')#FLAG
	#	ones = xp.ones((r0.shape[0], 1), dtype=r0.dtype)
	#	r0_aug = xp.concatenate([r0, ones], axis=1)
	#	#print(f'r0: { r0.shape} to {r0_aug.shape}')#FLAG
	#
	#	rf = xp.einsum('mn,in->im', m, r0_aug)
	#	return rf


class Lens(Element):
	"""Lens element class for round lenses / symmetric focusing. https://en.wikipedia.org/wiki/Ray_transfer_matrix_analysis#Thin_lens_example or Brown1983 page 105

		Parameters
		----------
		name : str, optional
			Name of the lens, by default ''
		length : float, optional
			thickness of the lens, which affects the rotation of the beam
		strength : float, optional
			lens strength, proportional to lens current I, or magnetic field strength B. strength^2 is propogational to 1/f
		calibration : list or float, optional
			if a float is provided, a linear scaling will be applied to strength
			if a list is provided, terms are used in a series: strength =A+B*nominal+C*nominal^(1/2)+D*nominal^(1/3)+...
		position : float, optional
			The position of the element along the z-axis, by default None
		rotation : bool, optional
			if set to False, lens rotation for finite-thickness lenses is overridden and turned off.
		"""
	def __init__(self, name:str='', length:float=0.,
				 strength:float=None, calibration:float=None,
				 position:float=None,focal_length:float=None,allow_diverging=False) -> SEASerializable:
		
		if length == 0: kind = 'Thin lens'
		else:		   kind = 'QLens'

		super().__init__(name=name,kind=kind)
		self._position = position
		self.length = length
		if strength is not None and length==0:
			self._focal_length = xp.inf if strength==0 else 1/xp.sqrt(strength) ; self.strength = 0
			print("WARNING: YOU SPECIFIED STRENGTH",strength,"BUT DID NOT SPECIFY A LENGTH. PLEASE USE FOCAL LENGTH INSTEAD: f =1/sqrt("+str(strength)+")="+str(self._focal_length))
		if strength is not None and length!=0: # TODO need to enforce condition of strength and length=0?
			self.strength = strength
		if focal_length is not None and length==0:
			self._focal_length = focal_length
		self.calibration = calibration
		self.rotation = 0
		self.allow_diverging = allow_diverging

	def transfer_matrix(self) -> xp.ndarray:
		r"""Transfer matrix for ray propogation.
		"""

		# HANDLE CALIBRATION SCALING
		if self.length == 0:
			f = self._focal_length ; K=0
		else:
			K=self.strength ; f=0

		if self.calibration is not None:
			# linear scaling from mA (lens current) to lens strength?
			if isinstance(self.calibration,(int,float)):
				c = self.calibration
				K *= c
			else:
				Kvals = [self.calibration[0]] + [ v*K**(1/(i+1)) for i,v in enumerate(self.calibration[1:]) ]
				K = sum( Kvals ) #; print("lens","calibration",self.calibration,"strength",self.strength,"Kvals",Kvals)

		# FINITE LENGTH LENS, ZERO STRENGTH = DRIFT (try inserting a zero-strength lens and seeing if the result changes)
		if self.length==0 and f==xp.inf or self.length>0 and K==0:
			m = xp.eye(4) # IDENTITY MATRIX, OR DRIFT-EQUIVALENT
			m[0,1]=self.length
			m[2,3]=self.length
			self.rotation = 0
			return fix_mat_dims(m,["x","xt","y","yt"])

		# THIN LENS, NO ROTATION (thick lens math will have sine term going to zero)
		if self.length==0:
			if not self.allow_diverging:
				f = abs(f)
			#sign = -1*xp.sign(K) # sign allows negative calibration to give you diverging beams???
			X=xp.asarray([[    1   , 0 ],
					     [ -1/f , 1 ]])
			Y=xp.asarray([[    1   , 0 ],
						 [ -1/f , 1 ]])
			self.rotation = 0
			return xp.matmul( fix_mat_dims(X,["x","xt"]) , fix_mat_dims(Y,["y","yt"]) )

		# THICK LENS, FINITE K (zero K will have iK going to infinite)
		kL=K*self.length ; iK=1/K
		C=xp.cos(kL) ; S=xp.sin(kL)
		#XY=xp.asarray([	[ C**2  , iK*S*C  ,   S*C  , iK*S**2 ],	# Brown1983 page 105
		#				[-K*S*C ,  C**2   ,-K*S**2 ,   S*C   ],	# similar to standard
		#				[ -S*C  ,-iK*S**2 ,   C**2 , iK*S*C  ],	# [  1   0 ] but with
		#				[ K*S**2,  -S*C   , -K*S*C ,  C**2   ]] )# [ -1/f 1 ] rotation
		XY= xp.asarray([[  C , iK*S ,  0  ,  0   ],				# Brown1983 page 106
						[-K*S,  C   ,  0  ,  0   ],				# alternate definition
						[  0 ,  0   ,  C  , iK*S ],				# M = R(-KL) @ M_alt
						[  0 ,  0   ,-K*S ,  C   ]])
		R = xp.asarray([[  C ,  0 ,  S ,  0 ],					# | c -s | is normal,
						[  0 ,  C ,  0 ,  S ],					# | s  c | applied to x,y
						[ -S ,  0 ,  C ,  0 ],					# and xt,yt independently
						[  0 , -S ,  0 ,  C ]])					# here, flip signs to -KL
		#print(xp.matmul(R,XY2)-XY)
		#XY = xp.matmul(R,XY) # *xp.asarray([[1,1,0,0],[1,1,0,0],[0,0,1,1],[0,0,1,1]])

		# TWP 2026-05-12: new procedure: never rotate, but Element.propagate_ray will track rotation angle JK SEE BELOW
		#if not self.rotation:
		#	XY = XY2
		#else:
		#XY = xp.matmul(R,XY2)
		#	zeroer=xp.asarray([[1,1,0,0],[1,1,0,0],[0,0,1,1],[0,0,1,1]])
		#	XY*=zeroer
		#print("lens",self.name,"adds rotation",kL)
		# TWP 2026-07-23: upon discussion with Eric, we decided to always rotate. R is still tracked to allow you to return to the rotating reference frame for the purposes of quick-and-easy plane detection etc, although that stuff should be improved too (e.g., once we add aberrations, we will need to look for a beam waist. interpolate between drift endpoints, calculate Diameter(z) from all rays, d^2 diameter / dz^2 tells you where the beam is at a minimum diameter. check bundles of rays for diffraction planes?)
		XY = xp.matmul(R,XY)
		self.rotation = -kL
		M = fix_mat_dims(XY,["x","xt","y","yt"])
		return M

	@property
	def focal_length(self):
		# if thin lens, simply return _focal_length
		if self._focal_length !=0 and self.length==0:
			return self.focal_length
		# otherwise, calculate from angle of ray exiting transfer_matrix
		columns = [ columnByName(k) for k in ["x","xt","y","yt"] ]
		M = self.transfer_matrix()[columns,:][:,columns]
		r0 = [1,0,1,0] # parallel starting ray
		r1 = xp.matmul(M,r0)
		# positions and angles exiting lens
		x = xp.sqrt(r1[0]**2+r1[2]**2) ; xt = xp.sqrt(r1[1]**2+r1[3]**2)
		return x/xt # f = x/theta

	# unlike below(?), here we'll *measure* focal length at the current K=I*C and L, then adjust C and L to preserve focal length and set beam rotation (K*L) to match R in radians at this current I.
	def get_C_L_from_rotation_at_I(self,I,R):
		from scipy.optimize import minimize
		print(self.name,I,R)
		def FR(C,L):
			new = Lens(strength = I, calibration = C, length = L) # TODO now we have lens.focal_length property, should use that instead of fresh calculation
			#columns = [ columnByName(k) for k in ["x","xt","y","yt"] ]
			#M = new.transfer_matrix()[columns,:][:,columns]
			#r0 = [1,0,1,0] # parallel starting ray
			#r1 = xp.matmul(M,r0)
			#x = xp.sqrt(r1[0]**2+r1[2]**2) ; xt = xp.sqrt(r1[1]**2+r1[3]**2)
			#f = x/xt # f = x/theta
			f = new.focal_length
			M = new.transfer_matrix()
			rot = new.rotation
			return f,rot
		f0,_ = FR(self.calibration,self.length)	# initial focal length
		print("currently focuses to",f0)
		def dz(vals):
			f,rot = FR(*vals)
			return ((f-f0)/f0)**2 + ((R-rot)/R)**2
		res = minimize(dz,x0=(self.calibration,self.length))
		print(res)
		f,rot = FR(*res['x'])
		print( "fitted focuses to",f,"and with rotation of",rot )
		return res['x']

	def calibration_from_f_and_I(self,f,I,rotationPerAmp=None):
		print("for lens",self.name,"seeking a calibration factor C, which focuses strength",I,"to focal length",f,"and rotationPerAmp",rotationPerAmp)
		# noting xt=f(x) cell from matrix is -1/f or -K*sin(K*L)*cos(K*L):
		# APPROXIMATION: noting that at small angle, sin(K*L) ≈ K*L and cos(K*L) ≈ 1.
		# if K=C*I (strength = linear scaling * electrical current)
		# 1/f ≈ K²L = (C*I)² L  --> C = √(1/f/L)/I
		# and if length L is not fixed, but rotationPerAmp is given:
		# R = K*L = C*I*L --> R/I = C*L
		# BEWARE: inaccurate for large L. we really need to "solve for" C
		if self.length == 0:
			self.calibration = xp.sign(f)*xp.sqrt(1/abs(f))/I # 1/f = K² = (I*C)²,
			return
		self.calibration = xp.sqrt(1/f/self.length)/I
		# NOT AN approximation: 1/f = K*S*C = (C*I)*sin(C*I*L)*cos(C*I*L)
		# is there an analytical solution?
		# trig identity: sin(x)*cos(y) = 1/2*(sin(x+y)+sin(x-y)) so if x=y, sin(x)*cos(x) = 1/2*sin(2*x)
		#1/f = (C*I)*½*sin(2*C*I*L) idk how to solve this lol. if all you have is a hammer (scipy minimize) everything looks like a nail (a minimization problem)
		# OPE: 1/f = K*sin, not 1/f = K*sin*cos. (note how the alternate form of M from Brown1983 eliminated the cos. the cos is there for rotation)
		from scipy.optimize import minimize
		if rotationPerAmp is None:
			def dz(C):
				#return ( C*I*xp.sin(C*I*self.length)*xp.cos(C*I*self.length)-1/f )**2 # 1/f = K*S*C = (C*I)*sin(C*I*L)*cos(C*I*L)
				return (C*I*xp.sin(C*I*self.length)-1/f)**2	# 1/f = K*S = (C*I)*sin(C*I*L)
			x0 = self.calibration
			self.calibration = minimize(dz,x0=x0)['x'][0]
		else:
			def dz(CL):
				#print(CL)
				C,L=CL
				#return ( C*I*xp.sin(C*I*L)*xp.cos(C*I*L)-1/f )**2 + ( rotationPerAmp-C*L )**2
				return ( C*I*xp.sin(C*I*L)-1/f )**2 + ( rotationPerAmp+C*L )**2 # rad/A so don't multiply by I
			x0 = ( self.calibration, self.length )
			self.calibration,self.length = minimize(dz,x0=x0)['x']
			#import matplotlib.pyplot as plt
			#Cs = xp.linspace(0,4*x0,100) ; Ys=Cs*I*xp.sin(Cs*I*self.length)#*xp.cos(Cs*I*self.length)
			#plt.plot(Cs,Ys) ; plt.plot(Cs,[1/f]*100) ; plt.show()
		print("for lens",self.name,"found calibration factor",self.calibration,"and len",self.length)

		#print("calibration_from_f_and_I found",self.calibration,"from starting guess",x0,dz(self.calibration))

class Prism(Element):
	def __init__(self, name:str='', 
				 position:float=None, length:float=0.,
				 radius:float=None, angle:float=45., w:float=1., g:float=1., k1:float=0.,
				strength:float=0, calibration:float=None) -> SEASerializable:
		"""Prism.

		Parameters
		----------
		name : str, optional
			Name given to the lens, by default ''
		position : float, optional
			The position of the element along the z-axis, by default 0
		length : int, optional
			Length of the element, by default 0
		strength : float, optional
			Defined as the field strength (related to inverse focal length,
			see equations in brown1983), by default 0
			Note this in not the focusing strength (K) and is simply f.
			A thin lens is defind as KL=-1/fas L goes to zero.
		calibration : float, optional
			Currnet calibration of the lens in units of ???/A, by default None
		label : bool, optional
			If the element should be labeled when plotted, by default False
		print_fancy : bool, optional
			If a fancy table should be used when printed, by default True

		To do
		-----
		#TODO: Include dispersion (R16). See Egerton eq 2.14
		#TODO: Include other spectrometer aberrations. See Egerton eq 2.13
		"""

		self.angle = angle
		self.angle_rad = xp.deg2rad(angle)
		if radius is not None and length is not None:
			raise ValueError('Only specify the length or radius as providing both with radius is not unique.')
		elif radius is not None:
			length = self.angle_rad * radius
		elif length is not None:
			radius = length/self.angle_rad
		else:
			raise ValueError('Either radius or length need to be specified.')

		super().__init__(name=name,kind='Prism')
		self._position = position
		self.length = length
		self.strength = strength
		self.calibration = calibration
		self.radius = radius
		self.w = w
		self.g = g
		self.K1 = k1

	def focus_matrix(self,
					 #type='Hills' TODO: Add `type` in paramaters to describe the type of transfer matrix. Hill's, Twiss, etc.
					 ) -> xp.ndarray:
		r"""Transfer matrix for the entrance/exit surfaces of the spectrometer used for ray propogation.
		"""
		m = xp.eye(6)

		if self.strength!=0:
			m[1,0] = xp.tan(self.strength) / self.radius
			if self.K1 == 0:
				m[1,0] = - xp.tan(self.strength) / self.radius
			else: #include fringe fields
				psi = (self.g/self.radius) * self.K1 * (1+xp.sin(self.strength)**2)/xp.cos(self.strength)
				m[2:4,2:4] = - xp.tan(self.strength - psi) / self.radius
		else: #drif
			pass

		return fix_mat_dims(m,["x","xt","y","yt","z","E"])
	
	def bending_matrix(self,
					   s:float,
					   #type='Hills' TODO: Add `type` in paramaters to describe the type of transfer matrix. Hill's, Twiss, etc.
					   ) -> xp.ndarray:
		r"""Transfer matrix for the bending of the spectrometer used for ray propogation.
		"""
		m = xp.eye(6)
		u = s/self.radius

		if self.strength!=0:
			m[0,0] = xp.cos(u)
			m[1,1] = xp.cos(u)
			m[0,1] = self.radius * xp.sin(u)
			m[1,0] = - xp.sin(u)/self.radius
		else:
			m = Drift.transfer_matrix(s)

		return fix_mat_dims(m,["x","xt","y","yt","z","E"])
		
	def transfer_matrix(self) -> xp.ndarray:
		r"""Transfer matrix for ray propogation.
		"""
		
		m_focus1 = self.focus_matrix()
		m_bend   = self.bending_matrix(self.strength)
		m_focus2 = self.focus_matrix()

		m = m_focus2 @ m_bend @ m_focus1

		return fix_mat_dims(m,["x","xt","y","yt","z","E"])


element_list = ["Element"] + [subclass.__name__ for subclass in Element.__subclasses__()]
