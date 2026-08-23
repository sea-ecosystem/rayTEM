import numpy as xp
from numpy.typing import ArrayLike
from typing import List

flag_gpu = False
import pickle
import sys,inspect,os,datetime,shutil

from .postprocessing import plot2D,findPlanes,zFromFractional,measureAtZ
from .elements import Element,Source,Drift,Lens,Dipole,Quadrapole,columnByName,Aperture,convention,_propagate_method_name
from typing import Literal
from .seashells import SEASerializable

from copy import deepcopy


def _stack_wavefields(planes, name="wavefield"):
	"""Stack a list of 2D wavefield Signals into one calibrated 3D wavefield Signal.

	Reads each per-plane wavefield and assembles a single ``(n_planes, ny, nx)``
	complex Signal whose z axis is unstructured (explicit per-plane positions) and
	whose transverse ``(x, y)`` calibration and wavelength are shared.

	Parameters
	----------
	planes : Sequence
		Per-plane wavefields (Signals or ``seashells._Wavefield`` fallbacks), all on
		the same transverse grid.
	name : str, optional
		Name for the stacked Signal, by default ``"wavefield"``.

	Returns
	-------
	Signal or seashells._Wavefield
		The stacked wavefield.

	Related
	-------
	seashells.make_wavefield_signal, seashells.read_wavefield
	"""
	from .seashells import read_wavefield, make_wavefield_signal
	datas=[] ; zs=[] ; dx=dy=wavelength=None
	for p in planes:
		d, dx, dy, wavelength, z = read_wavefield(p)
		datas.append(d) ; zs.append(z if z is not None else 0.0)
	stacked = xp.asarray(datas)
	return make_wavefield_signal(stacked, dx, dy, wavelength, z=xp.asarray(zs), name=name)


def _stack_scaled_wavefields(planes, name="scaled wave"):
	"""Stack per-plane scaled wavefields into one ``SignalSet`` with frame companions.

	Reads each scaled plane (reduced field U plus its ``s``/``R``/``τ``/``z``
	frame state) and assembles the Signal-backed result of a scaled-Fresnel run:
	the ``(n_planes, nη, nξ)`` U stack as main Signal with companion 1-D Signals
	``s(z)``, ``R(z)``, ``tau(z)`` sharing the unstructured plane-z axis — the
	same pattern as :func:`seashells.make_rays_signalset`.

	Parameters
	----------
	planes : Sequence
		Per-plane scaled wavefields (Signals or ``seashells._ScaledWavefield``
		fallbacks), all on the same ξ/η grid.
	name : str, optional
		Name for the SignalSet, by default ``"scaled wave"``.

	Returns
	-------
	SignalSet or None
		The stacked scaled-wave SignalSet, or ``None`` when sea_eco is absent
		(the per-plane list on ``._wave_scaled_planes`` remains usable).

	Related
	-------
	seashells.make_scaled_wave_signalset, seashells.read_scaled_wavefield
	"""
	from .seashells import read_scaled_wavefield, make_scaled_wave_signalset, scaled_frame_tag
	datas=[] ; ss=[] ; Rs=[] ; taus=[] ; zs=[] ; tags=[] ; dxi=deta=wavelength=None
	for p in planes:
		U, dxi, deta, wavelength, s, R, tau, z = read_scaled_wavefield(p)
		datas.append(U) ; ss.append(s) ; Rs.append(R) ; taus.append(tau)
		zs.append(z if z is not None else 0.0)
		tags.append(scaled_frame_tag(p))
	return make_scaled_wave_signalset(xp.asarray(datas), dxi, deta, wavelength,
									  s=ss, R=Rs, tau=taus, z=zs, tags=tags, name=name)


def _scaled_wave_cross_section(planes, ax, named_positions=None, crossovers=None,
							   image_planes=None, title=None):
	"""Draw the |ψ(x, y=0, z)| cross-section of a scaled-wave run into an axis.

	The wave analog of the geometric ray diagram: each logged plane is
	reconstructed to physical coordinates on its native grid (``Δx = |s|·Δξ``,
	so the pixel spans nm at the foci to µm at the detector), its centre row
	``|ψ(x, 0)|`` is normalized to its peak and resampled onto one common x
	axis, and the planes are rendered as a z–x pcolormesh. Element positions
	are annotated as white dashed lines with labels and crossover (focal)
	planes as cyan dotted lines — the same overlays the ray diagram carries.

	Parameters
	----------
	planes : Sequence
		Per-plane scaled wavefields (Signals or ``_ScaledWavefield`` fallbacks)
		from a scaled/hybrid run, in any z order.
	ax : matplotlib axis
		Axis to draw into.
	named_positions : dict, optional
		``{label: z}`` element annotations (white dashed), by default none.
	crossovers : Sequence[float], optional
		Crossover z positions of the family the wave's own frame follows —
		the diffraction / back-focal planes for the usual flat-wavefront seed
		(cyan dotted), by default none.
	image_planes : Sequence[float], optional
		The conjugate family: image-plane z positions from
		:meth:`Microscope.conjugate_planes` (magenta dashed), by default none.
	title : str, optional
		Axis title, by default none.

	Returns
	-------
	None
		Draws into ``ax``.

	Related
	-------
	Microscope.show : Calls this for ``kind='wave-scaled'/'wave-hybrid'``
		when no ``plane`` is selected.
	waveoptics.reconstruct_physical_wave : The per-plane reconstruction.

	Notes
	-----
	Planes are individually peak-normalized — the panel shows the beam's
	shape and envelope, not absolute intensity (which spans many orders of
	magnitude between a focus and the detector). A column built with finely
	subdivided drifts yields a smoother section (see
	``examples/04_scaledWave_basic_column.py``).
	"""
	from .seashells import read_scaled_wavefield
	from .waveoptics import reconstruct_physical_wave
	recon = []
	for p in planes:
		U, dxi, deta, lam, s, R, tau, z = read_scaled_wavefield(p)
		psi, dx, dy = reconstruct_physical_wave(U, dxi, deta, lam, s, R)
		recon.append((z if z is not None else 0.0, psi, dx))
	recon.sort(key=lambda r: r[0])
	zs = xp.array([r[0] for r in recon])
	z_edges = xp.concatenate([[zs[0] - 1e-4], (zs[:-1] + zs[1:]) / 2, [zs[-1] + 1e-4]]) * 1e3
	n = recon[0][1].shape[1]
	half = max(abs(r[2]) * n / 2 for r in recon)
	x_common = xp.linspace(-half, half, 600)
	prof = xp.zeros((len(recon), x_common.size))
	for i, (z, psi, dx) in enumerate(recon):
		x = (xp.arange(n) - n // 2) * dx
		row = xp.abs(psi[psi.shape[0] // 2, :])
		prof[i] = xp.interp(x_common, x, row / row.max(), left=0, right=0)
	x_edges = xp.linspace(-half, half, x_common.size + 1) * 1e6
	ax.pcolormesh(z_edges, x_edges, prof.T, cmap="magma", shading="flat")
	ax.set_xlabel("z (mm)")
	ax.set_ylabel("x (µm)")
	if named_positions:
		for label, zp in named_positions.items():
			if not label:			# unnamed elements share one blank key; skip them
				continue
			ax.axvline(zp * 1e3, color="w", lw=0.6, ls="--", alpha=0.6)
			ax.text(zp * 1e3, half * 1e6 * 0.95, label, color="w", rotation=90,
					ha="right", va="top", fontsize=7)
	if crossovers is not None and len(crossovers):
		for zc in crossovers:
			ax.axvline(zc * 1e3, color="cyan", lw=0.8, ls=":", alpha=0.9)
			ax.text(zc * 1e3, -half * 1e6 * 0.95, "crossover", color="cyan",
					rotation=90, ha="right", va="bottom", fontsize=7)
	if image_planes is not None and len(image_planes):
		for zi in image_planes:
			ax.axvline(zi * 1e3, color="magenta", lw=0.8, ls="-.", alpha=0.9)
			ax.text(zi * 1e3, -half * 1e6 * 0.95, "image", color="magenta",
					rotation=90, ha="left", va="bottom", fontsize=7)
	if title:
		ax.set_title(title)


class MicroscopeSection(SEASerializable):
	"""MicroscopeSection class represents a portion of a microscope, and contains multiple Elements. propagation through a Section results in propagation through individual Elements.

		Parameters
		----------
		name : str, optional
			Name of the Section, by default ''
		elements : list
			ordered list of Element objects (or inheriting classes: Source, Drift, Lens, etc). each Element's "position" attribute is used to determine the position of the Element within the Section *OR* Drift Elements can be inserted to define the spacing. elements=[Source,Lens,Lens] (with each lens position defined) will insert Drifts as appropriate. elements=[Source,Drift,Lens,Drift,Lens] (without lens positions defined) will simply stack elements in order, with positions determined by all previous elements' thicknesses.
		position : float, optional
			The position of the Section along the z-axis, by default None
		ignoreLensThickness : bool, optional
			if set to True, all lenses are set to zero thickness??
		"""

	def __init__(self, name:str='',
				 elements:ArrayLike=None, # list of Elements, or list of dicts
				 position:float=0., ignoreLensThickness=False ) -> SEASerializable:
		self.name = name
		#if isinstance(elements[0],dict):
		#	self.elements = []
		#else:
		self.elements = elements
		self.position = position
		self.ignoreLensThickness = ignoreLensThickness
		self.length = 0 #= self.position #xp.sum([e.length for e in self.elements])
		self.rays = None
		self.I = None		# per-plane, per-ray intensity (parallel to self.rays, not a ray coordinate)
		self.R = None		# per-plane, per-ray cumulative Larmor rotation (radians)
		self.mu = None					# per-plane mean state vector (beam-envelope mode)
		self.covariance_matrix = None	# per-plane covariance matrix (beam-envelope mode)
		self.wave = None				# per-plane complex wavefield Signal (wave-optics mode)
		self.wave_scaled = None			# per-plane scaled-Fresnel SignalSet (U + s/R/tau companions)

		
		if self.elements is None or (self.elements)==0:
			return

		new = []
		for n,ele in enumerate(elements):
			#print("process element",n,"=",repr(ele))
			#print("self.length",self.length,"adding ele",ele.kind,ele.name,ele.position,ele.length)
			if ele.position is None:						# e.g. pass Lens(l1),Drift(l2),Lens(l3) --> Drift.position=l1, Drift.position=l1+l2
				#print("(no position, add to end)")
				ele._position = self.length
			# COMMENTING OUT GAP/OVERLAP HANDLING HERE, AND SWITCHING TO USING "repair" FUNCTION AT THE END
			# SANITY CHECK: gaps or overlaps: between this element's position and end of previous
			#dz = ele.position - self.length
			#tol = 1e-7 # Why a relatively loose 1e-7? float imprecision isn't this bad, but json reload is rounded
			# GAP, WITHIN TOLERANCE, LENGTHEN PREVIOUS ELEMENT
			#if 0 < dz < tol:
			#	new[-1].length += dz
			# GAP, OUT OF TOLERANCE, ADD DRIFT
			#if dz > tol:
			#	new.append( Drift(length=dz,position=ele.position-dz) )
			#	self.length += dz
			# OVERLAP, WITHIN TOLERANCE, SHORTEN PREVIOUS ELEMENT
			#if 0 > dz > -tol:
			#	new[-1].length += dz
			# OVERLAP, OUT OF TOLERANCE, RAISE A WARNING
			#if dz < -tol:
			#	print('WARNING: previous Element ('+str(elements[n-1])+') overlaps with specified Element position '+str(ele))
			new.append(ele)
			# SANITY CHECK: if there is an "overlap" between end of previous element and this element
			#if self.length > ele.position:
			#	dz = ele.position - self.length
			#if self.length-ele.position > 1e-7: # if length > position, but allows for float imprecision.
			#	print('WARNING: previous Element ('+str(elements[n-1])+') overlaps with specified Element position '+str(ele))
			if ignoreLensThickness and ele.kind in ['Thin lens','QLens','Thin quad','Quad']:
				continue
			#print("increment length by",getattr(ele,"length",0))
			self.length += getattr(ele,"length",0)
			#print("new length = ",self.length)
		self.elements = new

		repair(self)

	#####################################
    # region: Dunders
	
	def __delitem__(self,item):
		if isinstance(item,str):
			item = self.index(item)
		if self.elements[item-1].kind != "Drift":
			print("WARNING: unable to delete "+str(element)+" at "+str(index)+" (preceeding element must be a Drift???)")
		self.elements[item-1].length += getattr(self.elements[item],"length",0)
		del self.elements[item]

	# TWP 2025-11-05: allow indexing of the assembly by name: section["PL1"] should return the section by that name! see removed_private_instrument_tree/PRIVATE_INSTRUMENT/fine_PLs.py.
	# 2026-02-05: also allow slicing by name: section["sample":] should return a new section with all elements including and after "sample"
	# 2026-06-18: and slicing by z position: section[2.5:] should return a new section trimmed to z>=2.5?
	def __getitemOLD__(self, item):
		#print("section __getitem__",item)
		# REFERENCE TO SINGLE ITEMS
		if isinstance(item,str):	# convert "PL1" into an integer index
			#print("index lookup, single item")
			item = self.index(item)
		if isinstance(item,int):
			#print("simple index lookup, returning reference to element")
			return self.elements[item]
		# SLICES, POTENTIALLY MULTIPLE ITEMS, ALWAYS RETURN A COPY
		if isinstance(item,slice):	# convert "sample:" (which results in "item" being a slice) to an integer-indexed slice, e.g. slice(3,None,None)
			a,b,n=item.start,item.stop,item.step
			trim_first = 0 ; trim_last = 0
			a,b,n=[ self.index(v) if isinstance(v,str) else v for v in [a,b,n] ] # convert "PL1:" to whatever the index is for PL1
			if isinstance(a,float):
				trim_first = a
				positions = xp.asarray([ e.position for e in self.elements ])
				a = xp.where(positions<a)[0][-1]
			#if isinstance(b,float): # TODO finish implementing
			#	trim_last = self.elements[b].length-
			#	b=int(np.ceil(b))
			item = slice(a,b,n)
		#print("returning copied slice")
		ret = self.copy().elements[item]
		if trim_first > 0:
			ret[0].length-=trim_first ; ret[0]._position+=trim_first
		#if trim_last > 0:	# TODO finish implementing
		p0 = ret[0].position
		for i,e in enumerate(ret):
			ret[i]._position -= p0		# shift all element positions so first is at zero

		if isinstance(ret,list):
			#print("CONSTRUCT NEW SECTION WITH\n",ret)
			return MicroscopeSection(name=self.name,elements=ret,position=self.position)
		return ret

	def __getitem__(self, item):
		# RETURN A REFERENCE TO A SINGLE ITEM, BY NAME OR INDEX
		if isinstance(item,str):	# convert "PL1" into an integer index
			#print("index lookup, single item")
			item = self.index(item)
		if isinstance(item,int):
			#print("simple index lookup, returning reference to element")
			return self.elements[item]
		# RETURN A COPY OF A SLICE (SUBSET OF ITEMS), BY NAME, INDEX, OR Z-LOCATION
		if isinstance(item,slice):
			# "sample:" --> a="sample",b=None,c=None. or "2.5:11.0" --> a=2.5,b=11.5,c=None.
			a,b,n=item.start,item.stop,item.step
			trim_first = 0 ; trim_last = 0
			positions = xp.asarray([ e.position for e in self.elements ])
			if isinstance(a,float):
				i = xp.where(positions<a)[0][-1] 	# index of last element which starts prior to z=a, i.e., spans across z=a
				trim_first = a-positions[i]			# we'll need to cut off dz from that element (e.pos=4.5, a=5, trim 0.5)
				a = i								# index slicing of self.elements will begin with the element spanning z=a
			if isinstance(b,float):
				i = xp.where(positions<b)[0][-1]
				trim_last = b-positions[i]
				b=i+1								# we want to include the element spanning z=b, hence +1
			if isinstance(a,str):
				a = self.index(a)
			if isinstance(b,str):
				b = self.index(b)
			item = slice(a,b,n)
		#print("returning copied slice")
		# DEEP COPY MYSELF, SLICE ELEMENTS, MAKE ADJUSTMENTS TO POSITIONS AND LENGTHS
		new = self.copy()
		new.elements = new.elements[item]		# list of elements, deeeeeep-copied (including copies of all elements)
		# front-side trimming: first element's length (if it was chopped midway), and positions of all elements (if a!=0)
		if trim_first > 0:
			new.elements[0].length-=trim_first			# trim first element
			new.elements[0]._position+=trim_first		# then "scoot it back" so it ends where it ended previously
		p0 = new.elements[0]._position
		for i,e in enumerate(new.elements):
			new.elements[i]._position -= p0		# shift all element positions so first is at zero
		if trim_last > 0:
			new.elements[-1].length-=trim_last
		# scoot ALL elements forwards so element 0 starts at 0
		p0 = new.elements[0].position
		for i,e in enumerate(new.elements):
			new.elements[i]._position -= p0		# shift all element positions so first is at zero
		# and finally, update new's length
		new.length = new.elements[-1].position+new.elements[-1].length
		return new

	def __setitem__(self, item, value): # TWP 2026-02-04: allow setting by assembly name or index. sec1[0]=sec2[0] should work
		if isinstance(item,str):
			names = [ e.name for e in self.elements ]
			item = names.index(item)
		self.elements[item] = value

	def __repr__(self) -> str:
		if self.elements is None:
			return ''
		else:
			columns=['name', 'kind', 'position', 'length', 'strength', 'calibration']
			reps = []
			for e in self.elements:
				reps.append([])
				values = [ getattr(e,c,"") for c in columns]
				for v in values:
					if isinstance(v,float):
						v=xp.round(v,7)
					v=str(v) ; v=v+" "*(8-len(v)) ; v=v[:8]
					reps[-1].append(v)
			columns = [c+" "*(8-len(c)) for c in columns ]
			columns = [ c[:8] for c in columns ]
			rows = [ " ".join(columns) ] + [ " ".join([str(v) for v in rep ]) for rep in reps ]
			return "\n".join(rows)
		
	def __str__(self):
		if self.name is None or self.name=='': name = 'Unamed'
		else: name = self.name
		return f'{name} (Section)'

	def __len__(self):
		return len(self.elements)


	# endregion
    #####################################

	#####################################
    # region: SEASerializable integration

	def _get_tree_html(self, recursive_level: List[str] = 0,
					   exclude_keys: List[str] = ['rays', 'I', 'R', 'mu', 'covariance_matrix', 'wave', 'wave_scaled'],
                       exclude_hidden: bool = True,
                       exclude_properties:bool = False,
                       promote_itterable_keys: List[str] = ['elements']
                       ) -> str:
		return super()._get_tree_html(recursive_level, 
                                     exclude_keys=exclude_keys, 
                                     exclude_hidden=exclude_hidden,
                                     exclude_properties=exclude_properties,
                                     promote_itterable_keys=promote_itterable_keys
                                     )

    # endregion
    #####################################

	# given a string for an element name, return the index of that element
	def index(self,item):
		names = [ e.name for e in self.elements ]
		return names.index(item)

	# TWP 2026-02-05 allow element insertion by index OR coordinate ("add a lens midway through this drift section at z=etc")
	def insert(self,index,element): # TODO bug: if we insert a huge drift ("big enough to fill the space") it's just a huge drift. we should update the length based on the space it'll fit. either here, or in Section.insert.
		if isinstance(index,int):				# basic list insertion: section.insert(0,newsurce) places newsource at the beginning
			if index == len(self.elements):
				self.length+=element.length
			self.elements.insert(index,element)
		else:									# coordinate-based insertion: section.insert(25.0,newlens) places newlens in drift that spans 25.0
			for i,ele in enumerate(self.elements): # "looking for element spanning 25.0: 5th element is a Drift which goes from 21.0 to 30.0"
				# inserted Source is ALWAYS first (idk where else you'd put one).
				# other inserted elements may only go inside a Drift (e.pos < z & e.pos+D.len > z & e.kind==Drift),
				if element.kind=="Source" or (ele.position<=index and ele.position+getattr(ele,"length",0)>index and ele.kind=="Drift"):
					#print("INSERTING ELEMENT",element.name,"AT",index,"(",ele,ele.position,ele.length,")","AT POSITION",i)
					elementlength=0 if self.ignoreLensThickness else getattr(element,"length",0)
					l1=index-ele.position ; l2=getattr(ele,"length",0)-elementlength-l1 # "this drift needs to be length 4.0, and we'll need another drift after the insertion"
					#print("PRE DRIFT",l1,"+ ELEMENT",element.length,"+ POST DRIFT",l2,"=",ele.length)
					self.elements[i].length=l1			# "shorten" initial drift
					element._position = index			# update new element's position
					self.elements.insert(i+1,element)	# add new element
					if l2>0:							# add following drift
						self.elements.insert(i+2,Drift(length=l2,position=index+elementlength))
					if l1==0:							# possible drift1 is length zero, so delete it
						del self.elements[i]
					break
			else:
				print("WARNING: unable to insert "+str(element)+" at "+str(index)+" (coordinate may be out of bounds, or non-drift element)")

	def append(self,element):
		self.insert(len(self.elements),element)

	def move_element(self,elementName,z=None,dz=None,allow_unsafe=False): # TODO are we still making the massive assumption that we're adjusting non-first non-last element positions?? are all edge cases handled?
		i=self.index(elementName)
		# always move by "dz". for a MicroscopeSection object, assume z is relative to beginning of the sectio
		if z is not None:
			dz = z-self.elements[i].position
		if i==0 and dz>0:	# 0th element, add preceeding drift, increment i to keep track. subsequent code will inflate Drift by dz
			self.elements.insert(0,Drift(position=0,length=0))
			i+=1
		elif i==0:
			raise NotImplementedError("MicroscopeSection.move_element does not yet support backwards movement of the 0th element")
		self.elements[i]._position+=dz			# element position is updated
		self.elements[i-1].length+=dz			# previous element is lengthened
		if self.elements[i-1].length < 0 and i>2 and self.elements[i-2].kind == "Drift": # edge case: if two drifts in a row, and one is shortened to below zero length, simply combine them
			self.elements[i-2].length += self.elements[i-1].length
			del self.elements[i-1] ; i-=1
		if i+1<len(self.elements):
			self.elements[i+1]._position+=dz	# subsequent element is also moved
			self.elements[i+1].length-=dz		# subsequent element is shortened
			if self.elements[i+1].length<0:		# another edge case: what if we push L1 past L2? intermediate drift is now negative
				if allow_unsafe:				# for now (TODO) simply scoot all subsequent elements out, and lengthen the section
					self.elements[i+1].length+=dz
					for ii,e in enumerate(self.elements):
						if ii<=i+1:
							continue
						e._position += dz
					self.length += dz
				else:
					print("ATTEMPTING TO MOVE",elementName,"by",dz)
					print(repr(self))
					raise NotImplementedError("MicroscopeSection.move_element far enough to create negative length Drift is currently poorly handled. try allow_unsafe=True to bypass this error IFOF you know what you're doing (this will scoot all elements out)")
		else: # IF THIS IS THE LAST ELEMENT:
			if dz<0: # append Drift element if elementName is moved forwards...
				self.elements.append(Drift(length=-dz,position=self.elements[i]._position+self.elements[i].length))
			else:	# or lengthen section (dangerous!) if elementName is moved backwards...
				if not allow_unsafe:
					raise NotImplementedError("MicroscopeSection.move_element in +z is unsafe for last element, as it lengthens the MicroscopeSection. No good solution to this is implemented. try allow_unsafe=True to bypass this error IFOF you know what you're doing (this will lengthen the section)")
				self.length += dz
		#print(repr(self))

	def wobble(self,r0,elementIndex,func,kwargName,valRange,numSteps):
		vals=xp.linspace(valRange[0],valRange[1],numSteps)
		results=[]
		for v in vals:
			self.elements[elementIndex]=func(**{kwargName:v})
			rf=self.propagate_ray(r0)
			results.append(rf[-1,:,:]) # indices are: point in scope, which ray, which value (x,xt,y,yt...)
		return results

	@property
	def named_positions(self):
		positions = {}
		for e in self.elements:
			if e.name is None:
				continue
			p = e.position
			l = 0 if e.kind=="Drift" else getattr(e,"length",0)
			positions[e.name] = p+l/2 # mark the *center* of each element, since that's the optical center of a lens?
		return positions

	# returns nthElement,nthRay,xythetaetc
	def propagate_ray(self, r0:xp.ndarray=None,
					   I0:xp.ndarray=None, R0:xp.ndarray=None,
					   z: float = None,
					   verbose=False):
		"""Propagate rays through every element in the section, bottom-up.

		Intensity (``I``) and cumulative Larmor rotation (``R``) travel as separate
		parallel arrays rather than as ray coordinates; they are stored on
		``self.I`` / ``self.R`` alongside ``self.rays``. When chaining sections
		(see :meth:`Microscope.propagate_ray`) the exit ``I``/``R`` of one section
		seed ``I0``/``R0`` of the next so rotation and attenuation accumulate.

		Parameters
		----------
		r0 : xp.ndarray, optional
			Initial geometric ray table. If ``None`` and the first element is a
			``Source``, rays are generated from it; otherwise a ``UserWarning`` is raised.
		I0 : xp.ndarray, optional
			Per-ray intensity entering the section, shape ``(n_rays,)``. Defaults to ones.
		R0 : xp.ndarray, optional
			Per-ray cumulative rotation (radians) entering the section. Defaults to zeros.
		z : float, optional
			Position within an element to propagate to, by default ``None``.
		verbose : bool, optional
			Print per-element progress, by default ``False``.

		Returns
		-------
		xp.ndarray
			Geometric rays, shape ``(n_planes, n_rays, len(convention))``. The
			matching intensity and rotation are on ``self.I`` / ``self.R``.

		Raises
		------
		UserWarning
			If ``r0`` is ``None`` and the first element is not a ``Source``.
		"""
		#print("Section r0",r0)
		if r0 is None:
			if isinstance(self.elements[0], Source):
				r0 = self.elements[0].rays()
			else:
				raise UserWarning("First element is not a Source, and no r0 provided to propagate_ray. Please provide initial rays or ensure first element is a Source.")
		n_rays = len(r0)
		if I0 is None:
			I0 = xp.ones(n_rays)
		if R0 is None:
			R0 = xp.zeros(n_rays)
		ri=[r0] ; Ii=[I0] ; Ri=[R0]
		for i,ele in enumerate(self.elements):
			if verbose:
				print("propate:",ele.name,"@",ele.position,"x,y",xp.amax(ri[-1][:,columnByName("x")]),xp.amax(ri[-1][:,columnByName("y")])) #,"xt,yt",xp.amax(ri[-1][:,columnByName("xt")]),xp.amax(ri[-1][:,columnByName("yt")]))
			# intensity/rotation are evaluated relative to the incoming rays; rotation
			# must follow propagate_ray so thick-lens self.rotation is already set.
			ele_I  = ele.apply_intensity(Ii[-1], ri[-1])
			ele_ri = ele.propagate_ray(ri[-1], z=z)
			ele_R  = ele.apply_rotation(Ri[-1])
			#ele_ri[...,-2] += ele.position # TWP 2025/08/27 - do not add distance. drift already should update z
			#print(ele_ri.shape,r0.shape)
			if getattr(ele,"length",0) != 0 or ele.kind == "Aperture":
				ri.append(ele_ri[:,:]) ; Ii.append(ele_I) ; Ri.append(ele_R)
			else:
				ri[-1]=ele_ri[:,:] ; Ii[-1]=ele_I ; Ri[-1]=ele_R
		self.rays = xp.asarray(ri) # xp.swapaxes(xp.asarray(ri),0,1)
		self.I = xp.asarray(Ii)
		self.R = xp.asarray(Ri)
		return self.rays

	def propagate_moments(self, mu0:xp.ndarray=None, Sigma0:xp.ndarray=None,
						   z: float = None):
		"""Propagate beam moments (mean + covariance) through every element.

		The envelope-mode analog of :meth:`propagate_ray`: transports a mean state
		``mu`` and covariance ``Sigma`` element-by-element via
		:meth:`Element.propagate_moments`. Per-plane results are stored on
		``self.mu`` and ``self.covariance_matrix`` using the same append/replace
		logic as :meth:`propagate_ray` (a plane is logged after each finite-length
		element or aperture).

		Parameters
		----------
		mu0 : xp.ndarray, optional
			Initial mean state. If ``None`` and the first element is a ``Source``,
			it is seeded from :meth:`Source.moments`.
		Sigma0 : xp.ndarray, optional
			Initial covariance. Seeded from the ``Source`` when ``None``.
		z : float, optional
			Unused placeholder mirroring :meth:`propagate_ray`, by default ``None``.

		Returns
		-------
		xp.ndarray
			Per-plane covariance matrices, shape
			``(n_planes, len(convention), len(convention))``. Means are on ``self.mu``.

		Raises
		------
		UserWarning
			If moments are not provided and the first element is not a ``Source``.
		"""
		if mu0 is None or Sigma0 is None:
			if isinstance(self.elements[0], Source):
				mu0, Sigma0 = self.elements[0].moments()
			else:
				raise UserWarning("First element is not a Source, and no (mu0, Sigma0) provided to propagate_moments. Please provide initial moments or ensure first element is a Source.")
		mui=[mu0] ; Si=[Sigma0]
		for ele in self.elements:
			mu, S = ele.propagate_moments(mui[-1], Si[-1])
			if getattr(ele,"length",0) != 0 or ele.kind == "Aperture":
				mui.append(mu) ; Si.append(S)
			else:
				mui[-1]=mu ; Si[-1]=S
		from .seashells import make_covariance_signal
		self.mu = xp.asarray(mui)
		self.covariance_matrix = make_covariance_signal(xp.asarray(Si), self.mu[:, columnByName('z')],
														convention, name=(self.name or 'section') + ' covariance')
		return self.covariance_matrix

	def propagate_wave(self, wave0=None, mode:Literal['fixed','scaled','hybrid']='fixed',
					   s_min:float=1e-3, absorb:float=0.1,
				   crossover:Literal['flat','jump']='flat', rotate:bool=False):
		r"""Propagate a wavefield through every element in the section.

		The one wave-optics analog of :meth:`propagate_ray`, covering all three
		wave representations via ``mode`` (see :meth:`Element.propagate_wave`).
		Threads the wave element-by-element, logging a plane after each
		finite-length element or aperture (same append/replace logic as
		ray/moment propagation); on ``mode='hybrid'`` the interior frame-switch
		and crossover (focal) planes logged by the engine are inserted in z
		order as well.

		Results are stored by representation: ``mode='fixed'`` stacks the
		per-plane fields into a calibrated ``(n_planes, ny, nx)`` ``Signal`` on
		``self.wave``; ``'scaled'``/``'hybrid'`` stack the per-plane states
		into a ``SignalSet`` — U ``(n_planes, nη, nξ)`` plus companion
		``s(z)``/``R(z)``/``tau(z)`` (and ``frame``) Signals on the shared
		plane-z axis — on ``self.wave_scaled``.

		Parameters
		----------
		wave0 : Signal or seashells._Wavefield or seashells._ScaledWavefield, optional
			Initial wavefield in the representation matching ``mode``. If
			``None`` and the first element is a ``Source``, it is generated
			from :meth:`Source.wave` with the same ``mode``.
		mode : {'fixed', 'scaled', 'hybrid'}, optional
			Wave representation, by default ``'fixed'``.
		s_min : float, optional
			Backstop crossover guard for the scaled/hybrid paths, by default
			``1e-3``.

		Returns
		-------
		Signal or SignalSet or None
			The stacked result (also stored on ``self.wave`` /
			``self.wave_scaled``). Per-plane states are kept on
			``self._wave_planes`` / ``self._wave_scaled_planes`` for chaining
			by :class:`Microscope` (and usable even when sea_eco is absent and
			the SignalSet is ``None``).

		Raises
		------
		UserWarning
			If ``wave0`` is ``None`` and the first element is not a ``Source``.
		ValueError
			``mode='scaled'`` only: a single frame reaching its ``s = 0``
			crossover (use ``mode='hybrid'`` to switch frames through it).
		"""
		if wave0 is None:
			if isinstance(self.elements[0], Source):
				wave0 = self.elements[0].wave(mode=mode)
			else:
				raise UserWarning("First element is not a Source, and no wave0 provided to propagate_wave. Please provide an initial wavefield or ensure first element is a Source.")
		fi=[wave0]
		for ele in self.elements:
			interior = [] if mode == 'hybrid' else None
			f = ele.propagate_wave(fi[-1], mode=mode, s_min=s_min, log=interior, absorb=absorb, crossover=crossover, rotate=rotate)
			if interior:
				fi.extend(interior)		# frame switches / crossover planes, in z order
			if getattr(ele,"length",0) != 0 or ele.kind == "Aperture":
				fi.append(f)
			else:
				fi[-1]=f
		if mode == 'fixed':
			self._wave_planes = fi
			self.wave = _stack_wavefields(fi, name=(self.name or 'section') + ' wave')
			return self.wave
		self._wave_scaled_planes = fi
		self.wave_scaled = _stack_scaled_wavefields(fi, name=(self.name or 'section') + ' scaled wave')
		return self.wave_scaled

	def rays_signalset(self):
		"""Return a sea_eco ``SignalSet`` view of the traced rays (rays + I + R).

		Wraps the most recent ray-mode result (``self.rays``/``self.I``/``self.R``)
		as a calibrated ``SignalSet`` via the seashells seam, propagating first if
		needed. This is the Signal-backed container form of the ray result; the raw
		arrays remain the primary working representation.

		Returns
		-------
		SignalSet or None
			SignalSet of ``[rays, I, R]``; ``None`` if sea_eco is unavailable.

		Related
		-------
		seashells.make_rays_signalset : Builds the SignalSet.
		"""
		from .seashells import make_rays_signalset
		if self.rays is None:
			self.propagate_ray()
		return make_rays_signalset(self.rays, self.I, self.R, convention,
								   name=(self.name or 'section') + ' rays')

	def propagate(self, *args, kind:Literal["ray","rays","moments","envelope","covariance","wave","wave-scaled","wave_scaled","wave-hybrid","wave_hybrid"]="ray", **kwargs):
		"""Unified propagation dispatcher across the three modes.

		Routes to :meth:`propagate_ray`, :meth:`propagate_moments`, or
		:meth:`propagate_wave` according to ``kind``; all arguments are forwarded
		unchanged to the selected method.

		Parameters
		----------
		*args
			Positional arguments forwarded to the selected ``propagate_*`` method.
		kind : {'ray','rays','moments','envelope','covariance','wave'}, optional
			Propagation mode, by default ``'ray'``.
		**kwargs
			Keyword arguments forwarded to the selected ``propagate_*`` method.

		Returns
		-------
		object
			Whatever the selected ``propagate_*`` method returns.

		Raises
		------
		ValueError
			If ``kind`` is not a recognized propagation mode.
		"""
		method, forced = _propagate_method_name(kind)
		return getattr(self, method)(*args, **{**kwargs, **forced})

		#Include the initial ray. #TODO: Add conditional if source is included
		#ri = xp.append(r0[:,None,:], ri, axis=1)
		#return ri

	#@property
	#def rays(self):
	#	if self._rays is None:
	#		self.propagate_ray()
	#	return self._rays

	#@property
	#def planes(self):
	#	if self._planes is None:
	#

	def show(self,filename=None,title=None,ylims=None,zlims=None,regenerate=True):
		if self.rays is None or regenerate:
			r1 = self.propagate_ray()
		plot2D(self.rays,self.R,zpts = self.named_positions, filename=filename ,title=title, ylims=ylims,xlims=zlims)

	#def save(self,filename):
	#	with open(filename+".pkl",'wb') as f:
	#		pickle.dump(self,f)

	def copy(self):
		return deepcopy(self)
		#print(self,self.elements)
		elements = [ e.copy() for e in self.elements ]
		dic = self.__dict__ ; dic["elements"]=elements
		allowed_kwargs = inspect.signature(MicroscopeSection).parameters.keys() # infer allowed kwargs from function itself, and filter down to only those.
		dic = { k:v for k,v in dic.items() if k in allowed_kwargs } # e.g., Source doesn't accept "length" even though it technically has one
		#print("creating copy with dic",dic)
		#print("elem0 ids",id(elements[0]),id(self.elements[0]))
		return MicroscopeSection(**dic)


class Microscope(SEASerializable):
	"""Microscope class represents a whole microscope, and is comprised of multiple MicroscopeSections. propagation through a Microscope results in propagation through individual MicroscopeSections

		Parameters
		----------
		name : str, optional
			Name of the Section, by default ''
		sections : list
			ordered list of MicroscopeSection objects. each Sections "position" attribute is used to determine the position of the Section within the Microscope. sections=[Section1,Section2] will append Drifts to Sections appropriate to ensure spacing is correct.
		"""

	def __init__(self, name:str='',
				 sections:ArrayLike=None ) -> SEASerializable:
		self.name = name
		self.sections = sections
		self.rays = None ; self._planes = None
		self.I = None ; self.R = None		# per-plane, per-ray intensity and cumulative rotation (parallel to self.rays)
		self.mu = None ; self.covariance_matrix = None	# beam-envelope mode results
		self.wave = None								# wave-optics mode result (complex wavefield Signal)
		self.wave_scaled = None							# scaled-Fresnel mode result (SignalSet: U + s/R/tau)
		self.crossovers = None							# focal planes logged by the hybrid wave run: the conjugate family the wave's own frame follows (diffraction/back-focal for the usual flat seed). conjugate_planes() gives both families.
		if self.sections is not None and len(self.sections)>1: # check if consecutive sections are correct length. if not, insert drift at tail of first one
			for s,s2 in zip(self.sections[:-1],self.sections[1:]):
				dz = s2.position-(s.position+s.length)
				#if 0 < dz < 1e-7:
				#	print("FLOAT INTOLERANT GAP IGNORED")
				if dz > 1e-7: # Why a relatively loose 1e-7? float imprecision isn't this bad, but json reload is rounded
					#print("ADDING DRIFT OF LENGTH",dz,"BETWEEN",s,"AND",s2)
					dz = s2.position-(s.position+s.length)
					s.insert( len(s.elements) , Drift(position = s.length, length = dz ) )
				if s2.position==0:
					s2.position = s.position+s.length

	################
    # region: Dunder
	
	# DISCUSSION: when do we return a reference to the section or element, and when do we return a copy?
	# I think for singular elements/sections, we should return a reference:
	# 'Microscope["PL1"].strength = newval' should update
	# and we for sub-chunks of the microscope, we should return a copy:
	# 'Microscope[:"PL1"]' is required to edit the 3rd section of CLs/OLs/PLs, so the "edited" version should be a full copy
	def __getitemOLD__(self, item):
		#print("microscope __getitem__",item)
		# SINGLE ELEMENT OR SECTION
		# string passed (e.g., name of section or element), "PL1" or "PLs", convert to indices
		if isinstance(item,str):
			#print("index lookup from stringf")
			item = self.index(item)
		# single item specified: "PL1" or "PLs"
		if isinstance(item,int): # microscope["PLs"] will find index of section, and return that section
			#print("simple index, return reference to section")
			return self.sections[item]
		if isinstance(item,tuple): # microscope["PL1"] finds the element inside a section (indexOfPLss,indexOfPL1WithinPLs)
			#print("tuple, return single element")
			return self.sections[item[0]].elements[item[1]]
		# POTENTIALLY MULTIPLE ITEMS: '"OLs":', or '"sample":' or '3:'
		if isinstance(item,slice):	# convert "sample:" (which results in "item" being a slice) to an integer-indexed slice, e.g. slice(3,None,None)
			a,b,n=item.start,item.stop,item.step
			a,b,n=[ self.index(v) if isinstance(v,str) else v for v in [a,b,n] ]
			if False not in [ v is None or isinstance(v,int) for v in [a,b,n] ]: #False if tuple, i.e., these are just ints
				item = slice(a,b,n)
				ret = self.copy().sections[item] # ALWAYS MAKE A COPY
				# SINGLE SECTION, RETURN REFERENCE
				#if isinstance(ret,int):			# microscope["PLs"] will find index of section, and return that section
				#	return ret
				# GROUP OF SECTIONS, RETURN COPY
				#if isinstance(ret,list):		# microscope["OLs":] will return list of sections OLs,DQCM,PLs, etc, so form into a new Microscope
				#print("all int or none, simple slice, return new microscope with copied sections")
				return Microscope(name=self.name,sections=ret)

			if isinstance(a,float):
				positions = xp.asarray([ s.position for s in self.sections ])
				j = xp.where(positions<a)[0][-1]
				a = (j,a-positions[j]) # "slice the nth section to coordinated x.yz, relative to the beginning of that section
			#if isinstance(b,float): # TODO finish implementing
			#	trim_last = self.elements[b].length-
			#	b=int(np.ceil(b))

			# ONE OR MORE SECTIONS IS SLICED: ':"PL1"' includes preceeding sections AND a portion of the PLs section
			# microscope["sample":] should return a Microscope containing the sections/elements starting at "sample". if section "OLs" contains "sample", the returned Microscope should contain OLs, plus subsequent sections (e.g. DQCM and PLs), and the OLs section should only contain elements from "sample" and beyond
			a1,b1,n1 = [ v[0] if isinstance(v,tuple) else v for v in [a,b,n] ]
			if isinstance(b,(tuple,list)) and b[1]>0:
				b1+=1

			# TRIM LIST OF SECTIONS
			#print("needs trimmed copy")
			ret = self.copy().sections[slice(a1,b1,n1)]
			#print(ret,self.copy())
			# TODO what if we do: "PL1:PL3", these are inside the same section, we ought to check if a1==b1
			# TRIM FIRST SECTION'S ELEMENTS
			if isinstance(a,tuple) and a[1]>0:
				ret[0].elements = ret[0].elements[a[1]:]
				p0 = ret[0].elements[0].position			# now-first element's position
				for i,e in enumerate(ret[0].elements):
					ret[0].elements[i]._position -= p0		# shift all element positions so first is at zero
				ret[0].length -= p0							# update section length
				p1 = ret[0].position						# now-first section's position
				for i,s in enumerate(ret):					# shift so first section starts at 0
					ret[i].position -= p1					# shift all sections so first is at zero
					if i>0:									# subsequent sections ALSO need to be brought forwards by the the shortening of
						ret[i].position -= p0
			# TRIM LAST SECTION'S ELEMENTS
			if isinstance(b,tuple) and b[1]<len(ret[-1].elements):
				ret[-1]=ret[-1][:b[1]]						# trim last section's elements
				ret[-1].length = ret[-1][-1].position + ret[-1][-1].length # update last section's length

			return Microscope(name=self.name,sections=ret)

	# indexable names (via getitem below) consists of: section names, or element names within any section
	def keys(self):
		kys = [ s.name for s in self.sections ] +\
			sum([ [ e.name for e in s.elements ] for s in self.sections ] , [] )
		kys = [ k for k in kys if k is not None and len(k)>0 ]
		return kys

	def __getitem__(self, item):
		# RETURN A REFERENCE TO A SINGLE ITEM (SECTION OR ELEMENT), BY NAME OR INDEX
		if isinstance(item,str):	# convert "PL1" into an integer index
			item = self.index(item)							# tuple (indexOfSection,indexOfElementInThatSection)
		if isinstance(item,tuple):
			return self.sections[item[0]].elements[item[1]]
		if isinstance(item,int):
			return self.sections[item]
		# RETURN A COPY OF A SLICE (SUBSET OF ITEMS), BY NAME, INDEX, OR Z-LOCATION
		if isinstance(item,slice):
			# "sample:" --> a="sample",b=None,c=None. or "2.5:11.0" --> a=2.5,b=11.5,c=None.
			a,b,n=item.start,item.stop,item.step
			trim_first = 0 ; trim_last = 0
			positions = xp.asarray([ s.position for s in self.sections ])
			if isinstance(a,float):
				i = xp.where(positions<a)[0][-1] 	# index of last section which starts prior to z=a, i.e., spans across z=a
				trim_first = a-positions[i]			# we'll need to cut off dz from that section (s.pos=4.5, a=5, trim 0.5)
				a = i								# index slicing of self.sections will begin with the section spanning z=a
			if isinstance(b,float):
				i = xp.where(positions<b)[0][-1]
				trim_last = b-positions[i]
				b=i+1								# we want to include the section spanning z=b, hence +1
			if isinstance(a,str):
				a = self.index(a)					# may be a tuple! be careful
			if isinstance(b,str):
				b = self.index(b)
			# since self.index(namedElement) might be a tuple (sectionIndex,elementIndex), we need to filter to section indices
			a1,b1,n1 = [ v[0] if isinstance(v,tuple) else v for v in [a,b,n] ]
			item = slice(a1,b1,n1)
		# DEEP COPY MYSELF, SLICE SECTIONS, MAKE ADJUSTMENTS TO POSITIONS AND LENGTHS
		new = self.copy()
		new.sections = new.sections[item]		# list of sections, deeeeeep-copied (including copies of all sections/elements)
		# front side trimming by index: hand off to section trimming
		if isinstance(a,tuple):
			l1 = new.sections[0].length
			new.sections[0] = new.sections[0][a[1]:]
			new.sections[0].position += l1-new.sections[0].length # then "scoot it back" so it ends where it ended previously
		# front-side trimming by z_position: hand off to section trimming
		if trim_first > 0:
			new.sections[0] = new.sections[0][trim_first:]	# let MicroscopeSection.__getattr__ handle section trimming
			new.sections[0].position+=trim_first		# then "scoot it back" so it ends where it ended previously
		p0 = new.sections[0].position
		for i,s in enumerate(new.sections):
			new.sections[i].position -= p0		# shift all sections positions so first is at zero
		if trim_last > 0:
			new.sections[-1] = new.sections[-1][:trim_last]	# let MicroscopeSection.__getattr__ handle section trimming
		# scoot ALL sections forwards so section 0 starts at 0
		p0 = new.sections[0].position
		for i,s in enumerate(new.sections):
			new.sections[i].position -= p0		# shift all section positions so first is at zero
		# and finally, update new's length
		new.length = new.sections[-1].position+new.sections[-1].length
		return new

	def __repr__(self) -> str:

		strings = []
		for s in self.sections:
			header = "Section: "+s.name+" @ "+str(s.position)+" , length="+str(s.length)
			#if self.print_fancy:
			#	print(header)
			strings.append( header )
			strings.append( s.__repr__() )
		#if self.print_fancy:
		#	return ''
		return "\n".join(strings)
		
	def __str__(self):
		if self.name is None or self.name=='': name = 'Unamed'
		else: name = self.name
		return f'{name} (Section)'

	# endregion
    ################

	# TODO rather silly to need to infer planes in every script and measure. we should have a property for each to standardize it
	#@property
	def beam_current(self,regenerate=False):
		if regenerate:
			self.propagate_ray()
		return self.I[-1,0]
	#@property
	def convergence_angle(self,regenerate=False):
		if regenerate:
			self.propagate_ray()
		z = self.get_element_position("O1")+self["O1"].length+.001
		x,y,xt,yt,R,I = measureAtZ(z,section=self)
		return xt
	#@property
	def focus_error(self,expected_C3_crossover=0,regenerate=False):
		if regenerate:
			self.propagate_ray()
		planes = findPlanes(self.rays,"x") #['x']['diff' or 'image']['z' or 'M' or 'R' or 'p']
		zp = planes['x']['diff']['z']	# findPlanes returns fractional coordinated. 1.4 is 40% of the way through element 1
		zp = [ zFromFractional(self.rays[:,0,columnByName('z')],z) for z in zp ]
		zp=xp.asarray(zp)
		i = xp.where(zp > self.get_element_position("C3"))[0][0] # first plane after CL3 (not closest, as we did for mag/rot w/r/t CCD)
		return zp[i]-expected_C3_crossover

	#####################################
    # region: SEASerializable integration

	def _get_tree_html(self, recursive_level: List[str] = 0,
					   exclude_keys: List[str] = ['rays', 'labels', 'I', 'R', 'mu', 'covariance_matrix', 'wave', 'wave_scaled'],
                       exclude_hidden: bool = True,
                       exclude_properties:bool = False,
                       promote_itterable_keys: List[str] = ['sections']
                       ) -> str:
		return super()._get_tree_html(recursive_level, 
                                     exclude_keys=exclude_keys, 
                                     exclude_hidden=exclude_hidden,
                                     exclude_properties=exclude_properties,
                                     promote_itterable_keys=promote_itterable_keys
                                     )

    # endregion
    #####################################

	# given a string for an element name, return the index of that element
	def index(self,item):
		names = [ s.name for s in self.sections ]
		if item in names:
			return names.index(item)
		subnames = [ [ getattr(e,"name","") for e in s.elements ] for s in self.sections ]
		if item not in sum(subnames,[]):
			print("ERROR: name",item,"not found in Microscope or Microscope's sections' elements")
			return None
		for i,names in enumerate(subnames):
			if item in names:
				return (i,names.index(item))

	# TWP 2026-03-05 allow element insertion by coordinate ("add a lens midway through this drift section at z=etc"
	def insert(self,index,elementOrSection):
		#print("microscope insertion",index,element)
		for i,s in enumerate(self.sections):
			if s.position <= index < s.position+s.length:
				#print("INSERT ELEMENT",elementOrSection.name,"AT",index-s.position,"IN SECTION",s.name)
				if isinstance(elementOrSection,MicroscopeSection):
					elements = s.elements
					l = s.length ; l1 = index-s.position				# | lens1 driiiiiift1 lens2 driiiiiift2 lens3 |
					l2 = elementOrSection.length ; l3 = l-l1-l2			# | lens1 dr|l2l3|ft1 lens2 driiiiiift2 lens3 |
					ele1 = [ e for e in elements if e.position < l1 ]	# s1 needs elements trimmed, new total len / last drift len
					sec1 = s ; sec1.elements=ele1 ; sec1.length = l1 ; sec1[-1].length = l1-sec1[-1].position
					sec2 = elementOrSection ; sec2.position = sec1.position+l1 # s2 needs position set
					ele3 = [ e for e in elements if e.position > l1 ]	# s3 needs
					for e in ele3:
						e._position -= (l1+l2)
					#print(ele3)
					sec3 = MicroscopeSection(name="added",position=sec1.position+l1+l2,elements=ele3)
					#sec3.insert(0,Drift(length=l3-sec3.length)) ; print(ele3,sec3.length)
					self.sections[i]=sec1 ; self.sections.insert(i+1,sec2) ; self.sections.insert(i+2,sec3)
					#print("original length",l,"split into",l1,l2,l3)
					#print("new section lengths:",[ s.length for s in self.sections ])
					break
				s.insert(index-s.position,elementOrSection)
				break
		else: # TODO bug: if we insert a huge drift ("big enough to fill the space") it's just a huge drift. we should update the length based on the space it'll fit. either here, or in Section.insert.
			s = self.sections[-1]
			dz = index-(s.position+s.length)
			s.insert( len(s.elements), Drift(length=dz,position=s.length) )
			#print("APPENDING ELEMENT",element.name,"AT END OF",index-s.position,"IN SECTION",s.name)
			elementOrSection.position = index-s.position
			s.insert( len(s.elements), elementOrSection )

	def adjust_element_length(self,element,newlength): #,centering=True):
		i,j = self.index(element)							# whichSection,whichElementInSection
		L1 = self.sections[i].elements[j].length			# current length
		dL = newlength-L1									# change in length
		#if centering:
		#	self.move_element(element,dz=-dL/2)
		self.sections[i].elements[j].length = newlength		# update the element
		if len(self.sections[i])>j+1:						# if this element isn't the last in its section
			self.sections[i].elements[j+1]._position+=dL	# update subsequent element position...
			self.sections[i].elements[j+1].length-=dL		# ...and length
		else:
			print("ADJUST ELEMENT LENGTH NOT YET IMPLEMENTED FOR LAST ELEMENT IN SECTION")

	def get_element_position(self,e):
		i,j = self.index(e)
		return self.sections[i].position+self.sections[i][j].position

	def move_element(self,element,z=None,dz=None,allow_unsafe=False):
		# always move by "dz". for a Microscope object, assume z is relative to the Microscope, not the Section.
		if z is not None:
			dz = z-self.get_element_position(element)
		# find what MicroscopeSection contains the element
		i,j = self.index(element)
		L0 = self.sections[i].length
		# simply call into section's move_element code
		self.sections[i].move_element(element,dz=dz,allow_unsafe=allow_unsafe)
		# look out! MicroscopeSection.move_element with allow_unsafe=True allows for lengthening! scoot all subsequent sections out
		if self.sections[i].length > L0:
			dL = self.sections[i].length-L0
			for ii,s in enumerate(self.sections[i:]):
				if ii==0:
					continue
				s.position+=dL

	# generalized "setter" function: pass a dict of any {element:{attribute:value}} and we'll set them all. useful for the various fitting and error functions: a minimizer can supply "P1":{"calibration":trialvalue} and so on to fit for calibrations, or "P1":{"strength":trialvalue} to find the strength which yields a crossover at a given point, etc
	def update_with_settings(self,settings):
		for element in settings.keys():
			for attribute,value in settings[element].items():
				if not hasattr(self[element],attribute):
					raise AttributeError("Attribute \""+attribute+"\" not found on "+str(type(self[element]))+" Element")
				if attribute == "length":
					L = self[element].length
					self.adjust_element_length(element,value)
					self.move_element(element,dz=-value/2+L/2)
				elif attribute == "position":
					z = self[element].position
					self.move_element(element,dz=value-z)
				else:
					setattr(self[element],attribute,value)

	# commenting out. too many edge cases
	#def move_element(self,element,newposition,preserve_others=True):
	#	current = self.get_element_position(element)
	#	i,j = self.index(element)
	#	dz = newposition - current
	#	if dz>0: # increase z position: adjust subsequent drift, OR, increase length of section
	#		if len(self.sections[i].elements) == j+1: # LAST ELEMENT IN ITS SECTION
	#			self.sections[i][j].kind == "Drift":
	#				if dz > self.sections[i][j].length:
	#
	#			if preserve_others:
	#				raise NotImplementedError("Microscope.move_element does not support pushing elements into subsequent sections with preserve_others = True")
	#			self.sections[i].length+=dz
	#			for ii,s in enumerate(self.sections[i]):
	#				if ii==0:
	#					continue
	#				s.position+=dz
	#		else:									# NON-LAST
	#			if self.sections[i][j+1].kind != "Drift" and self.sections[i][j].kind == "Drift":

	#				if preserve_others:
	#					raise NotImplementedError("Microscope.move_element does not support pushing elements into subsequent sections with preserve_others = True")

	# nuclear option. destructive raw position setting, then cleanup (delete all unnamed drifts, then fill in the gaps)
	#def set_positions(self,element_position_pairs):
	#def move_element(self,element,z=None,dz=None,fail_hard_or_warn="fail"):


	@property
	def named_positions(self):
		l = {}
		for s in self.sections:
			ls = s.named_positions
			ls = { k:v+s.position for k,v in ls.items() }
			l = l | ls
		return l

	def subdivided(self, zpts):
		"""Return a copy of this column with its plain drifts split for dense sampling.

		Propagation logs one plane per element exit (plus the frame events of a
		hybrid wave run), so the z sampling of any result — and of the
		cross-section drawn by :meth:`show` — is whatever the column's element
		list defines. This helper builds a **new** ``Microscope`` whose unnamed
		drifts are cut into shorter drifts, giving finer z resolution without
		changing the optics: element order, lengths, section positions and
		therefore every entry of :attr:`named_positions` are preserved exactly
		(the cut drifts sum to the original length, and the copy's elements are
		restacked sequentially by ``MicroscopeSection``). This object is left
		untouched, including any propagation result already stored on it.

		Parameters
		----------
		zpts : float or Sequence[float]
			``float`` — maximum drift length (metres); every unnamed drift
			longer than this is split into equal chunks. ``Sequence`` —
			absolute z positions (metres) at which to cut; each unnamed drift
			is split at the positions falling strictly inside it, so those z
			values become logged planes.

		Returns
		-------
		Microscope
			The subdivided copy (no propagation results).

		Raises
		------
		ValueError
			If ``zpts`` is a non-positive spacing.

		Related
		-------
		show : Accepts ``zpts`` and plots from a temporary subdivided copy.
		crossovers : Focal planes, always logged exactly by the hybrid engine
			regardless of drift subdivision.

		Notes
		-----
		Only **unnamed** drifts are split — a named drift marks a plane a user
		asked for, so its identity (and its entry in ``named_positions``) is
		kept intact. Crossover planes never need subdivision: the hybrid
		engine splits its own propagation at the analytic focus
		``z_cross = z + |R|`` and logs that plane exactly.

		Examples
		--------
		>>> dense = scope.subdivided(5e-3)          # a plane every <= 5 mm
		>>> dense.propagate_wave(mode='hybrid')     # doctest: +SKIP
		>>> scope.subdivided([0.3, 0.45]).propagate_wave(mode='hybrid')  # doctest: +SKIP
		"""
		scope = deepcopy(self)
		if xp.ndim(zpts) == 0:
			dz = float(zpts)
			if dz <= 0:
				raise ValueError(f"subdivided(zpts={zpts}) needs a positive drift spacing in metres, "
								 "or a sequence of absolute z positions to cut at.")
			cuts = None
		else:
			dz = None
			cuts = sorted(float(z) for z in zpts)
		sections = []
		for sec in scope.sections:
			z = sec.position
			elements = []
			for ele in sec.elements:
				L = getattr(ele, "length", 0) or 0
				if isinstance(ele, Drift) and L > 0 and not ele.name:
					if dz is not None:
						n = max(1, int(xp.ceil(L / dz - 1e-9)))
						lengths = [L / n] * n
					else:
						tol = 1e-12 + 1e-9 * L
						edges = [z] + [c for c in cuts if z + tol < c < z + L - tol] + [z + L]
						lengths = [b - a for a, b in zip(edges[:-1], edges[1:])]
					elements += [Drift(length=l) for l in lengths]
				else:
					ele._position = None		# restack sequentially in the new section
					elements.append(ele)
				z += L
			sections.append(MicroscopeSection(name=sec.name, elements=elements,
											  position=sec.position))
		return Microscope(name=scope.name, sections=sections)

	def _accumulate_blocks(self, axis:Literal['x','y']='x', reference=None):
		"""Walk the column accumulating the 2x2 block, yielding per-element state.

		The single mechanism behind :meth:`conjugate_planes` and
		:meth:`beam_waists`: it visits every element in z order and reports the
		accumulated rotating-frame block **at that element's entrance**, plus a
		callable giving the element's own block at any partial depth. Because
		the accumulated block is what the scaled wave frame advances by (the
		frame *is* a reference ray, ``(h, u) = (s, s/R)``), the ray, matrix and
		wave descriptions all read the same numbers off this walk.

		Parameters
		----------
		axis : {'x', 'y'}, optional
			Transverse axis, by default ``'x'``.
		reference : str, float, or None, optional
			Where to start accumulating — the *object* plane whose conjugates
			are being sought. A name from :attr:`named_positions`, a z in
			metres, or ``None`` (default) for the column entrance. Elements
			upstream of it are skipped.

		Yields
		------
		tuple
			``(z0, ele, L, M, block)`` — the element's entrance z (metres), the
			element, its length, the accumulated ``2x2`` block at that entrance,
			and ``block(dz)`` returning the element's own block at depth ``dz``.

		Raises
		------
		KeyError
			If ``reference`` names a position this column does not have.
		ValueError
			If an element with a finite body has a non-symplectic block
			(``det != 1``) — its matrix is unphysical, so no plane downstream of
			it can be trusted.

		Related
		-------
		Element.transfer_block : Supplies each element's block.
		conjugate_planes, beam_waists : Consumers.
		"""
		z_ref = 0.0
		if reference is not None:
			z_ref = (self.named_positions[reference] if isinstance(reference, str)
					 else float(reference))
		flat = []
		for sec in self.sections:
			for ele in sec.elements:
				flat.append((sec.position + (ele.position or 0.0), ele,
							 getattr(ele, "length", 0) or 0.0))
		flat.sort(key=lambda e: e[0])
		M = xp.eye(2)
		z_prev = z_ref
		for z0, ele, L in flat:
			if z0 + L < z_ref - 1e-12:			# entirely upstream of the reference
				continue						# (zero-length elements AT z_ref count)
			start = max(z0, z_ref)
			if start - z_prev > 1e-12:			# free space since the last element
				M = xp.asarray([[1.0, start - z_prev], [0.0, 1.0]]) @ M
			z_prev = z0 + L
			if start > z0 + 1e-12:				# reference falls inside this element
				# homogeneous body: the block from depth (start-z0) to L is the
				# element's own block over the remaining length
				M = xp.asarray(ele.transfer_block(dz=z0 + L - start, axis=axis),
							   dtype=float) @ M
				continue
			blk = xp.asarray(ele.transfer_block(axis=axis), dtype=float)
			if L > 0:
				# a real element conserves phase-space area (Liouville), so its
				# block must be symplectic; without that, partial-length
				# composition inside the body is meaningless
				det = float(xp.linalg.det(blk))
				if abs(det - 1.0) > 1e-9:
					raise ValueError(
						f"{type(ele).__name__} {ele.name or ''!r} has a non-symplectic "
						f"{axis}-block over its {L} m body (det = {det:.6f}, must be 1), "
						"so planes inside or beyond it cannot be located. Its "
						"transfer_matrix needs fixing before this element can be walked "
						"(a defocusing body should be cosh/sinh, not cos/sin); until "
						"then, model it as thin (length = 0).")
			yield z0, ele, L, M, lambda dz, _e=ele: xp.asarray(
				_e.transfer_block(dz=dz, axis=axis), dtype=float)
			M = xp.asarray(ele.transfer_block(axis=axis), dtype=float) @ M

	def _element_roots(self, ele, L, block, P0, Q0):
		"""Solve ``m00(dz)*P0 + m01(dz)*Q0 = 0`` inside one element.

		The plane condition, evaluated with the element's own partial
		propagator so a plane inside a thick body is exact rather than
		interpolated between its faces.

		Parameters
		----------
		ele : Element
			The element being traversed.
		L : float
			Its length (metres).
		block : callable
			``block(dz)`` giving the element's ``2x2`` at depth ``dz``.
		P0, Q0 : float
			The accumulated pair — ``(A, C)`` for diffraction planes,
			``(B, D)`` for image planes.

		Returns
		-------
		list of float
			Roots inside the element, metres from its entrance.

		Related
		-------
		conjugate_planes : Collects these across the column.
		"""
		if L <= 0:
			return []
		K = ele._effective_strength() if isinstance(ele, Lens) else 0
		if isinstance(ele, Lens) and (K or 0) != 0 and L > 0:
			if P0 == 0 and Q0 == 0:
				return []
			theta = xp.arctan2(-K * P0, Q0)		# cos*P0 + sin*Q0/K = 0
			roots = [(theta + n * xp.pi) / K for n in range(-2, 4)]
			return sorted(min(float(d), L) for d in roots if 1e-15 < d <= L + 1e-15)
		m00, m01 = block(L)[0]					# linear in dz for these elements
		slope = Q0 if m01 != 0 else 0.0
		if slope == 0:
			return []
		dz = -P0 / slope
		return [dz] if 1e-15 < dz <= L + 1e-15 else []

	def _waist_roots(self, ele, L, block, S):
		r"""Solve ``Sigma_12(dz) = 0`` inside one element (the waist condition).

		Distinct from :meth:`_element_roots`: the covariance transports as
		:math:`\Sigma' = M \Sigma M^T`, so inside an element ``Sigma_12`` is
		*quadratic* in the element's matrix entries, not linear like a plane
		condition. In free space that still reduces to
		``Sigma_12 + dz*Sigma_22``, but through a thick lens body of strength
		``K`` (with :math:`\theta = K\,dz`)

		.. math::

			\Sigma_{12}(dz) = \cos 2\theta\, \Sigma_{12}
			+ \sin 2\theta \left[\frac{\Sigma_{22}}{2K}
			- \frac{K \Sigma_{11}}{2}\right]

		giving the closed form
		:math:`\tan 2\theta = -2K\Sigma_{12}/(\Sigma_{22} - K^2\Sigma_{11})`.

		Parameters
		----------
		ele : Element
			The element being traversed.
		L : float
			Its length (metres).
		block : callable
			``block(dz)`` giving the element's ``2x2`` at depth ``dz``.
		S : xp.ndarray
			The ``2x2`` covariance at the element's entrance.

		Returns
		-------
		list of float
			Waist positions inside the element, metres from its entrance.

		Related
		-------
		beam_waists : Collects these across the column.
		_element_roots : The (linear) plane condition.
		"""
		if L <= 0:
			return []
		K = ele._effective_strength() if isinstance(ele, Lens) else 0
		if isinstance(ele, Lens) and (K or 0) != 0:
			denom = S[1, 1] - K**2 * S[0, 0]
			if denom == 0 and S[0, 1] == 0:
				return []
			two_theta = xp.arctan2(-2 * K * S[0, 1], denom)
			roots = [(two_theta + n * xp.pi) / (2 * K) for n in range(-2, 4)]
			return sorted(min(float(d), L) for d in roots if 1e-15 < d <= L + 1e-15)
		if S[1, 1] == 0:
			return []
		dz = -S[0, 1] / S[1, 1]				# free space: Sigma_12 + dz*Sigma_22
		return [float(dz)] if 1e-15 < dz <= L + 1e-15 else []

	def conjugate_planes(self, axis:Literal['x','y']='x',
						 method:Literal['frame','ray']='ray',
						 reference=None, x0:float=1e-6, theta0:float=1e-6) -> dict:
		r"""Locate this column's image and diffraction (back-focal) planes, in metres.

		A column has **two** independent families of conjugate planes, set by
		the two independent reference rays of its transfer matrices:

		- **diffraction** (back-focal / reciprocal) planes — where rays that
		  left the reference plane *parallel to the axis* converge (``A = 0``);
		- **image** planes — where rays that left a single *on-axis point* of
		  the reference plane re-converge (``B = 0``).

		They interleave and neither is derivable from a single lens's focal
		length: each crossing reflects the whole system since the reference.
		This is the **same** calculation the scaled wave frame performs while
		propagating — the frame is a reference ray ``(h, u) = (s, s/R)``, so
		``s → 0`` on a flat seed is ``A = 0`` and on a point seed is ``B = 0``.
		Hence the wave's own crossovers (:attr:`crossovers`, the family its
		seed belongs to) appear here too, and the *other* family is available
		without a second wave run; :meth:`wavefield_at` reconstructs the field
		at any of them.

		Parameters
		----------
		axis : {'x', 'y'}, optional
			Transverse axis to analyze, by default ``'x'``. Round optics give
			the same answer on both; astigmatic optics do not.
		method : {'ray', 'frame'}, optional
			``'ray'`` (default) traces four reference rays through
			:func:`postprocessing.findPlanes` — the repo's established
			convention, interpolating between logged planes. ``'frame'``
			accumulates the transfer blocks and solves ``A = 0`` / ``B = 0`` in
			closed form, which is exact and additionally resolves planes
			*inside* a body; it is required for ``reference``. The two agree to
			~1e-12 wherever the planes fall in free space.
		reference : str, float, or None, optional
			The **object** plane whose conjugates are wanted: a name from
			:attr:`named_positions` (e.g. ``'sample'`` or a condenser
			aperture), a z in metres, or ``None`` (default) for the column
			entrance. Planes conjugate to different references are genuinely
			different sets. ``'ray'`` supports only the entrance.
		x0 : float, optional
			Height of the parallel reference ray pair for ``method='ray'``
			(metres), by default 1e-6. Paraxial, so it only sets scale.
		theta0 : float, optional
			Angle of the on-axis reference ray pair for ``method='ray'``
			(radians), by default 1e-6.

		Returns
		-------
		dict
			``{'diff': ndarray, 'image': ndarray, 'z_reference': float,
			'diff_offset': ndarray, 'image_offset': ndarray}`` — absolute plane
			positions (metres) in column order, the reference position, and the
			same planes as offsets from it (signed, downstream positive).

		Raises
		------
		ValueError
			If ``method`` is unknown, or ``reference`` is given with
			``method='ray'``.
		KeyError
			If ``reference`` names a position this column does not have.

		Related
		-------
		crossovers : The family the hybrid wave run logs while propagating.
		beam_waists : The covariance mode's minimum-width planes.
		wavefield_at : Reconstructs the wave at any of these planes.
		postprocessing.findPlanes : The ray-side cross-check.

		Examples
		--------
		>>> p = scope.conjugate_planes()                      # doctest: +SKIP
		>>> p['image'], p['diff']                             # doctest: +SKIP
		>>> scope.conjugate_planes(reference='sample')['image_offset']  # doctest: +SKIP
		"""
		if method not in ('frame', 'ray'):
			raise ValueError(f"Unknown method {method!r}; expected 'frame' (accumulated "
							 "transfer blocks, exact) or 'ray' (findPlanes cross-check).")
		z_ref = 0.0
		if reference is not None:
			if method == 'ray':
				raise ValueError("method='ray' measures conjugates of the column entrance "
								 "only; use method='frame' for a reference plane.")
			z_ref = (self.named_positions[reference] if isinstance(reference, str)
					 else float(reference))
		if method == 'ray':
			scope = deepcopy(self)		# a reference trace must not clobber self.rays
			xi = columnByName('x' if axis == 'x' else 'y')
			ti = columnByName('xt' if axis == 'x' else 'yt')
			r0 = xp.zeros((4, len(convention)))
			r0[0, xi] = x0 ; r0[1, xi] = -x0			# diffraction pair: parallel in
			r0[2, ti] = theta0 ; r0[3, ti] = -theta0	# image pair: on-axis point
			scope.propagate_ray(r0=r0)
			zs = scope.rays[:, 0, columnByName('z')]
			found = findPlanes(scope.rays, scope.R, axis=axis)[axis]
			out = {}
			for family in ('diff', 'image'):
				idx = [min(float(f), len(zs) - 1 - 1e-9) for f in found[family]['z']]
				out[family] = xp.asarray([zFromFractional(zs, f) for f in idx])
		else:
			out = {'diff': [], 'image': []}
			for z0, ele, L, M, block in self._accumulate_blocks(axis=axis,
																 reference=reference):
				A, B, C, D = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
				for family, (P0, Q0) in (('diff', (A, C)), ('image', (B, D))):
					out[family] += [z0 + dz for dz
									in self._element_roots(ele, L, block, P0, Q0)]
			out = {k: xp.asarray(sorted(v)) for k, v in out.items()}
		out['z_reference'] = z_ref
		out['diff_offset'] = out['diff'] - z_ref
		out['image_offset'] = out['image'] - z_ref
		return out

	def beam_waists(self, axis:Literal['x','y']='x', sigma0:xp.ndarray=None) -> dict:
		r"""Locate the beam-envelope waists (minimum-width planes), in metres.

		The covariance mode's counterpart of a crossover. Transporting
		:math:`\Sigma' = M\Sigma M^T`, the RMS width :math:`\sqrt{\Sigma_{11}}`
		is stationary where the position-angle correlation vanishes:

		.. math::

			\frac{d\Sigma_{11}}{dz} = 2\Sigma_{12} \;\Rightarrow\;
			\text{waist at } \Sigma_{12} = 0

		which in free space is the closed form ``dz = -Σ₁₂/Σ₂₂`` and inside a
		thick lens body follows the same ``cos/sin`` law as the plane
		conditions. Unlike a geometric crossover a waist has **finite width**,
		and it sits slightly off the geometric focus — that displacement is the
		emittance-driven focal shift, which this reports directly.

		Parameters
		----------
		axis : {'x', 'y'}, optional
			Transverse axis, by default ``'x'``.
		sigma0 : xp.ndarray, optional
			Entrance covariance for this axis, ``[[<xx>, <xx'>], [<xx'>,
			<x'x'>]]`` (m², m·rad, rad²). ``None`` (default) takes it from the
			column's ``Source`` via :meth:`propagate_moments` on a copy.

		Returns
		-------
		dict
			``{'z': ndarray, 'width': ndarray, 'emittance': float}`` — waist
			positions (metres), the RMS width ``sqrt(Σ₁₁)`` at each (metres),
			and the invariant emittance ``sqrt(det Σ)`` (m·rad).

		Raises
		------
		ValueError
			If no entrance covariance is available and the column has no
			``Source`` to derive one from.

		Related
		-------
		conjugate_planes : The system's (emittance-free) geometric planes.
		propagate_moments : The covariance transport this mirrors.

		Notes
		-----
		The emittance is a transport invariant, so it is reported once rather
		than per plane; a waist's width is then ``emittance/sqrt(Σ₂₂)`` there.

		Examples
		--------
		>>> w = scope.beam_waists()                            # doctest: +SKIP
		>>> w['z'], w['width'], w['emittance']                 # doctest: +SKIP
		"""
		from .seashells import as_ndarray
		if sigma0 is None:
			scope = deepcopy(self)			# do not clobber this object's results
			scope.propagate_moments()
			cov = as_ndarray(scope.covariance_matrix)
			i = columnByName('x' if axis == 'x' else 'y')
			j = columnByName('xt' if axis == 'x' else 'yt')
			sigma0 = xp.asarray([[cov[0][i, i], cov[0][i, j]],
								 [cov[0][j, i], cov[0][j, j]]], dtype=float)
		S0 = xp.asarray(sigma0, dtype=float)
		if not xp.all(xp.isfinite(S0)) or S0[1, 1] <= 0:
			raise ValueError("beam_waists needs a finite entrance covariance with a "
							 "positive angular variance; pass sigma0=[[<xx>, <xx'>], "
							 "[<xx'>, <x'x'>]] explicitly.")
		emittance = float(xp.sqrt(max(xp.linalg.det(S0), 0.0)))
		zs, widths = [], []
		for z0, ele, L, M, block in self._accumulate_blocks(axis=axis):
			S = M @ S0 @ M.T				# covariance at this element's entrance
			# waist where Sigma_12 = 0; the same root condition as a plane, with
			# (P0, Q0) = (Sigma_12, Sigma_22) since d(Sigma_12)/ddz = Sigma_22
			for dz in self._waist_roots(ele, L, block, S):
				Md = block(dz) @ M
				Sd = Md @ S0 @ Md.T
				zs.append(z0 + dz)
				widths.append(float(xp.sqrt(max(Sd[0, 0], 0.0))))
		order = xp.argsort(xp.asarray(zs)) if zs else xp.asarray([], dtype=int)
		return {'z': xp.asarray(zs)[order] if zs else xp.asarray([]),
				'width': xp.asarray(widths)[order] if zs else xp.asarray([]),
				'emittance': emittance}

	def propagate_ray(self, r0:xp.ndarray=None, z: float = None, verbose=False):
		"""Propagate rays through every section, carrying intensity/rotation across boundaries.

		Each section's exit intensity (``I``) and rotation (``R``) seed the next
		section, so both accumulate across the whole instrument. The flattened
		per-plane intensity and rotation are stored on ``self.I`` / ``self.R``,
		parallel to ``self.rays``.

		Parameters
		----------
		r0 : xp.ndarray, optional
			Initial geometric ray table fed to the first section. If ``None`` the
			first section generates rays from its ``Source``.
		z : float, optional
			Position within an element to propagate to, by default ``None``.
		verbose : bool, optional
			Print per-element progress, by default ``False``.

		Returns
		-------
		xp.ndarray
			Flattened geometric rays, shape ``(n_planes, n_rays, len(convention))``.
		"""
		r=r0 ; I=None ; R=None #; print("Microscope r0",r0)# starting rays/intensity/rotation fed into section.propagate
		rs=[] ; Is=[] ; Rs=[]
		for n,s in enumerate(self.sections):
			#print("section",s)
			r1 = s.propagate_ray(z=z,r0=r,I0=I,R0=R,verbose=verbose) # r1 is shape nthElement,nthRay,xythetaetc
			#print(r1.shape)
			for k in range(len(r1)):
				#r[:,columnByName('z')]#+=s.position
				rs.append(r1[k]) ; Is.append(s.I[k]) ; Rs.append(s.R[k])
			#print(r1[-1,0,:])
			r=r1[-1,:,:] ; I=s.I[-1] ; R=s.R[-1] # rays/intensity/rotation fed into subsequent section are those exiting this section
		self.rays = xp.asarray(rs) # if you want the non-flattened nthSection,nthElement,nthRay,xyzthetaetc, you should access microscope.section.rays which contain the individual nthElement,nthRay,xyzthetaetc
		self.I = xp.asarray(Is)
		self.R = xp.asarray(Rs)
		#print(self.rays.shape)
		self._planes = None
		return self.rays

	def propagate_moments(self, mu0:xp.ndarray=None, Sigma0:xp.ndarray=None, z: float = None):
		"""Propagate beam moments through every section, chaining across boundaries.

		Envelope-mode analog of :meth:`propagate_ray`. Each section's exit moments
		seed the next section (eager re-chain), so the assembled instrument is always
		consistent. Flattened per-plane means and covariances are stored on
		``self.mu`` and ``self.covariance_matrix``.

		Parameters
		----------
		mu0 : xp.ndarray, optional
			Initial mean fed to the first section; seeded from its ``Source`` when ``None``.
		Sigma0 : xp.ndarray, optional
			Initial covariance fed to the first section; seeded from its ``Source`` when ``None``.
		z : float, optional
			Unused placeholder mirroring :meth:`propagate_ray`, by default ``None``.

		Returns
		-------
		xp.ndarray
			Flattened per-plane covariance matrices, shape
			``(n_planes, len(convention), len(convention))``. Means are on ``self.mu``.
		"""
		from .seashells import make_covariance_signal, as_ndarray
		mu=mu0 ; S=Sigma0
		mus=[] ; Ss=[]
		for s in self.sections:
			s.propagate_moments(mu0=mu, Sigma0=S, z=z)
			cov = as_ndarray(s.covariance_matrix)		# raw (n_planes, 6, 6) for chaining
			for k in range(len(cov)):
				mus.append(s.mu[k]) ; Ss.append(cov[k])
			mu = s.mu[-1] ; S = cov[-1]
		self.mu = xp.asarray(mus)
		self.covariance_matrix = make_covariance_signal(xp.asarray(Ss), self.mu[:, columnByName('z')],
														convention, name=(self.name or 'microscope') + ' covariance')
		return self.covariance_matrix

	def propagate_wave(self, wave0=None, mode:Literal['fixed','scaled','hybrid']='fixed',
					   s_min:float=1e-3, absorb:float=0.1,
				   crossover:Literal['flat','jump']='flat', rotate:bool=False):
		r"""Propagate a wavefield through every section, chaining boundaries.

		The one wave-optics analog of :meth:`propagate_ray`, covering all three
		wave representations via ``mode`` (see :meth:`Element.propagate_wave`).
		Each section's exit state seeds the next (eager re-chain), and all
		per-plane states are flattened by representation: ``mode='fixed'`` into
		a single calibrated ``(n_planes, ny, nx)`` ``Signal`` on ``self.wave``;
		``'scaled'``/``'hybrid'`` into the ``SignalSet`` (U stack + companion
		``s``/``R``/``tau``/``frame`` Signals on the shared plane-z axis) on
		``self.wave_scaled``. On ``mode='hybrid'`` the crossover (focal /
		back-focal) planes logged by the frame-switching engine are included,
		and their z positions are collected on ``self.crossovers``.

		Parameters
		----------
		wave0 : Signal or seashells._Wavefield or seashells._ScaledWavefield, optional
			Initial wavefield fed to the first section (representation matching
			``mode``); generated from its ``Source`` when ``None``.
		mode : {'fixed', 'scaled', 'hybrid'}, optional
			Wave representation, by default ``'fixed'``.
		s_min : float, optional
			Backstop crossover guard for the scaled/hybrid paths, by default
			``1e-3``.

		Returns
		-------
		Signal or SignalSet or None
			The stacked result for the whole instrument (also on ``self.wave``
			/ ``self.wave_scaled``).

		Raises
		------
		ValueError
			``mode='scaled'`` only: a single frame reaching its ``s = 0``
			crossover (use ``mode='hybrid'``).

		Related
		-------
		wavefield_at : Reconstruct the physical wave at a requested plane.
		crossovers : Focal-plane positions found by the hybrid run.
		"""
		f = wave0
		planes=[]
		for s in self.sections:
			s.propagate_wave(wave0=f, mode=mode, s_min=s_min, absorb=absorb, crossover=crossover, rotate=rotate)
			sec_planes = s._wave_planes if mode == 'fixed' else s._wave_scaled_planes
			planes.extend(sec_planes)
			f = sec_planes[-1]
		if mode == 'fixed':
			self.wave = _stack_wavefields(planes, name=(self.name or 'microscope') + ' wave')
			return self.wave
		self._wave_scaled_planes = planes
		if mode == 'hybrid':
			from .seashells import read_scaled_wavefield, scaled_frame_tag as _scaled_plane_tag
			self.crossovers = [read_scaled_wavefield(p)[7] for p in planes
							   if (_scaled_plane_tag(p) or "").startswith("crossover")]
		self.wave_scaled = _stack_scaled_wavefields(planes, name=(self.name or 'microscope') + ' scaled wave')
		return self.wave_scaled

	def wavefield_at(self, z, target_dx:float=None, target_shape:tuple=None):
		r"""Reconstruct the physical wavefield ψ(x, y) at a logged plane.

		The scaled-run boundary the plan mandates: coordinate-transform the
		propagated reduced field back to physical x, y (handoff Eq 37) at the
		logged plane nearest ``z``, returning a standard calibrated wavefield
		``Signal`` that an external package (e.g. multislice) can consume.
		With a prescribed ``target_dx``/``target_shape``, U is band-limited
		resampled so the output lands exactly on the requested grid (Eq 44);
		otherwise the native grid ``Δx = |s|·Δξ`` is used (Eq 41). Runs
		:meth:`propagate_wave_scaled` first if no scaled result is stored.

		Parameters
		----------
		z : float or str
			Physical plane position (metres), or a named position (an element
			name from :attr:`named_positions`). The nearest *logged* plane is
			used — planes are logged after every finite-length element or
			aperture.
		target_dx : float, optional
			Prescribed physical pixel size (metres); ``None`` (default) keeps
			the plane's native pixel ``|s|·Δξ``.
		target_shape : tuple, optional
			Prescribed output shape ``(ny, nx)``; required with ``target_dx``.

		Returns
		-------
		Signal or seashells._Wavefield
			The reconstructed physical wavefield at the selected plane (the
			longitudinal carrier ``e^{ikz}`` is not applied).

		Raises
		------
		KeyError
			If ``z`` is a name not present in :attr:`named_positions`.

		Related
		-------
		propagate_wave : ``mode='scaled'/'hybrid'`` produces the states read here.
		waveoptics.reconstruct_physical_wave : The Eq 37/41/44 core.
		"""
		from .seashells import read_scaled_wavefield, make_wavefield_signal
		from .waveoptics import reconstruct_physical_wave
		if getattr(self, "_wave_scaled_planes", None) is None:
			self.propagate_wave(mode='hybrid')
		if isinstance(z, str):
			z = self.named_positions[z]
		zs = []
		for p in self._wave_scaled_planes:
			zi = read_scaled_wavefield(p)[7]
			zs.append(zi if zi is not None else 0.0)
		plane = self._wave_scaled_planes[int(xp.argmin(xp.abs(xp.asarray(zs) - z)))]
		U, dxi, deta, wavelength, s, R, tau, z_plane = read_scaled_wavefield(plane)
		psi, dx, dy = reconstruct_physical_wave(U, dxi, deta, wavelength, s, R,
												target_dx=target_dx, target_shape=target_shape)
		return make_wavefield_signal(psi, dx, dy, wavelength, z=z_plane,
									 name=(self.name or 'microscope') + f' wavefield at z={z_plane:g}')

	def rays_signalset(self):
		"""Return a sea_eco ``SignalSet`` view of the traced rays (rays + I + R).

		Wraps the most recent ray-mode result (``self.rays``/``self.I``/``self.R``)
		as a calibrated ``SignalSet`` via the seashells seam, propagating first if
		needed. This is the Signal-backed container form of the ray result; the raw
		arrays remain the primary working representation.

		Returns
		-------
		SignalSet or None
			SignalSet of ``[rays, I, R]``; ``None`` if sea_eco is unavailable.

		Related
		-------
		seashells.make_rays_signalset : Builds the SignalSet.
		"""
		from .seashells import make_rays_signalset
		if self.rays is None:
			self.propagate_ray()
		return make_rays_signalset(self.rays, self.I, self.R, convention,
								   name=(self.name or 'microscope') + ' rays')

	def propagate(self, *args, kind:Literal["ray","rays","moments","envelope","covariance","wave","wave-scaled","wave_scaled","wave-hybrid","wave_hybrid"]="ray", **kwargs):
		"""Unified propagation dispatcher across the three modes.

		Routes to :meth:`propagate_ray`, :meth:`propagate_moments`, or
		:meth:`propagate_wave` according to ``kind``; all arguments are forwarded
		unchanged to the selected method.

		Parameters
		----------
		*args
			Positional arguments forwarded to the selected ``propagate_*`` method.
		kind : {'ray','rays','moments','envelope','covariance','wave'}, optional
			Propagation mode, by default ``'ray'``.
		**kwargs
			Keyword arguments forwarded to the selected ``propagate_*`` method.

		Returns
		-------
		object
			Whatever the selected ``propagate_*`` method returns.

		Raises
		------
		ValueError
			If ``kind`` is not a recognized propagation mode.
		"""
		method, forced = _propagate_method_name(kind)
		return getattr(self, method)(*args, **{**kwargs, **forced})

	# property for planes ("microscope.planes" instead of "postprocessing.findPlanes(microscope)"), which avoids the need to recalculate planes a bunch of times.
	@property
	def planes(self):
		if self._planes is None:
			if self.rays is None:
				self.propagate_ray()
			self._planes = findPlanes(self.rays,self.R,"x")
		return self._planes

	# TODO self.rays should be a property, self._rays should hold the previously-calculated rays. self.rays should do a hash on the microscope (use repr?) to check if the microscope has changed. if so, re-call propagate_ray, else, return self._rays.
	#@property
	#def rays():

	@property
	def named_sections(self):
		return { s.name+" ("+str(i)+")":[s.position,s.position+s.length] for i,s in enumerate(self.sections) }

	def show(self, kind:Literal["ray","rays","moments","envelope","covariance","wave","wave-scaled","wave_scaled","wave-hybrid","wave_hybrid"]="ray",
			 filename=None, title=None, ylims=None, zlims=None, regenerate=True, plt_ax=None,
			 plane:int|float|str=None, zpts=None, conjugates:bool=True):
		"""Visualize a propagation result.

		``kind="ray"`` draws the usual ray diagram (with element/plane overlays).
		``kind="moments"`` and ``kind="wave"`` delegate to the result **Signal's own**
		``.show()``: the covariance matrix at one plane, and the wavefield intensity
		``|E|²`` at one plane, respectively (sea_eco's ``Signal.show`` renders ≤2D, so a
		single z-plane is selected via ``plane``). ``kind="wave-scaled"`` /
		``"wave-hybrid"`` show the scaled-Fresnel result: with no ``plane``, the
		|ψ(x, y=0, z)| **cross-section** — the wave analog of the ray diagram,
		with element and crossover annotations; with a ``plane`` (index into the
		logged planes, a z in metres, or a named position like ``"sample"``),
		the reconstructed physical |ψ|² at that plane via the wavefield
		Signal's own ``.show()``.

		Parameters
		----------
		kind : {'ray','rays','moments','envelope','covariance','wave','wave-scaled','wave-hybrid'}, optional
			Which propagation result to show, by default ``'ray'``
			(underscore aliases accepted).
		filename : str, optional
			If given, save the figure here instead of showing it.
		title : str, optional
			Plot title.
		ylims, zlims : sequence, optional
			Axis limits for the ray diagram (``kind='ray'`` only).
		regenerate : bool, optional
			Re-propagate before plotting, by default ``True``.
		plt_ax : matplotlib axis, optional
			Draw into an existing axis instead of creating one.
		plane : int, float, or str, optional
			Which plane to image. ``kind='wave'``/``'moments'``: an integer
			z-plane index, ``None`` (default) meaning the last plane. The
			scaled kinds: ``None`` draws the cross-section; an integer indexes
			the logged planes, a float selects the nearest plane to that z
			(metres), and a string a named position (e.g. ``"sample"``) or a
			crossover z from :attr:`crossovers`.
		zpts : float or Sequence[float], optional
			Scaled kinds only: plot from a temporary :meth:`subdivided` copy of
			this column, for denser z sampling than the stored element list
			gives (a max drift spacing in metres, or explicit absolute z
			positions to cut at). A float ``plane`` is added to the cut set so
			that plane is logged exactly instead of snapping to the nearest
			existing one. The copy is propagated on the spot and discarded —
			this object's own stored result is never touched (so ``regenerate``
			does not apply to it).
		conjugates : bool, optional
			Scaled cross-section only: also annotate the **image** planes from
			:meth:`conjugate_planes` (magenta dash-dot) alongside the wave
			run's own crossovers (cyan dotted), by default True. Costs one
			four-ray reference trace on a copy.

		Returns
		-------
		None

		Raises
		------
		ValueError
			If ``kind`` is not one of the documented values, or ``zpts`` is
			given for a kind that does not support it.

		Related
		-------
		propagate_ray, propagate_moments, propagate_wave, wavefield_at
		subdivided : Builds the denser column ``zpts`` propagates.
		_scaled_wave_cross_section : The cross-section renderer.

		Examples
		--------
		>>> scope.show(kind='wave-hybrid')                       # doctest: +SKIP
		>>> scope.show(kind='wave-hybrid', zpts=5e-3)            # doctest: +SKIP
		>>> scope.show(kind='wave-hybrid', plane='sample')       # doctest: +SKIP
		>>> scope.show(kind='wave-hybrid', plane=scope.crossovers[0])  # doctest: +SKIP
		"""
		if zpts is not None and kind not in ("wave-scaled","wave_scaled","wave-hybrid","wave_hybrid"):
			raise ValueError(f"zpts is only supported for the scaled wave kinds, not {kind!r}; "
							 "call subdivided(zpts) and propagate that copy for other kinds.")
		# --- ray diagram (unchanged behavior) ---
		if kind in ("ray","rays"):
			if self.rays is None or regenerate:
				self.propagate_ray()
			sections = self.named_sections
			if zlims is None:
				zs = self.rays[:,0,columnByName("z")]
				zlims = [ xp.amin(zs),xp.amax(zs) ]
			plot2D(self.rays, self.R, zpts=self.named_positions, sections=sections, filename=filename, title=title, ylims=ylims, xlims=zlims,plt_ax=plt_ax)
			return
		# --- delegate to the result Signal's own .show() (sea_eco renders <=2D) ---
		import matplotlib.pyplot as plt
		from .seashells import read_wavefield, make_wavefield_signal
		ax = plt_ax if plt_ax is not None else plt.subplots()[1]
		if kind in ("moments","envelope","covariance"):
			idx = -1 if plane is None else plane
			if self.covariance_matrix is None or regenerate:
				self.propagate_moments()
			self.covariance_matrix[idx].show(ax=ax)			# 6x6 covariance at one plane
			if title:
				ax.set_title(title)
		elif kind == "wave":
			idx = -1 if plane is None else plane
			if self.wave is None or regenerate:
				self.propagate_wave()
			data, dx, dy, wavelength, zvals = read_wavefield(self.wave)
			zval = float(zvals[idx]) if hasattr(zvals, "__len__") else 0.0
			# wavefield is complex; show |E|^2 as a calibrated 2D wavefield Signal
			make_wavefield_signal(xp.abs(data[idx])**2, dx, dy, wavelength, z=zval,
								  name="wavefield |E|^2").show(ax=ax)
			if title:
				ax.set_title(title)
		elif kind in ("wave-scaled","wave_scaled","wave-hybrid","wave_hybrid"):
			mode = 'hybrid' if 'hybrid' in kind else 'scaled'
			scope = self
			if zpts is not None:
				# denser sampling: propagate a temporary copy, leave self alone
				cuts = zpts
				if xp.ndim(zpts) > 0 and isinstance(plane, float):
					cuts = list(zpts) + [plane]			# log the requested plane exactly
				scope = self.subdivided(cuts)
				scope.propagate_wave(mode=mode)
			elif getattr(self, "_wave_scaled_planes", None) is None or regenerate:
				self.propagate_wave(mode=mode)
			if plane is None:
				# the wave analog of the ray diagram, with the same overlays
				images = None
				if conjugates:
					images = scope.conjugate_planes(axis='x')['image']
				_scaled_wave_cross_section(
					scope._wave_scaled_planes, ax,
					named_positions=scope.named_positions,
					crossovers=getattr(scope, "crossovers", None),
					image_planes=images,
					title=title or (self.name or 'microscope') + f" {mode} wave |ψ(x, 0, z)|")
			else:
				if isinstance(plane, (int, xp.integer)) and not isinstance(plane, bool):
					from .seashells import read_scaled_wavefield
					from .waveoptics import reconstruct_physical_wave
					U, dxi, deta, wavelength, s, R, tau, zval = \
						read_scaled_wavefield(scope._wave_scaled_planes[plane])
					psi, dx, dy = reconstruct_physical_wave(U, dxi, deta, wavelength, s, R)
				else:
					# named position or z in metres -> nearest logged plane
					psi, dx, dy, wavelength, zval = read_wavefield(scope.wavefield_at(plane))
				make_wavefield_signal(xp.abs(psi)**2, dx, dy, wavelength, z=zval,
									  name="wavefield |ψ|^2").show(ax=ax)
				if title:
					ax.set_title(title)
		else:
			raise ValueError(f"Unknown show kind {kind!r}; expected 'ray', 'moments', 'wave', "
							 "'wave-scaled', or 'wave-hybrid'.")
		if filename is not None:
			plt.gcf().savefig(filename)
		elif plt_ax is None:
			plt.show()

	# Basically just json dumps all attributes, with some special considerations to make the json more human-readable: "Microscope name","Section name","Element name" instead of just "name" for each, specified ordering of attributes (name always first), and nesting lists to go down from Microscope -> Section -> Element
	def save(self,filename):
		jdict = {"Microscope name":self.name,"Sections":[]} | self.__dict__
		for k in ["sections","rays","I","R","mu","covariance_matrix","wave"]:
			jdict.pop(k,None)
		for s in self.sections:
			s_attrs = {"Section name":s.name,"position":s.position,"length":s.length,"Elements":[]} | s.__dict__
			for k in ["elements","rays","I","R","mu","covariance_matrix","wave","name"]:
				s_attrs.pop(k,None)
			for e in s.elements:
				e_attrs = {"Element name":e.name,"kind":e.kind,"position":e.position} | e.__dict__
				for k in ["name","rotation","_position"]: # "name" is to be saved as "Element name", and a lens's rotation is a locally-calculated value, NOT a meaningful attribute
					if k in e_attrs.keys():
						del e_attrs[k]
				s_attrs["Elements"].append(e_attrs)
			jdict["Sections"].append(s_attrs)
		import json
		#print(jdict)
		jdict = roundjson(jdict)
		with open(filename+'.json', 'w') as f:
			json.dump(jdict, f,indent=4)

	def copy(self):
		return deepcopy(self)
		sections = [ s.copy() for s in self.sections ]
		dic = self.__dict__ ; dic["sections"]=sections
		allowed_kwargs = inspect.signature(Microscope).parameters.keys() # infer allowed kwargs from function itself, and filter down to only those.
		dic = { k:v for k,v in dic.items() if k in allowed_kwargs } # e.g., Source doesn't accept "length" even though it technically has one
		#print("creating new Microscope with dic",dic)
		print("section0 ids",id(sections[0]),id(self.sections[0]))
		return Microscope(**dic)

	# json file containing selected lens strengths. currently not limited to where they are in the column, so be careful of mixed condenser/projector settings
	def save_as_setting(self,setting_name,keys):
		self.save_subset(setting_name,keys,kind="setting")

	# json file containing selected lens calibrations, but also conceivably things like element positions and lengths.
	def save_as_calibration(self,calibration_name,keys):
		self.save_subset(calibration_name,keys,kind="calibration")

	# save a subset of element attributes, pass a dicts of: { elementName:[attribute1,attribute2] or elementName:attribute }
	def save_subset(self,name,keys,kind):
		import json
		# always record kind ("setting" or "calibration"), name, version, timestamp
		jdict = {"kind":kind, "name":name, "version":0.001,
					"timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") }
		# loop through elements, save off each attribute
		for lens,k in keys.items():
			jdict[lens]={}
			if isinstance(k,str):
				k=[k]
			for kk in k:
				jdict[lens][kk] = getattr(self[lens],kk,None)
		# get ready to save off. if the file already exists, copy off old file and increment version number
		fout = kind+"s/"+name+'.json'
		if os.path.exists(fout):
			j_previous = json.loads("".join(open(fout).readlines()))
			move_to = fout.replace(".json","-v"+str(j_previous["version"]))+".json" # versioned filename
			move_to = move_to.replace(kind+"s/",kind+"s/old/")		# version-named files go "old" subfolder
			os.makedirs(kind+"s/old",exist_ok=True)
			shutil.move(fout,move_to)
			jdict["version"] = j_previous["version"]+0.001
		# save it off
		jdict = roundjson(jdict)
		os.makedirs(kind+"s",exist_ok=True)
		with open(fout, 'w') as f:
			json.dump(jdict, f,indent=4)

	def load_setting(self,setting_name):
		self.load_subset(setting_name,kind="setting")

	def load_calibration(self,calibration_name):
		self.load_subset(calibration_name,kind="calibration")

	def load_subset(self,name,kind):
		import json
		jdict = json.loads("".join(open(kind+"s/"+name+".json").readlines()))
		settings = { k:d for k,d in jdict.items() if k not in ["name","kind","version","timestamp"] }
		self.update_with_settings(settings)

def roundjson(jdict): # given a nested series of dicts/lists containing strings/floats/ints, iterate through all iterables, find floats, and round them
	if isinstance(jdict,dict):
		for k,v in jdict.items(): # kinda dumb for a position of 0.1 to be floated to 0.09999999999999964
			#print("checking",k,v)
			if isinstance(v,float):
				#print("rounding")
				jdict[k] = xp.round(v,8)
			elif isinstance(v,(dict,list)):
				#print("recursing")
				jdict[k] = roundjson(v)
		#else:
		#	#print("ignore")
	elif isinstance(jdict,list):
		jdict = [ roundjson(v) for v in jdict ]
	else:
		return jdict
	return jdict

def load_section(filename):
	if ".sea" in filename:
		loaded = MicroscopeSection()
		loaded.from_sea(filename)
		return loaded

	with open(filename+".pkl",'rb') as f:
		obj = pickle.load(f)
	return obj 

def load_microscope(filename):
	# RELOAD USING SEA INFRASTRUCTURE
	if ".sea" in filename:# and sea_available:
		loaded = Microscope() # dummy object, of correct type (or SEASerializable.to_sea will flag it)
		loaded.from_sea(filename) # load file into dummy object
		return loaded
		#loaded.sections = [ Section_SEAS(s) for s in loaded.sections ] # not sure why we need this, but seems like we only get the SEASerializable half of the inheritance on reload
		#sections = []
		#for s in loaded.sections:
		#	elements = [ e.copy() for e in s.elements ]
		#	s

		#= loaded.sections
		#for
		#return cloneAsObj(loaded,Microscope,childRecursion={"sections":(MicroscopeSection,{"elements":[Element]})})
		#return loaded.copy() # passing BACK into Microscope_SEAS means all sections and elements cascade back through the looping to ensure they are re-initialized as Section_SEAS and Element_SEAS which have double-inheritance from SEASerializable and MicroscopeSection or Element
		jdict = loaded.__dict__ # so far I can't figure out correct inheritance (so this Microscope_SEAS functions like a Microscope with all the appropriate functions, so instead we'll just assemble the jdict used below, which correctly casts things into the appropriate object types (Microscope > MicroscopeSection > Element)
		jdict["Sections"] = []
		for s in loaded.sections:
			jdict["Sections"].append(s.__dict__)
			jdict["Sections"][-1]["Elements"] = []
			for e in s.elements:
				jdict["Sections"][-1]["Elements"].append(e.__dict__)
	else:
		import json
		jdict = json.loads("".join(open(filename+".json").readlines()))

	import inspect
	mapping = { "Drift":Drift, "QLens":Lens, "Thin lens":Lens, "Source":Source, "Dipole":Dipole, "Thin dipole":Dipole, "Quad":Quadrapole, "Thin quad":Quadrapole, "Aperture":Aperture } # TODO Eventually need to support all Element types from elements.py. and is there a way to map these automatically instead of explicitly?

	sections = []
	for section in jdict["Sections"]: # list of dicts, "section" is a dict
		elements = []
		for element in section["Elements"]: # list of dicts, "element" is a dict
			kind = element["kind"]
			func = mapping[kind]
			element["name"] = element.get("Element name",element.get("name")) # undo the custom mapping we did inside MicroscopeSection.save
			if isinstance(element["name"],str) and "None" in element["name"]: element["name"]=''
			element.pop("Element name",None) # delete "Element name" entry, if it exists (pop avoids KeyError with del)
			allowed_kwargs = inspect.signature(func).parameters.keys() # infer allowed kwargs from function itself, and filter down to only those.
			element = { k:v for k,v in element.items() if k in allowed_kwargs } # e.g., Source doesn't accept "length" even though it technically has one
			element = func(**element) # convert dict to Element object of correct type (see elements.py)
			elements.append(element)
		section["name"] = section.get("Section name",section.get("name")) # custom mappings at section level too
		if isinstance(section["name"],str) and "None" in section["name"]: section["name"]=''
		section.pop("Section name",None) ; section.pop("Elements",None) ; section.pop("elements",None)
		allowed_kwargs = inspect.signature(MicroscopeSection).parameters.keys()
		section = { k:v for k,v in section.items() if k in allowed_kwargs } # e.g., MicroscopeSection doesn't accept "length", it builds it itself
		section = MicroscopeSection(elements = elements, **section)
		sections.append(section)
	jdict["name"] = jdict.get("Microscope name",jdict["name"])
	if isinstance(jdict["name"],str) and "None" in jdict["name"]: jdict["name"]=''
	jdict.pop("Microscope name",None) ; jdict.pop("Sections",None) ; jdict.pop("sections",None)
	allowed_kwargs = inspect.signature(Microscope).parameters.keys()
	jdict = { k:v for k,v in jdict.items() if k in allowed_kwargs }
	return Microscope(sections = sections, **jdict)

# sanity check: nextposition-thisposition should equal thislength. this function is used by tests and full builder to verify successful moves
def check_lengths(section):
	if isinstance(section,MicroscopeSection):
		zs = xp.asarray( [ e.position for e in section.elements ] )
		ls = xp.asarray( [ getattr(e,"length",0) for e in section.elements ] )
	else:
		for s in section.sections:
			check_lengths(s)
		zs = xp.asarray( [ e.position for e in section.sections ] )
		ls = xp.asarray( [ e.length for e in section.sections ] )
	#print("check_lengths",section)
	dz = zs[1:]-zs[:-1] ; print(zs,ls)
	assert xp.sum( xp.absolute( dz-ls[:-1] ) ) < .00001

# look for gaps and overlaps, adjust positions and lengths of Drifts only, and combine unnamed Drifts. (we're using this instead of fixing gaps/overlaps while building inside of MicroscopeSection > __init__)
def repair(section):
	# check all elements and their preceeding neighbor
	for i,e in enumerate(section.elements):
		if i==0:
			continue
		em = section.elements[i-1] # element_minus. previous element
		dz = e.position - ( em.position + getattr(em,"length",0) )
		# GAPS, positive dz
		if 0 < dz and e.kind == "Drift" and e.name == "": # SLIDE AND LENGTHEN THIS UNNAMED DRIFT
			e._position -= dz ; e.length += dz
		elif 0 < dz and em.kind == "Drift": # NAMED DRIFT, OR OTHER ELEMENT TYPE. EXPAND PRECEEDING DRIFT
			em.length += dz
		elif 1e-7 < dz: # THIS AND PREVIOUS ARE BOTH IMMOVABLE/UNLENGTHENABLE, AND OUT OF TOLERANCE.
			section.elements.insert(i, Drift(length=dz,position=em.position+getattr(em,"length",0)) )
		# OVERLAPS, negative dz
		if dz < 0 and e.kind == "Drift" and e.name == "": # SLIDE AND SHORTEN THIS UNNAMED DRIFT
			e._position -= dz ; e.length += dz
		elif dz < 0 and em.kind == "Drift": # NAMED DRIFT, OR OTHER ELEMENT TYPE. SHORTEN PRECEEDING DRIFT
			em.length += dz
		elif dz < -1e-7: # THIS AND PREVIOUS ARE BOTH IMMOVABLE/UNLENGTHENABLE, AND OUT OF TOLERANCE. NO SOLUTION, RAISE ERROR
			print("WARNING",section,"HAS ELEMENTS",em,"AND",e,"WHICH OVERLAP AT INDEX",i)
		#: # GAP, BUT NAMED DRIFT OR OTHER ELEMENT TYPE. INSERT DRIFT
	# special case: section.length vs last Drift's position and length:
	el = section.elements[-1]
	dz = section.length - ( el.position + getattr(el,"length",0) )
	# GAP, LAST SECTION IS DRIFT, LENGTHEN
	if 0 < dz and el.kind == "Drift":
		el.length += dz
	# OVERLAP: LENGTHEN SECTION
	if dz < 0:
		section.length -= dz
	# crawl the list backwards, look for pairs of Drifts (second one must be unnamed)
	for i in range(len(section.elements)-1,0,-1):
		e = section.elements[i]
		em = section.elements[i-1] # element_minus. previous element
		if em.kind == "Drift" and e.kind == "Drift" and e.name == "":
			em.length = getattr(em,"length",0) + getattr(e,"length",0)		# combine lengths into first Drift
			del section.elements[i]											# and delete the second

