import numpy as xp
from numpy.typing import ArrayLike
from typing import List

flag_gpu = False
import pickle
import sys,inspect

from .postprocessing import plot2D,findPlanes,zFromFractional,measureAtZ
from .elements import Element,Source,Drift,Lens,Dipole,Quadrapole,columnByName,Aperture
from .seashells import SEASerializable

from copy import deepcopy

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

		
		if self.elements is None or (self.elements)==0:
			return

		new = []
		for n,ele in enumerate(elements):
			#print("process element",n,"=",repr(ele))
			#print("self.length",self.length,"adding ele",ele.kind,ele.name,ele.position,ele.length)
			if ele.position is None:						# e.g. pass Lens(l1),Drift(l2),Lens(l3) --> Drift.position=l1, Drift.position=l1+l2
				#print("(no position, add to end)")
				ele._position = self.length
			# SANITY CHECK: if there's a "gap" between this element's position and end of previous, then add a drift
			if self.length < ele.position:
				#print("(gap before this element)")
				dz = ele.position - self.length #; print("dz",dz,"position",ele.position-dz)
				if dz>1e-10:
					#print("add drift",n,ele.position,self.length,dz)
					new.append( Drift(length=dz,position=ele.position-dz) )
				self.length += dz
			new.append(ele)
			if self.length > ele.position:
				print('WARNING: previous Element ('+str(elements[n-1])+') overlaps with specified Element position '+str(ele))
			if ignoreLensThickness and ele.kind in ['Thin lens','QLens','Thin quad','Quad']:
				continue
			#print("increment length by",getattr(ele,"length",0))
			self.length += getattr(ele,"length",0)
			#print("new length = ",self.length)
		self.elements = new

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
					   exclude_keys: List[str] = ['rays'], 
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
				if ele.position<=index and ele.position+ele.length>index and ele.kind=="Drift":
					#print("INSERTING ELEMENT",element.name,"AT",index,"(",ele,ele.position,ele.length,")","AT POSITION",i)
					elementlength=0 if self.ignoreLensThickness else getattr(element,"length",0)
					l1=index-ele.position ; l2=ele.length-elementlength-l1 # "this drift needs to be length 4.0, and we'll need another drift after the insertion"
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

	def move(self,elementName,z=None,dz=None): # TODO massive assumption here is that we're adjusting non-first non-last element positions!
		i=self.index(elementName)
		if z is not None:
			dz = z-self.elements[i].position
		self.elements[i]._position+=dz			# element position is updated
		self.elements[i-1].length+=dz			# previous element is lengthened
		if self.elements[i-1].length < 0 and i>2 and self.elements[i-2].kind == "Drift": # edge case: if two drifts in a row, and one is shortened to below zero length, simply combine them
			self.elements[i-2].length += self.elements[i-1].length
			del self.elements[i-1] ; i-=1
		if i+1<len(self.elements):
			self.elements[i+1]._position+=dz	# subsequent element is also moved
			self.elements[i+1].length-=dz		# subsequent element is shortened
		else: # IF THIS IS THE LAST ELEMENT:
			if dz<0: # append Drift element if elementName is moved forwards...
				self.elements.append(Drift(length=-dz,position=self.elements[i]._position+self.elements[i].length))
			else:	# or lengthen section (dangerous!) if elementName is moved backwards...
				self.length += dz
		print(repr(self))

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
		return { e.name:e.position for e in self.elements if e.name is not None }

	# returns nthElement,nthRay,xythetaetc
	def propagate_ray(self, r0:xp.ndarray=None,
					   z: float = None, 
					   verbose=False):
		#print("Section r0",r0)
		if r0 is None:
			if isinstance(self.elements[0], Source):
				r0 = self.elements[0].rays()
			else:
				raise UserWarning("First element is not a Source, and no r0 provided to propagate_ray. Please provide initial rays or ensure first element is a Source.")
		ri=[r0]
		for i,ele in enumerate(self.elements):
			if verbose:
				print("propate:",ele.name,"@",ele.position,"x,y",xp.amax(ri[-1][:,columnByName("x")]),xp.amax(ri[-1][:,columnByName("y")])) #,"xt,yt",xp.amax(ri[-1][:,columnByName("xt")]),xp.amax(ri[-1][:,columnByName("yt")]))
			ele_ri = ele.propagate_ray(ri[-1], z=z)
			#ele_ri[...,-2] += ele.position # TWP 2025/08/27 - do not add distance. drift already should update z
			#print(ele_ri.shape,r0.shape)
			if getattr(ele,"length",0) != 0 or ele.kind == "Aperture":
				ri.append(ele_ri[:,:])
			else:
				ri[-1]=ele_ri[:,:]
		self.rays = xp.asarray(ri) # xp.swapaxes(xp.asarray(ri),0,1)
		return self.rays

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
		plot2D(self.rays,zpts = self.named_positions, filename=filename ,title=title, ylims=ylims,xlims=zlims)

	def save(self,filename):
		with open(filename+".pkl",'wb') as f:
			pickle.dump(self,f)

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
		if self.sections is not None and len(self.sections)>1: # check if consecutive sections are correct length. if not, insert drift at tail of first one
			for s,s2 in zip(self.sections[:-1],self.sections[1:]):
				if s.position+s.length<s2.position:
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
		return self.rays[-1,0,columnByName('I')]
	#@property
	def convergence_angle(self,regenerate=False):
		if regenerate:
			self.propagate_ray()
		z = self.get_element_position("OL1")+self["OL1"].length+.001
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
		i = xp.where(zp > self.get_element_position("CL3"))[0][0] # first plane after CL3 (not closest, as we did for mag/rot w/r/t CCD)
		return zp[i]-expected_C3_crossover

	#####################################
    # region: SEASerializable integration

	def _get_tree_html(self, recursive_level: List[str] = 0, 
					   exclude_keys: List[str] = ['rays', 'labels'], 
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

	def adjust_element_length(self,element,newlength):
		i,j = self.index(element)							# whichSection,whichElementInSection
		l1 = self.sections[i].elements[j].length			# current length
		dl = newlength-l1									# change in length
		self.sections[i].elements[j].length = newlength		# update the element
		if len(self.sections[i])>j+1:						# if this element isn't the last in its section
			self.sections[i].elements[j+1]._position+=dl		# update subsequent element position...
			self.sections[i].elements[j+1].length-=dl			# ...and length
		else:
			print("ADJUST ELEMENT LENGTH NOT YET IMPLEMENTED FOR LAST ELEMENT IN SECTION")

	def get_element_position(self,e):
		i,j = self.index(e)
		return self.sections[i].position+self.sections[i][j].position

	@property
	def named_positions(self):
		l = {}
		for s in self.sections:
			ls = s.named_positions
			ls = { k:v+s.position for k,v in ls.items() }
			l = l | ls
		return l

	def propagate_ray(self, r0:xp.ndarray=None, z: float = None, verbose=False):
		r=r0 #; print("Microscope r0",r0)# starting rays (optional) to be fed into section.propagate
		rs=[]
		for n,s in enumerate(self.sections):
			#print("section",s)
			r1 = s.propagate_ray(z=z,r0=r,verbose=verbose) # r1 is shape nthElement,nthRay,xythetaetc
			#print(r1.shape)
			for r in r1:
				#r[:,columnByName('z')]#+=s.position
				rs.append(r)
			#print(r1[-1,0,:])
			r=r1[-1,:,:] # rays fed into subsequent section are the rays exiting this section
		self.rays = xp.asarray(rs) # if you want the non-flattened nthSection,nthElement,nthRay,xyzthetaetc, you should access microscope.section.rays which contain the individual nthElement,nthRay,xyzthetaetc
		#print(self.rays.shape)
		self._planes = None
		return self.rays

	# property for planes ("microscope.planes" instead of "postprocessing.findPlanes(microscope)"), which avoids the need to recalculate planes a bunch of times.
	@property
	def planes(self):
		if self._planes is None:
			if self.rays is None:
				self.propagate_ray()
			self._planes = findPlanes(self.rays,"x")
		return self._planes

	@property
	def named_sections(self):
		return { s.name+" ("+str(i)+")":[s.position,s.position+s.length] for i,s in enumerate(self.sections) }

	def show(self,filename=None,title=None,ylims=None,zlims=None,regenerate=True,plt_ax=None):
		if self.rays is None or regenerate:
			r1 = self.propagate_ray()

		sections = self.named_sections
		#print("SECTIONS",sections)
		if zlims is None:
			zs = self.rays[:,0,columnByName("z")]
			zlims = [ xp.amin(zs),xp.amax(zs) ]
		plot2D(self.rays, zpts=self.named_positions, sections=sections, filename=filename, title=title, ylims=ylims, xlims=zlims,plt_ax=plt_ax)

	# Basically just json dumps all attributes, with some special considerations to make the json more human-readable: "Microscope name","Section name","Element name" instead of just "name" for each, specified ordering of attributes (name always first), and nesting lists to go down from Microscope -> Section -> Element
	def save(self,filename):
		jdict = {"Microscope name":self.name,"Sections":[]} | self.__dict__
		del jdict["sections"],jdict["rays"]
		for s in self.sections:
			s_attrs = {"Section name":s.name,"position":s.position,"length":s.length,"Elements":[]} | s.__dict__
			del s_attrs["elements"],s_attrs["rays"],s_attrs["name"]
			for e in s.elements:
				e_attrs = {"Element name":e.name,"kind":e.kind,"position":e.position} | e.__dict__
				for k in ["name","rotation","_position"]: # "name" is to be saved as "Element name", and a lens's rotation is a locally-calculated value, NOT a meaningful attribute
					if k in e_attrs.keys():
						del e_attrs[k]
				s_attrs["Elements"].append(e_attrs)
			jdict["Sections"].append(s_attrs)
		import json
		#print(jdict)
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

