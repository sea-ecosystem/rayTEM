# try:
#	 import cupy as xp
#	 flag_gpu = True
#	 from cupy.typing import xp.ndarray
# except:
#	 import numpy as xp
#	 flag_gpu = False
#	 from numpy.typing import xp.ndarray
import numpy as xp
from numpy.typing import ArrayLike

flag_gpu = False
#from numpy.typing import xp.ndarray
import pickle
import sys

from pandas import DataFrame
from .postprocessing import plot2D
from IPython.display import display # # TWP 2025/08/27 - adding import, required if not running inside IPython (e.g. outside of jupyter)
from .elements import Element,Source,Drift,Lens,columnByName
from .seashells import SEASerializable

# # TWP 2025/08/27 - varying indices for matrices is hectic. 
# let's settle on a convention for things like: [whichZ,whichRay,[x,xθ,y,yθ,ϕ,E]]
# here I have modified the "z" convention to "ϕ", since this is primary interest 
# (what is the phase of the electron as a function of its distance travelled)
# as opposed to "z" which is our distance down the column 
# AND, I have swapped whichZ and whichRay. Why? feels weird to pass npts x 6 and get the
# new axis *inserted* as npts x nzs x 6. plus, whichZ out front makes looping easier

# TWP 2026/02/27 - should elements positions in sections (in microscopes) be referenced to the microscope, or to the section? if i have sections CL 0-100, OL 100-150, and PL 150-250, should the positions of PL1 be "160" because that's where it is in the column, OR, should it be "10" because that's where it is within the section, and section PL has position "150"?
# execute decision: i'm going to say relative to the section. if i want to set up a PL section, and plop it on a "shorter" microscope (e.g PRIVATE_INSTRUMENT vs private_instrument?) then I want everything to stay the same for the PL section. then to calculate the position relative to the microscope, i simply add section.position+element.position

class MicroscopeSection(SEASerializable):
	""" 
	TODO: Document

	To Do
	-----
	TODO: Remove pring_fancy.
		Revert back to __repr__ returning a str and add a print_fancy function.
	"""
	def __init__(self, name:str='',
				 elements:ArrayLike=None, # list of Elements, or list of dicts
				 position:float=0.,
				 ndim:int=2,
				 print_fancy:bool=True, ignoreLensThickness=False ) -> object:
		self.name = name
		#if isinstance(elements[0],dict):
		#	self.elements = []
		#else:
		self.elements = elements
		self.position = position
		self.ndim = ndim
		self.print_fancy = print_fancy
		self.ignoreLensThickness = ignoreLensThickness
		self.rays = None
		self.length = 0 #= self.position #xp.sum([e.length for e in self.elements])
		
		if self.elements is None or (self.elements)==0:
			return

		new = []
		for n,ele in enumerate(elements):
			#print("self.length",self.length,"adding ele",ele.kind,ele.name,ele.position,ele.length)
			if ele.position is None:						# e.g. pass Lens(l1),Drift(l2),Lens(l3) --> Drift.position=l1, Drift.position=l1+l2
				#print("(no position, add to end)")
				ele.position = self.length
			# SANITY CHECK: if there's a "gap" between this element's position and end of previous, then add a drift
			if self.length < ele.position:
				#print("(gap before this element)")
				dz = ele.position - self.length #; print("dz",dz,"position",ele.position-dz)
				new.append( Drift(length=dz,position=ele.position-dz) )
				self.length += dz
			new.append(ele)
			if self.length > ele.position:
				print('WARNING: previous Element ('+str(elements[n-1])+') overlaps with specified Element position '+str(ele))
			if ignoreLensThickness and ele.kind in ['Thin lens','QLens','Thin quad','Quad']:
				continue
			self.length += ele.length
		self.elements = new

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
				if ele.position<=index and ele.position+ele.length>=index and ele.kind=="Drift":
					#print("INSERTING ELEMENT",element.name,"AT",index,"(",ele.position,ele.length,")","AT POSITION",i)
					elementlength=0 if self.ignoreLensThickness else element.length
					l1=index-ele.position ; l2=ele.length-elementlength-l1 # "this drift needs to be length 4.0, and we'll need another drift after the insertion"
					#print("PRE DRIFT",l1,"+ ELEMENT",element.length,"+ POST DRIFT",l2,"=",ele.length)
					self.elements[i].length=l1			# "shorten" initial drift
					element.position = index			# update new element's position
					self.elements.insert(i+1,element)	# add new element
					if l2>0:							# add following drift
						self.elements.insert(i+2,Drift(length=l2,position=index+elementlength))
					if l1==0:							# possible drift1 is length zero, so delete it
						del self.elements[i]
					break
			else:
				print("WARNING: unable to insert "+str(element)+" at "+str(index)+" (coordinate may be out of bounds, or non-drift element)")

	def __delitem__(self,item):
		if isinstance(item,str):
			item = self.index(item)
		if self.elements[item-1].kind != "Drift":
			print("WARNING: unable to delete "+str(element)+" at "+str(index)+" (preceeding element must be a Drift???)")
		self.elements[item-1].length+=self.elements[item].length
		del self.elements[item]

	# TWP 2025-11-05: allow indexing of the assembly by name: section["PL1"] should return the section by that name! see removed_private_instrument_tree/PRIVATE_INSTRUMENT/fine_PLs.py. 2026-02-05: also allow slicing by name: section["sample":] should return a new section with all elements including and after "sample"
	def __getitem__(self, item):
		#return item
		if isinstance(item,str):	# convert "PL1" into an integer index
			item = self.index(item)
		if isinstance(item,slice):	# convert "sample:" (which results in "item" being a slice) to an integer-indexed slice, e.g. slice(3,None,None)
			a,b,n=item.start,item.stop,item.step
			a,b,n=[ self.index(v) if isinstance(v,str) else v for v in [a,b,n] ]
			item = slice(a,b,n)
		ret = self.elements[item]
		if isinstance(ret,list):
			return MicroscopeSection(name=self.name,elements=ret,position=self.position,ndim=self.ndim,print_fancy=self.print_fancy)
		return ret

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
			reps = [[e.name, e.kind, e.position, e.length, e.strength, e.calibration] for e in self.elements]
			
			if  self.print_fancy:
				#print('Section: '+self.name+" @ "+str(self.position)+", length="+str(self.length))
				display(DataFrame(reps, columns=columns))
				return ''
			else:
				return '\n'.join(['\t'.join([f"{key}: {value}, " for key,value in zip(columns,e)])for e in reps])

	def __len__(self):
		return len(self.elements)

	# returns nthElement,nthRay,xythetaetc
	def propagate_ray(self, r0:xp.ndarray=None,
					   z: float = None, 
					   verbose=False):
		#print("Section r0",r0)
		if r0 is None:
			r0 = self.elements[0].rays()
		ri=[r0]
		for i,ele in enumerate(self.elements):
			if verbose:
				print(ele.name,"@",ele.position,"x,y",xp.amax(ri[-1][:,columnByName("x")]),xp.amax(ri[-1][:,columnByName("y")])) #,"xt,yt",xp.amax(ri[-1][:,columnByName("xt")]),xp.amax(ri[-1][:,columnByName("yt")]))
			ele_ri = ele.propagate_ray(ri[-1], z=z)
			#ele_ri[...,-2] += ele.position # TWP 2025/08/27 - do not add distance. drift already should update z
			#print(ele_ri.shape,r0.shape)
			if ele.length != 0:
				ri.append(ele_ri[:,:])
			else:
				ri[-1]=ele_ri[:,:]
		self.rays = xp.asarray(ri) # xp.swapaxes(xp.asarray(ri),0,1)
		return self.rays

		#Include the initial ray. #TODO: Add conditional if source is included
		#ri = xp.append(r0[:,None,:], ri, axis=1)
		#return ri

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

	def show(self,filename=None,title=None,ylims=None,zlims=None,regenerate=True):
		if self.rays is None or regenerate:
			r1 = self.propagate_ray()
		plot2D(self.rays,zpts = self.labels, filename=filename ,title=title, ylims=ylims,xlims=zlims)

	#def copy(self):
	#	elements = [ e.copy() for e in self.elements ]
	#	dic = self.__dict__ ; dic["elements"]=elements
	#	allowed_kwargs = inspect.signature(Microscope).parameters.keys() # infer allowed kwargs from function itself, and filter down to only those.
	#	dic = { k:v for k,v in dic.items() if k in allowed_kwargs } # e.g., Source doesn't accept "length" even though it technically has one
	#	return MicroscopeSection(**dic)


class Microscope(SEASerializable):
	def __init__(self, name:str='',
				 sections:ArrayLike=None,
				 ndim:int=2,
				 print_fancy:bool=True) -> object:
		self.name = name
		self.sections = sections
		self.ndim = ndim
		self.print_fancy = print_fancy
		self.rays = None
		if self.sections is not None and len(self.sections)>1: # check if consecutive sections are correct length. if not, insert drift at tail of first one
			for s,s2 in zip(self.sections[:-1],self.sections[1:]):
				if s.position+s.length<s2.position:
					dz = s2.position-(s.position+s.length)
					s.insert( len(s.elements) , Drift(position = s.length, length = dz ) )
				if s2.position==0:
					s2.position = s.position+s.length

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

	def __getitem__(self, item):
		# string passed (e.g., name of section or element), "PL1" or "PLs", convert to indices
		if isinstance(item,str):
			item = self.index(item)
		# single item specified: "PL1" or "PLs"
		if isinstance(item,int): # microscope["PLs"] will find index of section, and return that section
			return self.sections[item]
		if isinstance(item,tuple): # microscope["PL1"] finds the element inside a section (indexOfPLss,indexOfPL1WithinPLs)
			return self.sections[item[0]].elements[item[1]]
		# multiple items specified: '"OLs":', or '"sample":' or '3:'
		if isinstance(item,slice):	# convert "sample:" (which results in "item" being a slice) to an integer-indexed slice, e.g. slice(3,None,None)
			a,b,n=item.start,item.stop,item.step
			a,b,n=[ self.index(v) if isinstance(v,str) else v for v in [a,b,n] ]
			if False not in [ v is None or isinstance(v,int) for v in [a,b,n] ]:
				item = slice(a,b,n)
				ret = self.sections[item]
				if isinstance(ret,int):			# microscope["PLs"] will find index of section, and return that section
					return ret
				if isinstance(ret,list):		# microscope["OLs":] will return list of sections OLs,DQCM,PLs, etc, so form into a new Microscope
					return Microscope(name=self.name,sections=ret,ndim=self.ndim,print_fancy=self.print_fancy)
			# microscope["sample":] should return a Microscope containing the sections/elements starting at "sample". if section "OLs" contains "sample", the returned Microscope should contain OLs, plus subsequent sections (e.g. DQCM and PLs), and the OLs section should only contain elements from "sample" and beyond
			a1,b1,n1 = [ v[0] if isinstance(v,tuple) else v for v in [a,b,n] ]
			ret = self.sections[slice(a1,b1,n1)]	# trimmed list of sections

			#for i,s in enumerate(ret):
			#	print("SECTION",i)
			#	print(s)

			if isinstance(a,tuple) and a[1]>0:		# trim first section
				ret[0].elements = ret[0].elements[a[1]:] # TODO what if we do: "PL1:PL3", these are inside the same section, we ought to check if a1==b1
				p0 = ret[0].elements[0].position
				for i,e in enumerate(ret[0].elements):
					ret[0].elements[i].position -= p0
				ret[0].length -= p0
			#if isinstance(b,tuple) and b[1]<len(new.sections[-1]): TODO FINISH IMPLEMENTING
			#	new.sections[-1]=new.sections[-1][:b[1]]
			p1 = ret[0].position
			for i,s in enumerate(ret):				# shift so first section starts at 0
				ret[i].position -= p1
				if i>0:
					ret[i].position -= p0			# subsequent sections ALSO need to be brought forwards by the the shortening of sec0

			#for i,s in enumerate(ret):
			#	print("SECTION",i)
			#	print(s)

			return Microscope(name=self.name,sections=ret,ndim=self.ndim,print_fancy=self.print_fancy)

	def __repr__(self) -> str:
		strings = []
		for s in self.sections:
			header = "Section: "+s.name+" @ "+str(s.position)+" , length="+str(s.length)
			if self.print_fancy:
				print(header)
			strings.append( header )
			strings.append( s.__repr__() )
		if self.print_fancy:
			return ''
		return "\n".join(strings)

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
		return self.rays

	# TWP 2026-03-05 allow element insertion by coordinate ("add a lens midway through this drift section at z=etc"
	def insert(self,index,element):
		#print("microscope insertion",index,element)
		for s in self.sections:
			if s.position <= index < s.position+s.length:
				#print("INSERT ELEMENT",element.name,"AT",index-s.position,"IN SECTION",s.name)
				s.insert(index-s.position,element)
				break
		else: # TODO bug: if we insert a huge drift ("big enough to fill the space") it's just a huge drift. we should update the length based on the space it'll fit. either here, or in Section.insert.
			s = self.sections[-1]
			dz = index-(s.position+s.length)
			s.insert( len(s.elements), Drift(length=dz,position=s.length) )
			#print("APPENDING ELEMENT",element.name,"AT END OF",index-s.position,"IN SECTION",s.name)
			element.position = index-s.position
			s.insert( len(s.elements), element )

	def show(self,filename=None,title=None,ylims=None,zlims=None,regenerate=True):
		if self.rays is None or regenerate:
			r1 = self.propagate_ray()
		sections = { s.name+" ("+str(i)+")":[s.position,s.position+s.length] for i,s in enumerate(self.sections) }# if s.name is not None }
		#print("SECTIONS",sections)
		plot2D(self.rays, zpts=self.labels, sections=sections, filename=filename, title=title, ylims=ylims, xlims=zlims)

	@property
	def labels(self):
		l = {}
		for s in self.sections:
			ls = s.labels
			ls = { k:v+s.position for k,v in ls.items() }
			l = l | ls
		return l

	# Basically just json dumps all attributes, with some special considerations to make the json more human-readable: "Microscope name","Section name","Element name" instead of just "name" for each, specified ordering of attributes (name always first), and nesting lists to go down from Microscope -> Section -> Element
	def save(self,filename):
		jdict = {"Microscope name":self.name,"Sections":[]} | self.__dict__
		del jdict["sections"],jdict["rays"]
		for s in self.sections:
			s_attrs = {"Section name":s.name,"position":s.position,"length":s.length,"Elements":[]} | s.__dict__
			del s_attrs["elements"],s_attrs["rays"],s_attrs["name"]
			for e in s.elements:
				e_attrs = {"Element name":e.name,"kind":e.kind,"position":e.position,"length":e.length} | e.__dict__
				del e_attrs["name"]
				s_attrs["Elements"].append(e_attrs)
			jdict["Sections"].append(s_attrs)
		import json
		with open(filename+'.json', 'w') as f:
			json.dump(jdict, f,indent=4)

	#def copy(self):
	#	sections = [ MicroscopeSection() for s in self.sections ]
	#	sections = [ s.copy() for s in self.sections ]
	#	dic = self.__dict__ ; dic["sections"]=sections
	#	allowed_kwargs = inspect.signature(Microscope).parameters.keys() # infer allowed kwargs from function itself, and filter down to only those.
	#	dic = { k:v for k,v in dic.items() if k in allowed_kwargs } # e.g., Source doesn't accept "length" even though it technically has one
	#	return Microscope(**dic)

	"""
	def to_sea(self,filename):
		if Microscope_SEAS is None:
			print("WARNING: sea_eco does not appear to be installed, so microscope.to_sea is unavailable. Please install sea_eco, or use microscope.save instead")
		else:
			ms = self.as_seatype()
			#print(ms.to_dict())
			ms.to_sea(filename)

	def as_seatype(self):
		if Microscope_SEAS is None:
			return self
		return Microscope_SEAS(self)
	"""

"""
sea_available = False
try:
	sys.path.insert(1,"../../")
	from pySEA.sea_eco.architecture.base_structure import SEASerializable
	sea_available = True
except:
	pass

if sea_available:
	class Microscope_SEAS(SEASerializable,Microscope):
		def __init__(self,microscope=None):
			if microscope is None:
				return
			for k,v in microscope.__dict__.items():
				print("setattr(self,k,v)",k,v)
				setattr(self,k,v)
			# In order for SEASerializable.to_sea to work (and friends: to_dict, to_hdf5_group...), all lists must be either lists of values, or lists of SEASerializable objects too (not generic other objects like Section or Element). Names must also be unique since the names are used for hdf5 group names.
			# Loop through Sections, converting each Section to a SEASerializable Section (Section_SEAS), and ensure each name is set.
			sections = self.sections ; self.sections = []
			for i,s in enumerate(sections):
				if s.name is None or len(s.name)==0:
					s.name = "None_"+str(i)
				self.sections.append( Section_SEAS(s) )
			# also make sure this Microscope object's name is set
			if self.name is None or len(self.name)==0:
				self.name="None"
		#def propagate_ray(self,*args,**kwargs):
		#	return super().propagate_ray(*args,**kwargs)
	class Section_SEAS(SEASerializable,MicroscopeSection):
		def __init__(self,section):
			for k,v in section.__dict__.items():
				setattr(self,k,v)
			# Loop through Elements, converting each Element to a SEASerializable Element (Element_SEAS), and ensure each name is set.
			elements = self.elements ; self.elements = []
			for i,e in enumerate(elements):
				if e.name is None or len(e.name)==0:
					e.name = "None_"+str(i)
				self.elements.append( Element_SEAS(e) )
	class Element_SEAS(SEASerializable,Element):
		def __init__(self,element):
			for k,v in element.__dict__.items():
				setattr(self,k,v)
else:
	Microscope_SEAS = None ; Section_SEAS = None; Element_SEAS = None
"""

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
	mapping = { "Drift":Drift, "QLens":Lens, "Source":Source } # TODO Eventually need to support all Element types from elements.py. and is there a way to map these automatically instead of explicitly?

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

	#
	#			jdict = {"Microscope name":self.name,"Sections":[]} | self.__dict__
	#	del jdict["sections"],jdict["rays"]
	#	for s in self.sections:
	#		s_attrs = {"Section name":s.name,"position":s.position,"length":s.length,"Elements":[]} | s.__dict__
	#		del s_attrs["elements"],s_attrs["rays"],s_attrs["name"]
	#		for e in s.elements:
	#			e_attrs = {"Element name":e.name,"kind":e.kind,"position":e.position,"length":e.length} | e.__dict__
	#			del e_attrs["name"]
	#			s_attrs["Elements"].append(e_attrs)
	#		jdict["Sections"].append(s_attrs)
	#	import json
	#	with open(filename+'.json', 'w') as f:
	#		json.dump(jdict, f,indent=4)
	#
	#	#with open(filename,"w") as fo:

