"""
seashells serves as a wrapper around the sea_eco SEASerializable object, enabling easy integration with sea_eco.
to install sea_eco and rayTEM side-by-side, this module should be installed as a plugin inside the pySEA folder, as a sibling to sea_eco, also in the pySEA folder
if sea_eco IS installed, the SEASerializable object is wrapped, enabling direct access to, or wrapping of, all SEASerializable functions
to wrap a function, we simply define it, do our custom stuff, then call super().funcname to call up to SEASerializable's version, and do more custom stuff after
if sea_eco is NOT installed, we create a dummy SEASerializable object, with dummy functions (which raise warnings) for the functions we expect to use
All objects we then expect to integrate with sea_eco then inherit the SEASerializable class from here (whether it is wrapping sea_eco's SEASerializable, or using the dummy version)
"""

import sys,inspect

sea_available = False
try:
	sys.path.insert(1,"../../")
	from pySEA.sea_eco.architecture.base_structure import SEASerializable as _SEASerializable
	sea_available = True
except Exception as e:
	print(str(e)+". This is just a warning from rayTEM.seashells: rayTEM pySEA integration will not work.")
	pass

if sea_available:
	class SEASerializable(_SEASerializable):
		def to_sea(self,filename):												# sea_eco's SEASerializable will default naming like "Drift_2" but we'll set them to None so we can easily undo the naming later
			if hasattr(self,"sections"):										# user might to_sea a Microscope object (loop section > elements)
				if self.name is None or len(self.name)==0:
					self.name = "None_Microscope"
				for s,sec in enumerate(self.sections):
					if sec.name is None or len(sec.name)==0:
						sec.name = "None_"+str(s)
					for e,ele in enumerate(sec.elements):
						if ele.name is None or len(ele.name)==0:
							ele.name = "None_"+str(e)
			elif hasattr(self,"elements"):										# ... or a MicroscopeSection object (loop elements only)
				if self.name is None or len(self.name)==0:
					self.name = "None_Section"
				for e,ele in enumerate(self.elements):
					if ele.name is None or len(ele.name)==0:
						ele.name = "None_"+str(e)
			super().to_sea(filename)
		def from_sea(self,filename):											# sea_eco will reaload purely-SEASerializable objects, so reinitalize as our shared-inheritance object
			super().from_sea(filename)
			if hasattr(self,"sections"):
				if "None" in self.name:
					self.name = ""
				sections = []
				for s,sec in enumerate(self.sections):
					if "None" in sec.name:
						sec.name = ""
					elements = []
					for e,ele in enumerate(sec.elements):
						if "None" in ele.name:
							ele.name = ""
						elements.append( safeReinstantiate(ele,ele.kind) )
					sec.elements = elements
					sections.append( safeReinstantiate(sec,"Section") )
				self.sections = sections
			elif hasattr(self,"elements"):
				if "None" in self.name:
					self.name = ""
				elements = []
				for e,ele in enumerate(self.elements):
					if "None" in ele.name:
						ele.name = ""
					elements.append( safeReinstantiate(ele,ele.kind) )
				self.elements = elements
else:
	class SEASerializable():
		def __init__(self):
			pass
		def to_sea(self,filename):
			print("WARNING: sea_eco does not appear to be installed, so microscope.to_sea is unavailable. Please install sea_eco, or use microscope.save instead")
		def from_sea(self,filename):
			print("WARNING: sea_eco does not appear to be installed, so microscope.to_sea is unavailable. Please install sea_eco, or use microscope.save instead")


#SEASerializable.from_sea will create a purely-SEASerializable object. rayTEM objects (Element, MicroscopeSection, Microscope, etc) will have inherited from SEASerializable, so we may need to reinstantiate rayTEM objects to ensure they have the rayTEM-specific functionality (e.g. "scope=Microscope(); scope.from_sea" will find scope.sections is a list of purely-SEASerializable objects without functions like "propagate_ray").
def safeReinstantiate(source,cls):
	from .elements import Drift,Lens,Source,Dipole,Quadrapole
	from .assemblies import Microscope,MicroscopeSection
	cls = {"Drift":Drift, "QLens":Lens, "Thin lens":Lens, "Source":Source, "Microscope":Microscope, "Section":MicroscopeSection, "Dipole":Dipole, "Thin dipole":Dipole, "Quad":Quadrapole, "Thin quad":Quadrapole }[cls]
	dic = source.__dict__
	allowed_kwargs = inspect.signature(cls).parameters.keys()	# infer allowed kwargs from the class itself
	dic = { k:v for k,v in dic.items() if k in allowed_kwargs }	# and filter kwargs to those accepted
	return cls(**dic)
