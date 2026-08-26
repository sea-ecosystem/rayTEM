# A SIMPLIFIED MODEL OF THE MACSTEM WITH: C1,C2,C3, OLs, PLs, no quads, no corrector, no dipoles

import sys,os
sys.path.insert(1,"../../../../")
from pySEA.rayTEM.elements import Lens,Drift,Source
from pySEA.rayTEM.assemblies import Microscope,MicroscopeSection
from pySEA.rayTEM.postprocessing import plot2D

import xml.etree.ElementTree as ET

from pySEA.rayTEM.xmlNion import lookupStrengthsXML,lookupPositions #,lookupCurrentStrengthsXML,rootControlSettingValue

# We're going to look up the positions (from the schematic) and strengths (from the AS2 config file), so we need to map our sane element names to their less-sane (and possibly different) names across data sources.

# addition planes we want labeled
planes = ["VOA", "sample", "CCD"]

positions_file = "lens_positions.txt"
xml_file = "AS2restore_20260103.xml"

# For parsing the AS2 xml files, we use lookupStrengthsXML (rather than lookupCurrentStrengthsXML), just in case the reference settings aren't "active" in the xml file. Use rootControlSettingValue to figure out the paths for each element.
#print( rootControlSettingValue(level="R",path="",filename=xml_file) ) ; sys.exit() # list "root" nodes (sections)
#print( rootControlSettingValue(level="C",path="S_Condensers",filename=xml_file) )
#print( rootControlSettingValue(level="C",path="S_OL",filename=xml_file) )
#print( rootControlSettingValue(level="C",path="S_Projectors",filename=xml_file) ) ; sys.exit()
path_lookup = { "condenser":"S_Condensers/30mrad15iRef", "objective":"S_OL/On", "projector":"S_Projectors/_Diffn 20mm  (ref)" }

microscope_origin = lookupPositions("CL1_2R",positions_file)+100 # positions file stores positions reversed (0 at end of microscope), so to get each element's position, we'll take this number, minus the element's nominal position

sections = [] ; length=0
# FOR EACH SECTION
for i,s in enumerate(elements_to_AS2names.keys()):
	if i==0: # first section needs an electron gun, and the section's "origin" x0 is at 0
		elements = [ Source(name="gun",size=(2e-4,2e-4),np_xy=(11,11),angle=(0,0),na_xy=(1,1)) ] ; x0=0
	else:
		elements = [] ; x0 = length
	# FOR EACH ELEMENT IN THAT SECTION
	for e in elements_to_AS2names[s].keys():
		# LOOK UP ITS POSITION
		position_alias = elements_to_schematicnames[s][e]			# |___P4___P3___P2___P1___|O2___O1___|C3___C2___C1___S|
		position = lookupPositions(position_alias,positions_file)	# |      projectors       | objective|   condensers   |
		position = microscope_origin - position - x0 				# position relative to start of section is flipped and minus x0
		# AND LENS STRENGTH
		AS2_alias = elements_to_AS2names[s][e]
		xml_path = path_lookup[s]+"/"+AS2_alias
		strength = lookupStrengthsXML(xml_path,filename=xml_file)
		# CREATE A NEW RAYTEM ELEMENT
		#L = 8. if s == "projector" else .08
		#C = .05 if s == "projector" else 1
		L = .08 ; C=1
		elements.append( Lens(strength=strength,length=L,name=e,position=position, calibration=C) )
		length = x0+position+L # update total length of the microscope (to ensure subsequent sections are placed correctly)
	# CREATE NEW RAYTEM SECTION
	section=MicroscopeSection(elements=elements,position=x0,name=s)
	sections.append(section)
microscope = Microscope(sections=sections)
# HANDLE ADDITIONAL NAMED PLANES: (these are named Drift sections, inserted)
for p,l in zip(planes,[10,10,100]):
	position = lookupPositions(p,positions_file)
	position = microscope_origin - position
	microscope.insert( position , Drift(length=l,name=p) )

print(microscope)
microscope.show()

microscope.save("macstem")

#os.system("xxdiff macstem.json ../MACSTEM_v3/macstem.json")



