# TESTS BELOW INCLUDE: (please run: "pytest elements_sections_microscopes.py")
# basic building of MicroscopeSections (comprised of multiple elements), Microscopes (comprised of multiple sections)
# building from stacked elements (Lens/Drift/Lens/...) and building from specified positions (Lens1 at z1, Lens2 at 2,...)
# image and diffraction plane determination (findPlanes function)
# microscope saving and reloading via both json and sea (see https://github.com/sea-ecosystem/sea-eco/)

import sys,os
sys.path.insert(1,"../../../")
from pySEA.rayTEM import Source,Lens,Drift,MicroscopeSection,Microscope,fix_ray_dims,plot2D,findPlanes,columnByName,load_microscope,load_section
import numpy as np

# basic Drift/Lens/Drift/Lens/Drift configuration. Manually-defined input rays (one pair of axial and one pair of field rays)
# test: resulting rays should always be identical (compare to numpy saved rays)
# test: inferred image and diffraction planes should always be identical (use findPlanes and compare to hardcoded result)
def test_basic_section_r0():
	elements = [ Drift(length=1), Lens(strength=3,length=.1), Drift(length=.4), Lens(strength=5,length=.1), Drift(length=1)  ]
	section = MicroscopeSection(elements=elements)
	r0=np.asarray( [[1,0,0,0],[.5,0,0,0],[0,0,1,0],[0,0,.5,0]] )
	r0=fix_ray_dims(r0,["x","y","xt","yt"])
	r1 = section.propagate_ray(r0)
	#plot2D(r1)
	filename = "elements_sections_microscopes_basic_section_r0_rays.npy"
	if not os.path.exists(filename):
		np.save(filename,r1)
	r1_old = np.load(filename)
	assert np.sqrt(np.sum((r1-r1_old)**2)) < .0001 # serves as a "hash" of sorts to ensure we're getting the same rays out
	ret = findPlanes(r1,axes="x")
	Zd=ret['x']['diff']['z'][0] ; Md=ret['x']['diff']['M'][0]
	Zi=ret['x']['image']['z'][0] ; Mi=ret['x']['image']['M'][0]
	planes = np.asarray([Zd,Md,Zi,Mi])
	planes_old = np.asarray([ 4.1993524879728845, 0.3779897737285624, 4.377429840313133, -0.33114434377184565 ])
	print(section[1])
	print(section)
	assert np.sqrt(np.sum((planes-planes_old)**2)) < .0001
#test_basic_section_r0()

# basic Drift/Lens/Drift/Lens/Drift configuration. automatically-defined input rays, via the Source object
# test: resulting rays should always be identical (compare to numpy saved rays)
def test_basic_section_wsource():
	elements = [ Source(size=(1,1),np_xy=(3,3),angle=(1,1),na_xy=(3,3)), Drift(length=1), Lens(strength=3,length=.1), Drift(length=.4), Lens(strength=5,length=.1), Drift(length=1)  ]
	section = MicroscopeSection(elements=elements)
	section.to_sea("elements_sections_microscopes_basic_section_wsource.sea")
	r1 = section.propagate_ray()
	#plot2D(r1)
	filename = "elements_sections_microscopes_basic_section_wsource_rays.npy"
	if not os.path.exists(filename):
		np.save(filename,r1)
	r1_old = np.load(filename)
	assert np.sqrt(np.sum((r1-r1_old)**2)) < .0001 # serves as a "hash" of sorts to ensure we're getting the same rays out
#test_basic_section_wsource()

# basic two-section assembly: Drift/Lens/Drift/Lens/Drift + Lens/Drift. lens/drift lengths define element positions, section lengths, etc
# test: resulting rays should always be identical (compare to numpy saved rays)
# test: inferred image and diffraction planes should always be identical (use findPlanes and compare to numpy saved concatenated lists)
def test_basic_microscope_defined_by_lengths():
	elements = [ Source(size=(1,1),np_xy=(3,3),angle=(1,1),na_xy=(3,3)), Drift(length=1), Lens(strength=3,length=.1), Drift(length=.4), Lens(strength=4.5,length=.1), Drift(length=1)  ]
	section1 = MicroscopeSection(elements = elements)
	elements = [ Lens(strength=5,length=.1), Drift(length=2) ]
	section2 = MicroscopeSection(elements = elements)
	microscope = Microscope(sections = [ section1,section2 ])
	#microscope.show()
	r1 = microscope.propagate_ray()
	filename = "elements_sections_microscopes_basic_microscope_defined_by_lengths_rays.npy"
	if not os.path.exists(filename):
		np.save(filename,r1)
	r1_old = np.load(filename)
	assert np.sqrt(np.sum((r1-r1_old)**2)) < .0001 # serves as a "hash" of sorts to ensure we're getting the same rays out
	ret = findPlanes(r1,axes="x")
	Zd=ret['x']['diff']['z'] ; Md=ret['x']['diff']['M']
	Zi=ret['x']['image']['z'] ; Mi=ret['x']['image']['M']
	#print(Zd,Md,Zi,Mi)
	planes = np.asarray(list(Zd)+list(Md)+list(Zi)+list(Mi))
	filename = "elements_sections_microscopes_basic_microscope_defined_by_lengths_planes.npy"
	if not os.path.exists(filename):
		np.save(filename,planes)
	planes_old = np.load(filename)
	assert np.sqrt(np.sum((planes-planes_old)**2)) < .0001
	#plot2D(r1)
	filename = "elements_sections_microscopes_basic_microscope_defined_by_lengths.json"
	microscope.save(filename)	# ALWAYS save off, so subsequent tests can make sure the saving still works (not just checking that reload hasn't changed). We will also manually(?*) back up these saved microscopes so we can test them (*is there a better way?)
	filename = "elements_sections_microscopes_basic_microscope_defined_by_lengths.sea"
	microscope.to_sea(filename)
	#print(microscope)
#test_basic_microscope_defined_by_lengths()

# basic two-section assembly: Drift/Lens/Drift/Lens/Drift + Lens/Drift. lens/drift, identical to above, except elements are defined by their positions and intermediate Drifts are inferred
# test: resulting rays should always be identical (compare to numpy saved rays)
def test_basic_microscope_defined_by_positions():
	elements = [ Source(size=(1,1),np_xy=(3,3),angle=(1,1),na_xy=(3,3)), Lens(strength=3,length=.1,position=1), Lens(strength=4.5,length=.1,position=1.5) ]
	section1 = MicroscopeSection(elements = elements)
	elements = [ Lens(strength=5,length=.1), Drift(length=2) ]
	section2 = MicroscopeSection(elements = elements,position=2.6)
	microscope = Microscope(sections = [ section1,section2 ])
	#microscope.show()
	r1 = microscope.propagate_ray()
	filename = "elements_sections_microscopes_basic_microscope_defined_by_lengths_rays.npy"
	if not os.path.exists(filename):
		print("ERROR: test_basic_microscope_defined_by_positions requires test_basic_microscope_defined_by_lengths to run first")
		assert 1==0
	r1_old = np.load(filename)
	assert np.sqrt(np.sum((r1-r1_old)**2)) < .0001 # serves as a "hash" of sorts to ensure we're getting the same rays out
#test_basic_microscope_defined_by_positions()

# basic two-section assembly, reloaded from json file.
# test: resulting rays should always be identical (compare to numpy saved rays)
def test_basic_microscope_reload_json():
	microscope = load_microscope("elements_sections_microscopes_basic_microscope_defined_by_lengths.json")
	#microscope.show()
	r1 = microscope.propagate_ray()
	filename = "elements_sections_microscopes_basic_microscope_defined_by_lengths_rays.npy"
	if not os.path.exists(filename):
		print("ERROR: test_basic_microscope_reload_json requires test_basic_microscope_defined_by_lengths to run first")
		assert 1==0
	r1_old = np.load(filename)
	assert np.sqrt(np.sum((r1-r1_old)**2)) < .0001 # serves as a "hash" of sorts to ensure we're getting the same rays out
#test_basic_microscope_reload_json()

# basic two-section assembly, reloaded from sea file.
# test: resulting rays should always be identical (compare to numpy saved rays)
def test_basic_microscope_reload_sea():
	microscope = load_microscope("elements_sections_microscopes_basic_microscope_defined_by_lengths.sea")
	#microscope.show()
	r1 = microscope.propagate_ray()
	filename = "elements_sections_microscopes_basic_microscope_defined_by_lengths_rays.npy"
	if not os.path.exists(filename):
		print("ERROR: test_basic_microscope_reload_sea requires test_basic_microscope_defined_by_lengths to run first")
		assert 1==0
	r1_old = np.load(filename)
	assert np.sqrt(np.sum((r1-r1_old)**2)) < .0001 # serves as a "hash" of sorts to ensure we're getting the same rays out
#test_basic_microscope_reload_sea()

#def test_insertion_section():
#	section = load_section("elements_sections_microscopes_basic_section_wsource.sea")
#	section.insert(1,Lens(name="inserted by index at 1",strength=0))
#	section.insert(1.25,Lens(name="inserted by position at 1.25",strength=0))
#	r1 = microscope.propagate_ray()
#	filename = "elements_sections_microscopes_basic_microscope_defined_by_lengths_rays.npy"
#	if not os.path.exists(filename):
#		print("ERROR: test_insertion_section requires test_basic_section_wsource to run first")
#		assert 1==0
#	r1_old = np.load(filename)
#	assert np.sqrt(np.sum((r1-r1_old)**2)) < .0001 # serves as a "hash" of sorts to ensure we're getting the same rays out

def test_element_insertion_microscope():
	filename = "elements_sections_microscopes_basic_microscope_defined_by_lengths.sea"
	if not os.path.exists(filename):
		print("ERROR: test_insertion_microscope requires test_basic_microscope_defined_by_lengths to run first")
		assert 1==0
	microscope = load_microscope(filename)
	#section.insert(1,Lens(name="inserted by index at 1",strength=0))
	microscope.insert(1.25,Lens(name="inserted by position at 1.25",strength=0))
	r1 = microscope.propagate_ray()[-1,:,:]
	filename = "elements_sections_microscopes_basic_microscope_defined_by_lengths_rays.npy"
	if not os.path.exists(filename):
		print("ERROR: test_insertion_microscope requires test_basic_microscope_defined_by_lengths to run first")
		assert 1==0
	r1_old = np.load(filename)[-1,:,:]
	assert np.sqrt(np.sum((r1-r1_old)**2)) < .0001 # serves as a "hash" of sorts to ensure we're getting the same rays out

def test_section_insertion_microscope():
	ele1 = [ Source(size=(1,1),np_xy=(3,3),angle=(1,1),na_xy=(3,3)),
			 Lens(strength=1,length=.1,position=1), Lens(strength=1,length=.1,position=3) ] # lens at 1 and 3
	ele2 = [ Lens(strength=1,length=.1,position=1) , Drift(name="CCD",length=1,position=3) ] # when placed at 4, yields a lens at 5 7
	sec1 = MicroscopeSection( elements = ele1, name="sec1" )
	sec2 = MicroscopeSection( elements = ele2, position=4, name="sec2" )
	microscope = Microscope(sections = [ sec1, sec2 ])
	#microscope.show()
	print("OLD") ; print(microscope)
	r1 = microscope.propagate_ray()
	filename = "elements_sections_microscopes_section_insertion_microscope_rays.npy"
	if not os.path.exists(filename):
		np.save(filename,r1)

	inserted_strength = 0
	ele3 = [ Lens(strength=inserted_strength,length=.1,position=0,name="added 1") ,
				Lens(strength=inserted_strength,length=.1,position=.25,name="added 2") ,
					Lens(strength=inserted_strength,length=.1,position=0.5,name="added 3") ]
	sec3 = MicroscopeSection( elements = ele3, name="newsec" )
	microscope.insert(2.0,sec3)

	print("NEW") ; print(microscope)
	#microscope.show()
	r1 = microscope.propagate_ray()

	r1_old = np.load(filename)
	#print(np.sqrt(np.sum((r1[-1]-r1_old[-1])**2)))
	assert np.sqrt(np.sum((r1[-1]-r1_old[-1])**2)) < .0001 # serves as a "hash" of sorts to ensure we're getting the same rays out

#test_section_insertion_microscope()





