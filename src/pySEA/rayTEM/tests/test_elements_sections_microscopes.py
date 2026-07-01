# TESTS BELOW INCLUDE: (please run: "pytest elements_sections_microscopes.py")
# basic building of MicroscopeSections (comprised of multiple elements), Microscopes (comprised of multiple sections)
# building from stacked elements (Lens/Drift/Lens/...) and building from specified positions (Lens1 at z1, Lens2 at 2,...)
# image and diffraction plane determination (findPlanes function)
# microscope saving and reloading via both json and sea (see https://github.com/sea-ecosystem/sea-eco/)

import sys,os,pytest
sys.path.insert(1,"../../../")
from pySEA.rayTEM import Source,Lens,Drift,Aperture,Dipole,Quadrapole
from pySEA.rayTEM import MicroscopeSection,Microscope
from pySEA.rayTEM import fix_ray_dims,plot2D,findPlanes,columnByName,load_microscope,load_section
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
	ret = findPlanes(r1,axis="x")
	Zd=ret['x']['diff']['z'][0] ; Md=ret['x']['diff']['M'][0]
	Zi=ret['x']['image']['z'][0] ; Mi=ret['x']['image']['M'][0]
	planes = np.asarray([Zd,Md,Zi,Mi]) ; print(planes)
	planes_old = np.asarray([ 4.19935249 , 0.45085376 , 4.37742984, -0.39497808])
	print(section[1])
	print(section)
	assert np.sqrt(np.sum((planes-planes_old)**2)) < .0001
test_basic_section_r0()

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

# basic stack including Source, Drift, Lens, Aperture (TODO add additional elements as support is added)
# test: resulting rays should always be identical (compare to numpy saved rays), plotting should work
def test_every_element():
	elements = [ Source(size=(1,1),np_xy=(3,3),angle=(1,1),na_xy=(3,3)), Drift(length=1), Lens(strength=3,length=.1), Aperture(radius=2,position=5,name="VOA"),Quadrapole(strength=5,length=.1,position=6), Drift(length=1,name="detector",position=7)  ]
	section = MicroscopeSection(elements=elements)
	section.show(filename="elements_sections_microscopes_every_element.png")
	r1 = section.propagate_ray()
	filename = "elements_sections_microscopes_every_element.npy"
	if not os.path.exists(filename):
		np.save(filename,r1)
	r1_old = np.load(filename)
	assert np.sqrt(np.sum((r1-r1_old)**2)) < .0001 # serves as a "hash" of sorts to ensure we're getting the same rays out
#test_every_element()

def test_dipole_transfer_matrix():
	r0 = np.asarray( [[0,0,0,0]] )
	r0 = fix_ray_dims(r0,["x","xt","y","yt"])
	r0[:,columnByName("I")] = 1

	r1 = Dipole(strength=.2,axis="x").propagate_ray(r0) ; print(r1)
	assert r1[0,columnByName("x")] == 0			# beam is centered, tilted, so should remain centered (until free-space propagation)
	assert r1[0,columnByName("xt")] == .2		# zero-thickess: strength is simply tilt angle

	r1 = Dipole(length=2,strength=.2,axis="y").propagate_ray(r0)
	assert r1[0,columnByName("y")] == 0			# beam is centered, tilted, so should remain centered
	assert r1[0,columnByName("yt")] == .4		# tilted in y this time, strength x length
	assert r1[0,columnByName("z")] == 2			# beam is propagated by length in z
#test_dipole_transfer_matrix()

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
	ret = findPlanes(r1,axis="x")
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

def test_cropping_section():
	# ASSEMBLE
	elements = [ Source(name="S",size=(1,1),np_xy=(3,3),angle=(1,1),na_xy=(3,3)),
				Drift(name="D1",length=1),
				Lens(name="L1",strength=3,length=.1),
				Drift(name="D2",length=.4),
				Lens(name="L2",strength=5,length=.1),
				Drift(name="D3",length=1)  ]
	section = MicroscopeSection(elements=elements)
	r1 = section.propagate_ray()
	#section.show()
	# CROP BY INDEX, REPLACE SOURCE
	section = section[1:]
	section.insert(0, Source(size=(1,1),np_xy=(3,3),angle=(1,1),na_xy=(3,3)) )
	r2 = section.propagate_ray()
	# RAYS MUST MATCH
	assert np.sqrt(np.sum((r1[-1]-r2[-1])**2)) < .0001

	# INFER POSITIONS OF ELEMENTS
	positions_1 = [ e.position for e in section.elements ]
	#print(positions_1) ; print(repr(section))
	# SLICE BY ELEMENT NAME
	#section.show()
	z_D2 = section["D2"].position
	section = section["D2":]
	section.insert(0, Source(size=(1,1),np_xy=(3,3),angle=(1,1),na_xy=(3,3)) )
	#section.show()
	#print(repr(section))
	# AGAIN INFER POSITIONS OF ELEMENTS
	positions_2 = [ e.position for e in section.elements ]
	#print(positions_2) ; print(repr(section))

	# CHECK: if we successfully sliced at D2, the microscope should now start at D2, and positions should be shifted by z_D2
	for v1,v2 in zip(reversed(positions_1[2:]),reversed(positions_2[2:])):
		assert v1==v2+z_D2

	# NOW TRY CROPPING BY FLOAT Z LOCATION
	section = MicroscopeSection(elements=elements)
	section = section[0.5:]
	positions_3 = [ e.position for e in section.elements ]
	#print(positions_3) ; print(repr(section))
	for v1,v3 in zip(reversed(positions_1[2:]),reversed(positions_3[2:])):
		#print(v1,v3)
		assert v1==v3+.5


def test_cropping_microscope():
	# ASSEMBLE
	ele1 = [ Source(name="S",size=(1,1),np_xy=(3,3),angle=(1,1),na_xy=(3,3)),
				Drift(name="D1",length=1),
				Lens(name="L1",strength=3,length=.1),
				Drift(name="D2",length=.4),
				Lens(name="L2",strength=5,length=.1),
				Drift(name="D3",length=1)  ]
	ele2 = [ Lens(name="L3",strength=3,length=.1),
				Drift(name="D4",length=.4),
				Lens(name="L4",strength=5,length=.1),
				Drift(name="D5",length=1)  ]

	s1 = MicroscopeSection(elements=ele1)
	s2 = MicroscopeSection(elements=ele2)
	microscope = Microscope(sections=[s1,s2])
	# INFER POSITIONS OF ELEMENTS
	positions_1 = sum( [[ e.position+s.position for e in s.elements ] for s in microscope.sections ] , [] )
	#print(positions_1) ; print(repr(microscope))
	#microscope.show()
	# SLICE
	z_D2 = microscope["D2"].position
	microscope = microscope["D2":]
	microscope.insert(0, Source(name="S2",size=(1,1),np_xy=(3,3),angle=(1,1),na_xy=(3,3)) )

	# AGAIN INFER POSITIONS OF ELEMENTS
	positions_2 = sum( [[ e.position+s.position for e in s.elements ] for s in microscope.sections ] , [] )
	#print(positions_2) ; print(repr(microscope))
	#microscope.show()

	# CHECK: if we successfully sliced at D2, the microscope should now start at D2, and positions should be shifted by z_D2
	for v1,v2 in zip(reversed(positions_1[2:]),reversed(positions_2[2:])):
		assert v1==v2+z_D2

	# NOW TRY CROPPING BY FLOAT Z LOCATION
	microscope = Microscope(sections=[s1,s2])
	microscope = microscope[0.5:]
	positions_3 = sum( [[ e.position+s.position for e in s.elements ] for s in microscope.sections ] , [] )
	#print(positions_2) ; print(repr(microscope))
	for v1,v3 in zip(reversed(positions_1[2:]),reversed(positions_3[2:])):
		assert v1==v3+.5


#test_cropping_section()
#test_cropping_microscope()

def test_element_move():
	elements = [ Drift(length=1), Lens(name="L1",strength=3,length=.1), Drift(length=1.4), Lens(name="L2",strength=5,length=.1), Drift(length=2)  ]
	section = MicroscopeSection(elements=elements)
	# user should *not* be allowed to update the position of element, since it requires updating surrounding elements!
	with pytest.raises(AttributeError):
		section["L2"].position += 1

	# sanity check: nextposition-thisposition should equal thislength. we'll use this function to verify a successful move
	def check_lengths(section):
		zs = np.asarray( [ e.position for e in section.elements ] )
		ls = np.asarray( [ e.length for e in section.elements ] )
		dz = zs[1:]-zs[:-1] ; print(zs,ls)
		assert np.sum( np.absolute( dz-ls[:-1] ) ) < .00001
	check_lengths(section)

	# user should *instead* use the move function to move an element (move function likely needs to be on the Section, not the Element, since the Element doesn't know it's parents??)
	section.move("L2",dz=1)
	check_lengths(section)
	section.move("L1",dz=-.5)
	check_lengths(section)


#test_element_move()

