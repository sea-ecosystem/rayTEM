# TESTS BELOW INCLUDE: (please run: "pytest elements_sections_microscopes.py")
# basic building of MicroscopeSections (comprised of multiple elements), Microscopes (comprised of multiple sections)
# building from stacked elements (Lens/Drift/Lens/...) and building from specified positions (Lens1 at z1, Lens2 at 2,...)
# image and diffraction plane determination (findPlanes function)
# microscope saving and reloading via both json and sea (see https://github.com/sea-ecosystem/sea-eco/)
#
# NOTE on the ray convention: rays are purely geometric — convention = ["x","xt","y","yt","z","E"].
# Intensity (I) and cumulative Larmor rotation (R) are tracked as separate parallel arrays on the
# section/microscope (.I and .R), NOT as ray columns, so convert_to_rotating_reference_frame and
# findPlanes now take the R array explicitly.

import sys,os,pytest
sys.path.insert(1,"../../../")
from pySEA.rayTEM import Source,Lens,Drift,Aperture,Dipole,Quadrapole
from pySEA.rayTEM import MicroscopeSection,Microscope,check_lengths
from pySEA.rayTEM import fix_ray_dims,plot2D,plot3D,findPlanes,columnByName,load_microscope,load_section
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
	rr = r1.convert_to_rotating_reference_frame() # 20260723: updated to default to rotate, so we need to convert to match previous rotating-reference-frame saved rays
	#print(repr(section))
	#print(r1)
	#plot2D(r1,section.R)
	filename = "elements_sections_microscopes_basic_section_r0_rays.npy"
	if not os.path.exists(filename):
		np.save(filename,rr)
	rr_old = np.load(filename)
	assert np.sqrt(np.sum((rr-rr_old)**2)) < .0001 # serves as a "hash" of sorts to ensure we're getting the same rays out
	ret = findPlanes(r1,axis="x")
	Zd=ret['x']['diff']['z'][0] ; Md=ret['x']['diff']['M'][0]
	Zi=ret['x']['image']['z'][0] ; Mi=ret['x']['image']['M'][0]
	planes = np.asarray([Zd,Md,Zi,Mi]) ; print(planes)
	planes_old = np.asarray([ 4.19935249 , 0.45085376 , 4.37742984, -0.39497808])
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
	r1 = r1.convert_to_rotating_reference_frame() # 20260723: updated to default to rotate, so we need to convert to match previous rotating-reference-frame saved rays
	#plot2D(r1,section.R)
	filename = "elements_sections_microscopes_basic_section_wsource_rays.npy"
	if not os.path.exists(filename):
		np.save(filename,r1)
	r1_old = np.load(filename)
	assert np.sqrt(np.sum((r1-r1_old)**2)) < .0001 # serves as a "hash" of sorts to ensure we're getting the same rays out
#test_basic_section_wsource()

def test_various_lenses():
	elements = [ Source(size=(1,1),np_xy=(3,3),angle=(1,1),na_xy=(3,3)),
					Lens(position=1,strength=3,length=.1),
					Lens(position=2,focal_length=3,length=0),
					Drift(position=2,length=1)
			 ]
	section = MicroscopeSection(elements=elements)
	#section.show()
	r1 = section.propagate_ray()
	filename = "elements_sections_microscopes_test_various_lenses.npy"
	if not os.path.exists(filename):
		np.save(filename,r1)
	r1_old = np.load(filename)[:,:,:6]
	assert np.sqrt(np.sum((r1-r1_old)**2)) < .0001 # serves as a "hash" of sorts to ensure we're getting the same rays out
#test_various_lenses()

# basic stack including Source, Drift, Lens, Aperture (TODO add additional elements as support is added)
# test: resulting rays should always be identical (compare to numpy saved rays), plotting should work
def test_every_element():
	elements = [ Source(size=(1,1),np_xy=(3,3),angle=(1,1),na_xy=(3,3)), Drift(length=1), Lens(strength=3,length=.1), Aperture(radius=2,position=5,name="VOA"),Quadrapole(strength=5,length=.1,position=6), Drift(length=1,name="detector",position=7)  ] # 20260723: updated to default to rotate, BUT, quad-after-rotation in rotating reference frame means the quad axis is dependnt on prior lens's rotation! very very strange. in other tests, we convert to match previous rotating-reference-frame saved rays, but here, we will simply regenerate our saved rays.
	section = MicroscopeSection(elements=elements)
	section.show(filename="elements_sections_microscopes_every_element.png")
	r1 = section.propagate_ray()
	print(repr(section))
	plot3D(r1,filename="elements_sections_microscopes_every_element2.png")
	filename = "elements_sections_microscopes_every_element.npy"
	if not os.path.exists(filename):
		np.save(filename,r1)
	r1_old = np.load(filename)
	assert np.sqrt(np.sum((r1-r1_old)**2)) < .0001 # serves as a "hash" of sorts to ensure we're getting the same rays out
#test_every_element()

def test_dipole_transfer_matrix():
	r0 = np.asarray( [[0,0,0,0]] )
	r0 = fix_ray_dims(r0,["x","xt","y","yt"])

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
	rr = r1.convert_to_rotating_reference_frame() # 20260723: updated to default to rotate, so we need to convert to match previous rotating-reference-frame saved rays
	filename = "elements_sections_microscopes_basic_microscope_defined_by_lengths_rays.npy"
	if not os.path.exists(filename):
		np.save(filename,rr)
	rr_old = np.load(filename)
	assert np.sqrt(np.sum((rr-rr_old)**2)) < .0001 # serves as a "hash" of sorts to ensure we're getting the same rays out
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
	#plot2D(r1,microscope.R)
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
	r1 = r1.convert_to_rotating_reference_frame() # 20260723: updated to default to rotate, so we need to convert to match previous rotating-reference-frame saved rays
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
	r1 = r1.convert_to_rotating_reference_frame() # 20260723: updated to default to rotate, so we need to convert to match previous rotating-reference-frame saved rays
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
	r1 = r1.convert_to_rotating_reference_frame() # 20260723: updated to default to rotate, so we need to convert to match previous rotating-reference-frame saved rays
	filename = "elements_sections_microscopes_basic_microscope_defined_by_lengths_rays.npy"
	if not os.path.exists(filename):
		print("ERROR: test_basic_microscope_reload_sea requires test_basic_microscope_defined_by_lengths to run first")
		assert 1==0
	r1_old = np.load(filename)
	assert np.sqrt(np.sum((r1-r1_old)**2)) < .0001 # serves as a "hash" of sorts to ensure we're getting the same rays out
#test_basic_microscope_reload_sea()

def test_at_z():
	elements = [ Source(size=(1,1),np_xy=(3,3),angle=(0,0),na_xy=(1,1)), Lens(focal_length=1,position=1), Lens(focal_length=1,position=3) ]
	section1 = MicroscopeSection(elements = elements)
	elements = [ Lens(focal_length=2,position=1), Drift(length=2) ]
	section2 = MicroscopeSection(elements = elements)
	microscope = Microscope(sections = [ section1,section2 ])
	#microscope.show()
	r1 = microscope.propagate_ray()
	rr = r1.convert_to_rotating_reference_frame()
	#print(repr(microscope))
	#print(r1)
	#print(r1.rays.shape,r1.R.shape,np.asarray(r1.z).shape)
	assert rr.at_z(0.5).x[-1] == 1 # parallel beam, sliced pre-lens, last ray's x position should be starting source size
	assert rr.at_z(1.5).x[-1] == 0.5 # 0.5 post-first-lens, focal length of 1, we should have come in by 0.5
	assert rr.at_z(3.5).x[-1] == -1	# post-second-lens, should be parallel again
	assert rr.at_z(4.5).x[-1] == -0.75	# post-last-lens, coming in more gradually again
	assert rr.at_z(4.5).xt[-1] == 0.5
#test_at_z()

def test_element_insertion_microscope():
	filename = "elements_sections_microscopes_basic_microscope_defined_by_lengths.sea"
	if not os.path.exists(filename):
		print("ERROR: test_insertion_microscope requires test_basic_microscope_defined_by_lengths to run first")
		assert 1==0
	microscope = load_microscope(filename)
	#lengths_0 = [ s.length for s in microscope.sections ]
	#section.insert(1,Lens(name="inserted by index at 1",strength=0))
	microscope.insert(1.25,Lens(name="inserted by position at 1.25",strength=0))
	#lengths_1 = [ s.length for s in microscope.sections ]
	r1 = microscope.propagate_ray()
	r1 = r1.convert_to_rotating_reference_frame()[-1,:,:] # 20260723: updated to default to rotate, so we need to convert to match previous rotating-reference-frame saved rays
	filename = "elements_sections_microscopes_basic_microscope_defined_by_lengths_rays.npy"
	if not os.path.exists(filename):
		print("ERROR: test_insertion_microscope requires test_basic_microscope_defined_by_lengths to run first")
		assert 1==0
	r1_old = np.load(filename)[-1,:,:]
	assert np.sqrt(np.sum((r1-r1_old)**2)) < .0001 # serves as a "hash" of sorts to ensure we're getting the same rays out
	#print(repr(microscope))
	# insertion at zero should NOT go before the zero-length source! it should only go into the first Drift.
	microscope.insert(0.,Drift(name="inserted by position at 0.0",length=1.))
	r1 = microscope.propagate_ray()
	r1 = r1.convert_to_rotating_reference_frame()[-1,:,:] # 20260723: updated to default to rotate, so we need to convert to match previous rotating-reference-frame saved rays
	assert np.sqrt(np.sum((r1-r1_old)**2)) < .0001
	# breaking up a drift, with zero left-over at the end...
	microscope.insert(.5,Drift(name="inserted by position at 0.5",length=.5))
	r1 = microscope.propagate_ray()
	r1 = r1.convert_to_rotating_reference_frame()[-1,:,:] # 20260723: updated to default to rotate, so we need to convert to match previous rotating-reference-frame saved rays
	assert np.sqrt(np.sum((r1-r1_old)**2)) < .0001
	# drift *replacement* by inserting one of the same-position same-length
	microscope.insert(.5,Drift(name="new inserted by position at 0.5",length=.5))
	r1 = microscope.propagate_ray()
	r1 = r1.convert_to_rotating_reference_frame()[-1,:,:] # 20260723: updated to default to rotate, so we need to convert to match previous rotating-reference-frame saved rays
	assert np.sqrt(np.sum((r1-r1_old)**2)) < .0001
	#print(repr(microscope))
	#lengths_2 = [ s.length for s in microscope.sections ]
	#print(lengths_0,lengths_1,lengths_2)
	#microscope.show()

def test_section_insertion_microscope():
	ele1 = [ Source(size=(1,1),np_xy=(3,3),angle=(1,1),na_xy=(3,3)),
			 Lens(strength=1,length=.1,position=1), Lens(strength=1,length=.1,position=3) ] # lens at 1 and 3
	ele2 = [ Lens(strength=1,length=.1,position=1) , Drift(name="CCD",length=1,position=3) ] # when placed at 4, yields a lens at 5 7
	sec1 = MicroscopeSection( elements = ele1, name="sec1" )
	sec2 = MicroscopeSection( elements = ele2, position=4, name="sec2" )
	microscope = Microscope(sections = [ sec1, sec2 ])
	#microscope.show()
	print("OLD") ; print(repr(microscope))
	r1 = microscope.propagate_ray()
	r1 = r1.convert_to_rotating_reference_frame() # 20260723: updated to default to rotate, so we need to convert to match previous rotating-reference-frame saved rays
	filename = "elements_sections_microscopes_section_insertion_microscope_rays.npy"
	if not os.path.exists(filename):
		np.save(filename,r1)

	inserted_strength = 0
	ele3 = [ Lens(strength=inserted_strength,length=.1,position=0,name="added 1") ,
				Lens(strength=inserted_strength,length=.1,position=.25,name="added 2") ,
					Lens(strength=inserted_strength,length=.1,position=0.5,name="added 3") ]
	sec3 = MicroscopeSection( elements = ele3, name="newsec" )
	microscope.insert(2.0,sec3)

	print("NEW") ; print(microscope.tabulate(columns=["name","length","position"]))
	#microscope.show()
	r1 = microscope.propagate_ray()
	r1 = r1.convert_to_rotating_reference_frame() # 20260723: updated to default to rotate, so we need to convert to match previous rotating-reference-frame saved rays

	r1_old = np.load(filename)

	#print(np.sqrt(np.sum((r1[-1]-r1_old[-1])**2)))
	assert np.sqrt(np.sum((r1[-1]-r1_old[-1])**2)) < .0001 # serves as a "hash" of sorts to ensure we're getting the same rays out

#test_section_insertion_microscope()

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
	r1 = r1.convert_to_rotating_reference_frame() # 20260723: updated to default to rotate, so we need to convert to match previous rotating-reference-frame saved rays
	#section.show()
	# CROP BY INDEX, REPLACE SOURCE
	section = section[1:]
	section.insert(0, Source(size=(1,1),np_xy=(3,3),angle=(1,1),na_xy=(3,3)) )
	r2 = section.propagate_ray()
	r2 = r2.convert_to_rotating_reference_frame().rays # 20260723: updated to default to rotate, so we need to convert to match previous rotating-reference-frame saved rays
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
	def fresh_section():
		elements = [ Source(name="S") , Drift(length=1), Lens(name="L1",strength=3,length=.1), Drift(length=1.4), Lens(name="L2",strength=5,length=.1), Drift(length=2.4) , Lens(name="L3",strength=1,length=.1) ]
		return MicroscopeSection(elements=elements)

	section = fresh_section()
	# user should *not* be allowed to update the position of element, since it requires updating surrounding elements!
	with pytest.raises(AttributeError):
		section["L2"].position += 1

	check_lengths(section)
	#section.show(title="original, L1,L2,L3 @ 1,2.5,5")

	print("ORIGINAL\n",repr(section))
	# user should *instead* use the move function to move an element (move function likely needs to be on the Section, not the Element, since the Element doesn't know it's parents??)
	section.move_element("L2",dz=1)
	check_lengths(section)
	#section.show(title="L2 dz +1, L1,L2,L3 @ 1,3.5,5")

	section.move_element("L1",dz=-.5)
	check_lengths(section)
	#section.show(title="L1 dz -0.5, L1,L2,L3 @ 0.5,3.5,5")

	# make sure -z move works for last element in section (pads with Drift)
	section.move_element("L3",dz=-.5)
	print(repr(section))
	check_lengths(section)
	#section.show(title="L3 dz -0.5, L1,L2,L3 @ 0.5,3.5,4.5")

	# make sure +z move works for first element in section (pads with Drift)
	section.move_element("S",dz=.25)
	print("L2+1,L1-.5,L3-.5\n",repr(section))
	check_lengths(section)

	section = fresh_section()
	# make sure +z move works for last element in section (conditionally! this changes the sections length!)
	section.move_element("L3",dz=+.5,allow_unsafe=True)
	print("L3+.5\n",repr(section))
	check_lengths(section)
	assert section.length == sum([e.length for e in section.elements])

def test_old_json_compatibility():
	# resolve the fixture beside this file so the test passes from any cwd
	load_microscope(os.path.join(os.path.dirname(os.path.abspath(__file__)),
								 "backwards_compatibility_sanity_check"))

#test_element_move()
#test_element_insertion_microscope()

def test_diffraction_rays():
	from pySEA.rayTEM import diffraction_bundles_at_z
	#np_xy=(3,3) ; na_xy=(3,3)
	np_xy=(5,7) ; na_xy=(3,9)
	elements = [ Source(name="S",size=(2,2),np_xy=np_xy,na_xy=na_xy) , Lens(name="L1",strength=3,length=.1,position=1), Lens(name="L2",strength=5,length=.1,position=2.8), Lens(name="L3",strength=1,length=.1,position=5.2) ]
	microscope = MicroscopeSection(elements=elements)
	microscope.propagate_ray()
	ret = diffraction_bundles_at_z(5,microscope.rays)
	microscope.show(title=str(ret))
test_diffraction_rays()

def test_rotated_quadrupole():
	"""A rotated quadrupole couples the planes; pi/2 swaps them exactly."""
	from pySEA.rayTEM.elements import Quadrapole as Q
	P = Q(strength=2.0).focal_powers[0]
	# 45 degrees: the classic rotated-stigmator kick, dxt = -P*y, dyt = -P*x
	M = np.asarray(Q(strength=2.0, rotation=np.pi / 4).transfer_matrix())
	r = np.zeros(6) ; r[0] = 1.0
	out = M @ r
	assert abs(out[1]) < 1e-12 and abs(out[3] + P) < 1e-12
	# rolling by pi/2 is the same as flipping the strength sign
	assert np.allclose(Q(strength=2.0, rotation=np.pi / 2).transfer_matrix(),
					   Q(strength=-2.0).transfer_matrix(), atol=1e-12)
	# a thick rotated body stays symplectic (unit determinant)
	Mt = np.asarray(Q(strength=30.0, length=0.02, rotation=0.3).transfer_matrix())
	assert abs(np.linalg.det(Mt[:4, :4]) - 1) < 1e-9
	# per-axis machinery must refuse rather than silently answer wrong
	with pytest.raises(NotImplementedError):
		Q(strength=30.0, length=0.02, rotation=0.3).transfer_block()
	# the rotation survives a .sea round trip
	sec = MicroscopeSection(name="S", elements=[
		Source(voltage=200, size=(2e-6, 2e-6), np_xy=(3, 3),
			   angle=(1e-4, 1e-4), na_xy=(3, 3)),
		Drift(length=0.05),
		Quadrapole(name="SQ", strength=2.0, rotation=np.pi / 4),
		Drift(length=0.05)])
	m = Microscope(sections=[sec])
	m.to_sea("t_rot.sea")
	back = load_microscope("t_rot.sea")
	os.remove("t_rot.sea")
	assert back["SQ"].rotation == pytest.approx(np.pi / 4)
	assert np.allclose(back.propagate_ray(), m.propagate_ray())


def test_section_level_aberrations():
	"""Aberrations declared on a section act, suspend, and round-trip."""
	def build(ab):
		sec = MicroscopeSection(name="S", elements=[
			Source(voltage=200, size=(2e-6, 2e-6), np_xy=(3, 3),
				   angle=(1e-4, 1e-4), na_xy=(3, 3)),
			Drift(length=0.05),
			Lens(name="L", strength=np.sqrt(1 / 0.02)),
			Drift(length=0.03)], aberrations=ab)
		return Microscope(sections=[sec]), sec
	m, sec = build({'C30': 1e-4})
	assert sec.focal_power == pytest.approx(50.0)		# the pupil scale
	r_ab = np.array(m.propagate_ray()).copy()
	r_id = np.array(m.propagate_ray(apply_aberrations=False)).copy()
	assert np.abs(r_ab[-1, :, 1] - r_id[-1, :, 1]).max() > 0
	assert sec.aberrations is not None					# suspension restored them
	# the screen is transient: same number of logged planes as the ideal run
	assert r_ab.shape == r_id.shape
	m.to_sea("t_secab.sea")
	back = load_microscope("t_secab.sea")
	os.remove("t_secab.sea")
	assert back.sections[0].aberrations
	assert np.allclose(back.propagate_ray(), r_ab)


def test_aberration_screen_element():
	"""A stand-alone AberrationScreen kicks rays and is transparent when idle."""
	from pySEA.rayTEM.elements import AberrationScreen
	def build(**kw):
		return Microscope(sections=[MicroscopeSection(name="S", elements=[
			Source(voltage=200, size=(2e-6, 2e-6), np_xy=(3, 3),
				   angle=(1e-4, 1e-4), na_xy=(3, 3)),
			Drift(length=0.05),
			AberrationScreen(name="plate", **kw),
			Drift(length=0.03)])])
	act = build(aberrations={'C30': 1e-4}, pupil_power=50.0)
	idle = build(aberrations={'C30': 1e-4}, pupil_power=0.0)
	r_act = np.array(act.propagate_ray())
	r_idle = np.array(idle.propagate_ray())
	r_ref = np.array(act.propagate_ray(apply_aberrations=False))
	assert np.abs(r_act - r_ref).max() > 0
	assert np.allclose(r_idle, r_ref)					# zero pupil power = transparent
	act.to_sea("t_plate.sea")
	back = load_microscope("t_plate.sea")
	os.remove("t_plate.sea")
	assert back["plate"].pupil_power == pytest.approx(50.0)
	assert np.allclose(back.propagate_ray(), r_act)


def test_microscope_index_raises():
	"""Microscope.index raises KeyError for unknown names instead of returning None."""
	m = Microscope(sections=[MicroscopeSection(name="S", elements=[
		Source(voltage=200), Drift(length=0.05), Lens(name="L1", strength=5.0)])])
	assert m.index("S") == 0
	assert m.index("L1") == (0, 2)
	with pytest.raises(KeyError):
		m.index("definitely-not-here")


def test_thick_lens_efl_vs_bfd_split():
	"""The three focal quantities and their matrix definitions.

	focal_power = -C = K*sin(KL) (the equivalent power, the pupil-angle
	scale); focal_length = 1/focal_power (the EFL, principal-plane
	referenced); back_focal_distance = -A/C = 1/(K*tan(KL)) (signed, exit
	face to BFP). focal_power and focal_length are reciprocals;
	back_focal_distance is NOT (their product is A = cos(KL)).
	"""

	lens = Lens(strength=.1, length=10,name="L1")
	section = MicroscopeSection(elements=[Source(angle=(0,0),na_xy=(2,2)),Drift(length=100),lens,Drift(length=100)])
	#section.show(title="f @ "+str(section["L1"].position+section["L1"].principal_distance+section["L1"].focal_length))

	K, L = 129.80, 0.010
	lens = Lens(strength=K, length=L,name="L1")
	section = MicroscopeSection(elements=[Source(angle=(0,0),na_xy=(2,2)),Drift(length=1),lens,Drift(length=1)])
	#section.show(title="f @ "+str(section["L1"].position+section["L1"].focal_length))
	# 1. matrix definitions
	assert lens.focal_power == pytest.approx(K * np.sin(K * L))
	assert lens.focal_length == pytest.approx(1 / lens.focal_power)
	assert lens.back_focal_distance == pytest.approx(1 / (K * np.tan(K * L)))
	# 2. thick-lens relationship: BFD = A * EFL
	assert lens.back_focal_distance == pytest.approx(np.cos(K * L) * lens.focal_length)
	assert lens.focal_power * lens.back_focal_distance == pytest.approx(np.cos(K * L))
	# 3. the traced crossing angle IS focal_power * h, not h/BFD
	#    (Larmor-safe via hypot)
	sec = MicroscopeSection(name="S", elements=[
		Lens(name="OL", strength=K, length=L), Drift(length=0.02)])
	m = Microscope(sections=[sec])
	h = 5e-5
	r0 = np.zeros((1, 6)); r0[0, 0] = h
	rays = np.asarray(m.propagate_ray(r0))
	alpha = np.hypot(rays[-1, 0, 1], rays[-1, 0, 3])
	assert alpha == pytest.approx(lens.focal_power * h, rel=1e-9)
	assert abs(alpha - h / lens.back_focal_distance) > 0.1 * alpha
	# 4. a drift of exactly BFD reaches a real BFP: accumulated A entry = 0
	M = np.matmul(np.asarray([[1.0, lens.back_focal_distance], [0.0, 1.0]]),
				  np.asarray(lens.transfer_block()))
	assert abs(M[0, 0]) < 1e-12
	# 5. thin limit: EFL and BFD converge as KL -> 0 (gap ~ (KL)^2/2), and a
	#    thin lens is one number all three ways
	tiny = Lens(strength=K, length=1e-5)
	gap = 1 - tiny.back_focal_distance / tiny.focal_length
	assert gap == pytest.approx((K * 1e-5) ** 2 / 2, rel=1e-3)
	thin = Lens(strength=np.sqrt(1 / 0.02))
	assert thin.focal_power == pytest.approx(1 / thin.focal_length)
	assert thin.back_focal_distance == pytest.approx(thin.focal_length)


def test_strong_lens_virtual_bfp_vs_internal_crossover():
	"""Past KL = pi/2 the BFD goes virtual; the real crossover is in-body.

	The parallel bundle physically crosses inside the field at
	dz = pi/(2K), while the complete exit matrix extrapolates backward to a
	virtual output-space BFP: back_focal_distance < 0. The two are
	different locations and neither substitutes for the other.
	"""
	K, L = 100.0, 0.028						# KL = 2.8 > pi/2
	lens = Lens(strength=K, length=L)
	assert lens.back_focal_distance < 0		# virtual BFP
	assert lens.back_focal_distance == pytest.approx(1 / (K * np.tan(K * L)))
	# the real crossover: the body's own partial-length A entry hits zero
	dz_cross = np.pi / (2 * K)
	assert abs(np.asarray(lens.transfer_block(dz=dz_cross))[0, 0]) < 1e-12
	assert dz_cross < L						# genuinely inside the body
	# and it is NOT where the virtual BFP extrapolates to
	assert abs((L + lens.back_focal_distance) - dz_cross) > 1e-3
	# focal_power stays the reciprocal of focal_length regardless
	assert lens.focal_length == pytest.approx(1 / lens.focal_power)


def test_focal_properties_round_trip():
	"""A thin lens defined by focal_length keeps all three focal numbers
	through a .sea round trip (the stored _focal_length re-seeds via the
	constructor kwarg and the recorded __dict__ wins verbatim)."""
	sec = MicroscopeSection(name="S", elements=[
		Source(voltage=200, size=(2e-6, 2e-6), np_xy=(3, 3),
			   angle=(1e-4, 1e-4), na_xy=(3, 3)),
		Drift(length=0.05),
		Lens(name="FL", focal_length=0.03, length=0),
		Drift(length=0.05)])
	m = Microscope(sections=[sec])
	m.to_sea("t_focal_rt.sea")
	back = load_microscope("t_focal_rt.sea")
	os.remove("t_focal_rt.sea")
	for prop in ("focal_length", "focal_power", "back_focal_distance"):
		assert getattr(back["FL"], prop) == pytest.approx(getattr(m["FL"], prop))
	assert back["FL"].focal_length == pytest.approx(0.03)


def test_aperture_masks_rays():
	"""An aperture is a true mask: blocked rays carry I = 0 onward,
	survivors pass unattenuated with their geometry untouched, masks
	compose across multiple apertures, and the smooth continuum estimate
	transmitted_fraction stays available for fitting."""
	sec = MicroscopeSection(name="S", elements=[
		Source(voltage=200, size=(10e-6, 10e-6), np_xy=(9, 9),
			   angle=(0.0, 0.0), na_xy=(1, 1), beam_current=1e-9),
		Drift(length=0.01),
		Aperture(name="A1", radius=6e-6),
		Drift(length=0.01),
		Aperture(name="A2", radius=3e-6),
		Drift(length=0.01)])
	m = Microscope(sections=[sec])
	rays = np.asarray(m.propagate_ray())
	I = np.asarray(m.I)
	r_at = np.hypot(rays[0, :, 0], rays[0, :, 2])	# parallel fan: radii constant
	# geometry is untouched everywhere (drifts aside, transverse coords const)
	assert np.allclose(rays[-1, :, 0], rays[0, :, 0])
	assert np.allclose(rays[-1, :, 2], rays[0, :, 2])
	# after A1: outside 6 um dead, inside alive and unattenuated
	# the drift exit and the aperture plane share a z; the aperture's own
	# (post-mask) plane is the LAST one logged at that z
	i_a1 = int(np.where(np.abs(rays[:, 0, 4] - m.get_element_position("A1")) < 1e-12)[0][-1])
	assert np.all(I[i_a1][r_at > 6e-6] == 0)
	assert np.allclose(I[i_a1][r_at <= 6e-6], I[0][r_at <= 6e-6])
	# after A2: the SECOND aperture masks further (composition -- the old
	# rescale could not do this, per the design comment in elements.py)
	assert np.all(I[-1][r_at > 3e-6] == 0)
	assert np.allclose(I[-1][r_at <= 3e-6], I[0][r_at <= 3e-6])
	# current bookkeeping: sum(I) is the surviving fraction of the stated 1 nA
	frac = float((r_at <= 3e-6).mean())
	assert np.isclose(float(I[-1].sum()), frac * 1e-9, rtol=1e-12)
	assert np.isclose(m.beam_current, frac * 1e-9, rtol=1e-12)
	# the smooth fitting estimate exists and brackets sensibly
	tf = m["A2"].transmitted_fraction(rays[i_a1])
	assert 0 < tf <= 1
	assert m["A1"].transmitted_fraction(rays[0] * 0) == 1.0


def test_findplanes_ignores_dead_rays():
	"""Plane detection only trusts rays that still carry intensity.

	With spherical aberration, parallel rays at different heights cross at
	different z (the focal surface) -- and an aperture selects which zone
	carries beam. The detected diffraction plane must follow the LIVE zone:
	ghost (masked, I = 0) tracers reporting the cut zone's crossing was the
	bug. Ideal optics are insensitive (all parallel rays share one crossing),
	and with nothing masked the tracer pair is the old first-two, bit for
	bit. When the aperture kills every candidate, no plane is reported at
	all -- there is no beam to have one.
	"""
	def build(radius, c30):
		from pySEA.rayTEM.aberrations import Aberrations
		lens = Lens(name="L", strength=np.sqrt(1 / 0.02))
		if c30:
			lens.aberrations = Aberrations({'C30': c30})
		sec = MicroscopeSection(name="S", elements=[
			Source(voltage=200, size=(40e-6, 40e-6), np_xy=(5, 5),
				   angle=(0.0, 0.0), na_xy=(1, 1)),
			Drift(length=0.01),
			Aperture(name="A", radius=radius),
			Drift(length=0.01),
			lens,
			Drift(length=0.03)])
		m = Microscope(sections=[sec])
		m.propagate_ray()
		return findPlanes(m.rays, axis="x")["x"]["diff"]["z"]
	# ideal lens: cutting the outer zone must not move the plane
	z_open  = build(radius=1.0,   c30=0.0)
	z_cut   = build(radius=25e-6, c30=0.0)
	assert len(z_open) == 1 and len(z_cut) == 1
	assert np.isclose(z_cut[0], z_open[0], atol=1e-9)
	# aberrated lens: outer rays cross EARLIER (spherical), so masking them
	# must move the detected plane DOWNSTREAM to the live inner zone
	za_open = build(radius=1.0,   c30=2.0)
	za_cut  = build(radius=25e-6, c30=2.0)
	assert len(za_open) == 1 and len(za_cut) == 1
	assert za_cut[0] > za_open[0] + 1e-4
	# everything masked: no beam, no plane
	assert build(radius=1e-9, c30=0.0) == []


def test_covariance_aberration_closure():
	"""Aberrations enter the moments mode analytically (Gaussian closure).

	The linear terms (C10, aligned C12) fold into the matrix exactly; the
	cubic spherical kick's cross- and self-moments close on Sigma by
	Isserlis' theorem. Verified against Monte-Carlo statistics of the exact
	per-ray kick, against the exact power-shift equivalence for C10, and
	bit-for-bit idle behavior without aberrations.
	"""
	rng = np.random.default_rng(7)
	ix, ixt, iy, iyt = (columnByName(k) for k in ("x", "xt", "y", "yt"))
	f, C30, N = 0.02, 3e3, 300000
	sx, st = 5e-6, 2e-6
	mu0 = np.zeros(6)
	Sig0 = np.zeros((6, 6))
	for i, s in ((ix, sx), (ixt, st), (iy, sx), (iyt, st)):
		Sig0[i, i] = s * s
	# Monte-Carlo reference: the exact per-ray kick on a Gaussian ensemble
	lens = Lens(name="L", strength=np.sqrt(1 / f), aberrations={'C30': C30})
	r0 = np.zeros((N, 6))
	for i, s in ((ix, sx), (ixt, st), (iy, sx), (iyt, st)):
		r0[:, i] = rng.normal(0, s, N)
	r1 = np.asarray(lens.propagate_ray(r0.copy()))
	Sig_MC = np.cov(r1[:, [ix, ixt, iy, iyt]].T)
	_, Sig1 = lens.propagate_moments(mu0, Sig0)
	An = Sig1[np.ix_([ix, ixt, iy, iyt], [ix, ixt, iy, iyt])]
	# the aberration-inflated entries agree with sampled statistics (~MC noise)
	assert np.isclose(An[1, 1], Sig_MC[1, 1], rtol=2e-2)
	assert np.isclose(An[0, 1], Sig_MC[0, 1], rtol=2e-2)
	assert np.isclose(An[3, 3], Sig_MC[3, 3], rtol=2e-2)
	# and the aberration genuinely inflates them vs the ideal lens
	ideal = Lens(strength=np.sqrt(1 / f))
	_, SigI = ideal.propagate_moments(mu0, Sig0)
	assert An[1, 1] > 1.05 * SigI[ixt, ixt]
	# C10 is a pure power change: the closure must equal the shifted lens
	P = ideal.focal_power
	c10 = 1e-2
	ab10 = Lens(strength=np.sqrt(1 / f), aberrations={'C10': c10})
	_, SigA = ab10.propagate_moments(mu0, Sig0)
	M = np.asarray(ideal.transfer_matrix())
	M[ixt, ix] -= c10 * P**2
	M[iyt, iy] -= c10 * P**2
	assert np.allclose(SigA, M @ Sig0 @ M.T, rtol=1e-12)
	# idle path bit-for-bit
	M0 = np.asarray(ideal.transfer_matrix())
	assert np.allclose(SigI, M0 @ Sig0 @ M0.T)


def test_frame_focal_surface():
	"""focal_surface(method='frame'): the aberrated surface in closed form.

	Zone-modified ABCD reproduces the traced surface on a thin lens to
	machine precision, both matching the closed form -C30*alpha^2; an ideal
	column gives an exactly flat surface at the paraxial plane.
	"""
	f, C30, ALPHA = 0.02, 2.0, 5e-3
	def build(c30):
		from pySEA.rayTEM.aberrations import Aberrations
		lens = Lens(name="L", strength=np.sqrt(1 / f))
		if c30:
			lens.aberrations = Aberrations({'C30': c30})
		sec = MicroscopeSection(name="S", elements=[
			Source(voltage=200, size=(1e-6, 1e-6), np_xy=(3, 3),
				   angle=(1e-6, 1e-6), na_xy=(3, 3)),
			Drift(length=0.01), lens, Drift(length=0.05)])
		return Microscope(sections=[sec])
	m = build(C30)
	sr = m.focal_surface(family="diff", aperture=ALPHA * f, radii=8, azimuths=4)
	sf = m.focal_surface(family="diff", aperture=ALPHA * f, radii=8, method="frame")
	assert np.isclose(sf["fit"]["c20"], sr["fit"]["c20"], rtol=1e-9)
	assert np.isclose(sf["sag"], sr["sag"], rtol=1e-9)
	assert np.isclose(sf["fit"]["c20"], -C30 * ALPHA**2, rtol=5e-3)
	s0 = build(0.0).focal_surface(family="diff", aperture=ALPHA * f, radii=8,
								  method="frame")
	assert s0["sag"] == 0.0
	assert np.allclose(s0["z"], s0["z_paraxial"], atol=1e-12)
