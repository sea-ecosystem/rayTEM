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
from pySEA.rayTEM import fix_ray_dims,plot2D,plot3D,findPlanes,columnByName,load_microscope,load_section,convert_to_rotating_reference_frame
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
	rr = convert_to_rotating_reference_frame(r1) # 20260723: updated to default to rotate, so we need to convert to match previous rotating-reference-frame saved rays
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
	r1 = convert_to_rotating_reference_frame(r1) # 20260723: updated to default to rotate, so we need to convert to match previous rotating-reference-frame saved rays
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
	rr = convert_to_rotating_reference_frame(r1) # 20260723: updated to default to rotate, so we need to convert to match previous rotating-reference-frame saved rays
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
	r1 = convert_to_rotating_reference_frame(r1) # 20260723: updated to default to rotate, so we need to convert to match previous rotating-reference-frame saved rays
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
	r1 = convert_to_rotating_reference_frame(r1) # 20260723: updated to default to rotate, so we need to convert to match previous rotating-reference-frame saved rays
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
	r1 = convert_to_rotating_reference_frame(r1) # 20260723: updated to default to rotate, so we need to convert to match previous rotating-reference-frame saved rays
	filename = "elements_sections_microscopes_basic_microscope_defined_by_lengths_rays.npy"
	if not os.path.exists(filename):
		print("ERROR: test_basic_microscope_reload_sea requires test_basic_microscope_defined_by_lengths to run first")
		assert 1==0
	r1_old = np.load(filename)
	assert np.sqrt(np.sum((r1-r1_old)**2)) < .0001 # serves as a "hash" of sorts to ensure we're getting the same rays out
#test_basic_microscope_reload_sea()

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
	r1 = convert_to_rotating_reference_frame(r1)[-1,:,:] # 20260723: updated to default to rotate, so we need to convert to match previous rotating-reference-frame saved rays
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
	r1 = convert_to_rotating_reference_frame(r1)[-1,:,:] # 20260723: updated to default to rotate, so we need to convert to match previous rotating-reference-frame saved rays
	assert np.sqrt(np.sum((r1-r1_old)**2)) < .0001
	# breaking up a drift, with zero left-over at the end...
	microscope.insert(.5,Drift(name="inserted by position at 0.5",length=.5))
	r1 = microscope.propagate_ray()
	r1 = convert_to_rotating_reference_frame(r1)[-1,:,:] # 20260723: updated to default to rotate, so we need to convert to match previous rotating-reference-frame saved rays
	assert np.sqrt(np.sum((r1-r1_old)**2)) < .0001
	# drift *replacement* by inserting one of the same-position same-length
	microscope.insert(.5,Drift(name="new inserted by position at 0.5",length=.5))
	r1 = microscope.propagate_ray()
	r1 = convert_to_rotating_reference_frame(r1)[-1,:,:] # 20260723: updated to default to rotate, so we need to convert to match previous rotating-reference-frame saved rays
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
	r1 = convert_to_rotating_reference_frame(r1) # 20260723: updated to default to rotate, so we need to convert to match previous rotating-reference-frame saved rays
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
	r1 = convert_to_rotating_reference_frame(r1) # 20260723: updated to default to rotate, so we need to convert to match previous rotating-reference-frame saved rays

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
	r1 = convert_to_rotating_reference_frame(r1) # 20260723: updated to default to rotate, so we need to convert to match previous rotating-reference-frame saved rays
	#section.show()
	# CROP BY INDEX, REPLACE SOURCE
	section = section[1:]
	section.insert(0, Source(size=(1,1),np_xy=(3,3),angle=(1,1),na_xy=(3,3)) )
	r2 = section.propagate_ray()
	r2 = convert_to_rotating_reference_frame(r2) # 20260723: updated to default to rotate, so we need to convert to match previous rotating-reference-frame saved rays
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
	load_microscope("backwards_compatibility_sanity_check")

#test_element_move()
#test_element_insertion_microscope()


def test_skew_quadrupole():
	"""A rolled quadrupole couples the planes; pi/2 swaps them exactly."""
	from pySEA.rayTEM.elements import Quadrapole as Q
	P = Q(strength=2.0).focal_powers[0]
	# 45 degrees: the classic skew stigmator kick, dxt = -P*y, dyt = -P*x
	M = np.asarray(Q(strength=2.0, skew=np.pi / 4).transfer_matrix())
	r = np.zeros(6) ; r[0] = 1.0
	out = M @ r
	assert abs(out[1]) < 1e-12 and abs(out[3] + P) < 1e-12
	# rolling by pi/2 is the same as flipping the strength sign
	assert np.allclose(Q(strength=2.0, skew=np.pi / 2).transfer_matrix(),
					   Q(strength=-2.0).transfer_matrix(), atol=1e-12)
	# a thick skew body stays symplectic (unit determinant)
	Mt = np.asarray(Q(strength=30.0, length=0.02, skew=0.3).transfer_matrix())
	assert abs(np.linalg.det(Mt[:4, :4]) - 1) < 1e-9
	# per-axis machinery must refuse rather than silently answer wrong
	with pytest.raises(NotImplementedError):
		Q(strength=30.0, length=0.02, skew=0.3).transfer_block()
	# skew survives a .sea round trip
	sec = MicroscopeSection(name="S", elements=[
		Source(voltage=200, size=(2e-6, 2e-6), np_xy=(3, 3),
			   angle=(1e-4, 1e-4), na_xy=(3, 3)),
		Drift(length=0.05),
		Quadrapole(name="SQ", strength=2.0, skew=np.pi / 4),
		Drift(length=0.05)])
	m = Microscope(sections=[sec])
	m.to_sea("t_skew.sea")
	back = load_microscope("t_skew.sea")
	os.remove("t_skew.sea")
	assert back["SQ"].skew == pytest.approx(np.pi / 4)
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
