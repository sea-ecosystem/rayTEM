# rayTEM
rayTEM is an electron optics simulator for a transmission electron microscope. A simulated microscope is constructed according to the Microscope > MicroscopeSection > Element hierarchy. Each Element has its own transformation matrix, which takes an input matrix of rays (list of rays, defined by their position, angle, energy, and so on), and returns the appropriately-modified rays. 

For example, a simplified aberration-free thin lens applies the following transformation in 2D:
```
| x₂ |  = | 1    0 | | x₁ |
| θ₂ |    | 1/f  1 | | θ₁ |
```
i.e., the position of the ray does not change, but the angle changes according to the position of the ray entering the lens, and the focal length of the lens. 

Element types include (but not limied to): 
Source - initializes rays according to pre-defined criteria (number of rays emitted, at defined spacing in x and y, at varying angles)
Lens - round lens, symetric focusing, beam rotation dependent on lens steength
Quadrupole - asymetric focus/defocus in x/y
Dipole - steering element that applies an angular kick in x or y
Drift - free-space beam propagation
Aperture - crops the beam, which affects the size, shape, and net intensity of the beam

MicroscopeSection objects are defined by a list of Element objects, and Microscope objects are defined by a list of MicroscopeSection objects

# Basic Example:
Sections and Microscopes can be constructed by specifying the positions of elements (e.g. lenses at fixed positions), or by specifying the lengths of elements (e.g. lens, followed by drift, followed by another lens, where the length of the drift determines the position of the second lens):
```
elements = [ Source(size=(1,1),np_xy=(3,3),angle=(1,1),na_xy=(3,3)), Lens(strength=3,length=.1,position=1), Lens(strength=4.5,length=.1,position=1.5) ]
section1 = MicroscopeSection(elements = elements)
elements = [ Lens(strength=5,length=.1), Drift(length=2) ]
section2 = MicroscopeSection(elements = elements,position=2.6)
microscope = Microscope(sections = [ section1,section2 ])
```
rays can be propagated explicitly, for both MicrocopeSections or Microscopes:
```
rays = section1.propagate_ray() # returns an array of shape: n_elements,n_rays,[positions,angles,etc]
rays = microscope.propagate_ray()
```
or propagation can occur automatically while plotting:
```
microscope.show()
```

# Advanced usage:
- advanced building: Elements can be referenced by name, in both the Microscope or MicroscopeSection, Elements can be deleted or inserted. See the examples in the "microscopes" folder for usage, e.g. REMOVED_PRIVATE_INSTRUMENT_TREE/PRIVATE_INSTRUMENT/fine_PLs.py or DQCM.py
- fitting: lens strengths can be fit based on desired image/diffraction plane magnification and position. see REMOVED_PRIVATE_INSTRUMENT_TREE/PRIVATE_INSTRUMENT/course_PLs.py includes an example of this
- sea-eco integration: if sea-eco (https://code.ornl.gov/sea-ecosystem/sea-eco) is installed, the sea file infrastructure can be used, allowing tight integration between data acquisition, analysis, and microscope simulation. rayTEM serves as a plugin for sea-eco in this context. 

