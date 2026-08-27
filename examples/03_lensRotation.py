import sys
sys.path.insert(1,"../")
from pySEA.rayTEM.elements import Lens,Drift,fix_ray_dims
from pySEA.rayTEM.assemblies import MicroscopeSection
from pySEA.rayTEM.postprocessing import plot2D,plot3D
import numpy as np

# Construct our microscope section: drifts and lenses
elements=[ 	Drift(length=1),
			Lens(strength=2.2,length=.1), 
			Drift(length=4), 
			Lens(strength=2.2,length=.1), 
			Drift(length=3) ]

section=MicroscopeSection(elements=elements)

# construct a list of rays: a whole series of positions, all parallel
r0=[]
for x in np.arange(-3,4):
	for y in np.arange(-3,4):
			r0.append([x,y])
r0=fix_ray_dims(np.asarray(r0),["x","y"])

# propoagate and 3D plot
r1=section.propagate_ray(r0)
plot3D(r1,section.R)#,filename="figs/03_basicRays_3D.png",elev=88,azi=10,roll=104)
