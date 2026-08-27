import sys
sys.path.insert(1,"../")
from pySEA.rayTEM.elements import Lens,Drift,Quadrapole,fix_ray_dims
from pySEA.rayTEM.assemblies import MicroscopeSection
from pySEA.rayTEM.postprocessing import plot2D,plot3D,plotSliceSeries
import numpy as np
import os

os.makedirs("figs", exist_ok=True)		# plots save here; running from any cwd works

# Construct our microscope section: drifts and lenses
elements=[ 	Drift(length=1),
			Lens(strength=3.5,length=.1), 
			Drift(length=1),Quadrapole(strength=.5,length=.1),Drift(length=1.9),
			#Drift(length=3), 
			Lens(strength=3,length=.1), 
			Drift(length=1),Quadrapole(strength=-.2,length=.1),Drift(length=3.9) ]
			#Drift(length=5) ]

section=MicroscopeSection(elements=elements)

# construct our list of rays: starting positions and angles. a minimum of two (is required for auto-detection of image/diffraction planes (one normal, one at an angle, both from the same point) 
r0=np.asarray( [[1,1,0,0],[1,1,-1,-1]] )
r0=fix_ray_dims(r0,["x","y","xt","yt"])

# propagate and 2D plot
r1=section.propagate_ray(r0)
plot2D(r1,section.R,filename="figs/01_basicRays_2D.png")

# alternate list of rays: a whole series of positions and angles should make visualizing the image and diffraction planes easier
r0=[]
for x in np.arange(-2,3):
	for y in np.arange(-2,3):
		for xt in [-1,0,1]:
			for yt in [-1,0,1]:
				r0.append([x,y,xt,yt])
r0=fix_ray_dims(np.asarray(r0),["x","y","xt","yt"])

# propoagate and 3D plot
r1=section.propagate_ray(r0)
plot3D(r1,section.R)#,filename="figs/01_basicRays_3D.png",elev=88,azi=10,roll=104)

plotSliceSeries(r1,20,20,filename="plotSliceSeries.png")