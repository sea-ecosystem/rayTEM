from rayTEM.elements import Lens,Drift,Quadrapole,Source,Aperture,columnByName,fix_ray_dims
from rayTEM.assemblies import MicroscopeSection
from rayTEM.postprocessing import findPlanes,fitForCrossover,plotRays
import numpy as np
import sys
sys.path.insert(1,"../niceplot")
from niceplot import *
from nicecontour import *


#print(s.rays())

elements=[ 	Drift(length=1),
			Lens(strength=1), 
			Drift(length=3), 
			Lens(strength=1) , 
			Drift(length=10) ]

section=MicroscopeSection(elements=elements)

r0=np.asarray( [[1,1,0,0],[1,1,-1,-1]] )
r0=fix_ray_dims(r0,["x","y","xt","yt"])



#r1=section.propagate_ray(r0)
#plotRays(r1)

fitForCrossover(r0,section,target=3,guesses={"1_strength":2},plane="image")