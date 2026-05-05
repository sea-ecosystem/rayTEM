import sys
sys.path.insert(1,"../")
from rayTEM.elements import Lens,Drift,fix_ray_dims,Source
from rayTEM.assemblies import MicroscopeSection
from rayTEM.postprocessing import fitForCrossover,plot2D
import numpy as np

def createFreshSection():
	elements=[ 	
			Source(size=(1,0),np_xy=(3,1),angle=(.5,0),na_xy=(3,1)),
			Drift(length=1),
			Lens(strength=3.8,length=.1), 
			Drift(length=3), 
			Lens(strength=4,length=.1), 
			Drift(length=3) ]
	section=MicroscopeSection(elements=elements)
	return section

#r0=[]
#for i in [1,0,-1]:
#	for th in [0,-.5,.5]:
#		r0.append([i,i,th,th])
#r0=fix_ray_dims(np.asarray(r0),["x","y","xt","yt"])

plot2D( createFreshSection().propagate_ray() )

fitForCrossover(createFreshSection(),targets=[{"plane":"image","z":6,"mag":3}],modifiable={2:"strength",4:"strength"})#,filename="figs/02_basicFitting_01")

fitForCrossover(createFreshSection(),targets=[{"plane":"image","z":6,"mag":3,"strength":"maximize"}],modifiable={1:"strength",3:"strength"})#,filename="figs/02_basicFitting_02")

fitForCrossover(createFreshSection(),targets=[{"plane":"image","z":6,"mag":3,"strength":"minimize"}],modifiable={1:"strength",3:"strength"})#,filename="figs/02_basicFitting_03")

fitForCrossover(createFreshSection(),targets=[{"plane":"image","z":6,"mag":"maximize"}],modifiable={1:"strength",3:"strength"})#,filename="figs/02_basicFitting_04")

fitForCrossover(createFreshSection(),targets=[{"plane":"diff","z":6,"mag":"maximize"}],modifiable={1:"strength",3:"strength"})#,filename="figs/02_basicFitting_05")


# TODOs
#C1 C2 aperture C3 C4 detector
#fit C1 C2 to define what hits aperture
#then fit C3 C4 for what hits detector

#start with all unfixed, when fitting for z=aperture, limit changes to lense before aperture ("temporarily fix C3 C4")
#then only fix those lenses before the aperture.
#second fit then only adjust C3 C4 since C1 C2 have just been fixed

#probably always want to do this C3 C4 fit no matter what the first fit is
#so maybe this second one could be automatically run

#fit should be a class function on the section, e.g. 
#section.fitFor("image&z=3&mag=2,diff&z=7&mag=4")









