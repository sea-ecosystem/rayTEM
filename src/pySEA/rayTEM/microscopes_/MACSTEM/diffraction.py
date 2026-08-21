import sys
sys.path.insert(1,"../../../../")
from pySEA.rayTEM import Source,Lens,Drift,MicroscopeSection,Microscope,fix_ray_dims,columnByName
import numpy as np

# Basic single-lens section....
elements = [ Drift(length=1), Lens(strength=2,length=0), Drift(length=3) ]
section = MicroscopeSection(elements=elements)


# When a focused probe diffracts through a sample, we get copies of the diverging "cone" at varying angles
# convergence semi-angle alpha = .5, diffraction angle of 4	--> combinations of of +/-4 (+/-0.5)
STEM = np.reshape([-4.5,-4,-3.5,-.5,0,.5,3.5,4,4.5],(9,1))
#r0 = fix_ray_dims(STEM,["xt"])
STEM = np.asarray( sum( [[[x,xt] for x in [-.2,0,.2] ] for xt in [-4.5,-4,-3.5,-.5,0,.5,3.5,4,4.5] ] , [] ) )
r0 = fix_ray_dims(STEM,["x","xt"])


r1 = section.propagate_ray(r0)
section.show(regenerate=False,title="STEM")


# When a wide but perfectly-parallel TEM beam diffracts through a sample, we get copies of the parallel beam at varying angles
TEM = Source(np_xy=(3,1),size=(3,1),na_xy=(3,1),angle=(4,4))
reorder = np.asarray([0,3,6,1,4,7,2,5,8])
r0 = TEM.rays()[reorder,:]

r1 = section.propagate_ray(r0)
section.show(regenerate=False,title="TEM")


# Midgley diffraction is series of *converging* cones. but if our sample plane is at z=0, then it's diverging cones with spatial offsets....
# convergence semi-angle alpha = 0.5, diffraction angle of 4, defocused distance of 1 --> x = +/- l*alpha, xt = +/-4 (+/-0.5)
upstream = np.asarray(
			[[.5,-4.5],[.5,-.5],[.5,3.5],
			[0,-4],[0,0],[0,4],
			[-.5,-3.5],[-.5,.5],[-.5,4.5]]		)
upstream = fix_ray_dims(upstream,["x","xt"])
sec = MicroscopeSection(elements=[ Drift(length=1)])
r1 = sec.propagate_ray(upstream)
sec.show(regenerate=False,title="Midgley upstream")		# UNCOMMENT TO PREVIEW
r0 = r1[-1]						# the rays entering our original sample plane are the exit rays after travelling the defocus distance
r0[:,columnByName("z")]=0

r1 = section.propagate_ray(r0)
section.show(regenerate=False,title="Midgley")


