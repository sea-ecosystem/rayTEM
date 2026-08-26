import sys,os
sys.path.insert(1,"../../../../")
from pySEA.rayTEM import load_microscope as mic_load
from pySEA.rayTEM import closest_plane,Source
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import multiprocessing,time
from scipy.optimize import minimize
sys.path.insert(1,"../../../../../../../niceplot/gitHelper_main")
from niceplot import *

microscope = mic_load("macstem_calibratedPL")["sample":]
microscope.insert(0,Source(name="gun",size=(1e-4,1e-4),angle=(1e-4,1e-4),na_xy=(3,1),np_xy=(3,1)))

# Two manually-found states where wobbling majorOL yields no change in
settings = [[123.45,0,0,234.56],[345.67,456.78,567.89,678.9]]

# start by measuring the OL's image plane, so we can preserve it
for n in range(4):
	microscope["PL"+str(n+1)].strength = 0
z_PL1 = microscope.get_element_position("PL1")
z_CCD = microscope.get_element_position("CCD")
z_image = closest_plane(microscope,z_PL1,"image")['z']

foundCs={}
def dz(z,show=False):
	global foundCs
	# move OL, adjust OL calibration, to maintain dixed image plane position
	microscope["objective"].move("OL2",z=z)
	for n in range(4):
		microscope["PL"+str(n+1)].strength = 0
	def dzz(C):
		microscope["OL2"].calibration = C[0]
		return ( closest_plane(microscope,z_PL1,"image")['z']-z_image )**2
	x0 = minimize(dzz,x0=microscope["OL2"].calibration)
	foundCs[z]=x0['x']
	# then, compare diffraction plane positions
	zs = []
	for PLs in settings:
		for n,v in enumerate(PLs):
			microscope["PL"+str(n+1)].strength = v/1000
		z_diff = closest_plane(microscope,z_CCD,"diff")
		zs.append(z_diff['z'])
		if show:
			microscope.show()
	return (zs[0]-zs[1])**2 #+ (z_image-z_im)**2

z_OL2 = microscope.get_element_position("OL2") ; C = microscope["OL2"].calibration
#guess = (z_OL2*.5)
#bounds = [[0,z_OL2]]
#x0 = minimize(dz,x0=guess,bounds=bounds)#,args=(True))
#print(x0)
#dz(x0['x'],show=True)

er = [] ; zs = np.linspace(.1,z_OL2*.99,100)
for z in reversed(zs):
	er.append(dz(z))
er = list(reversed(er))
i = np.argmin(er)
z = zs[i] ; C = foundCs[z]
#plot([zs],[list(reversed(er))])
microscope["OL2"].calibration = C
microscope["objective"].move("OL2",z=z)

dz(z,show=True)

microscope.save("macstem_calibratedOL")
