import sys,json,glob
import numpy as np
from scipy.optimize import minimize,curve_fit,brute
from PIL import Image

from scipy.interpolate import RegularGridInterpolator
from scipy.special import erf
sys.path.insert(1,"../../../../")
from tqdm import tqdm

from pySEA.rayTEM.assemblies import load_microscope as mic_load
from pySEA.rayTEM.elements import Source,columnByName,Drift
from pySEA.rayTEM.postprocessing import measureAtZ,zFromFractional,findPlanes
from pySEA.rayTEM.xmlNion import lookupStrengthsXML,rootControlSettingValue

#sys.path.insert(1,"/media/qwe/Data/Various Code/sea-pearl/TWP20260513MAC2/src")
from pySEA.rayTEM.utilities import findEllipse

sys.path.insert(1,"../../../../../../../niceplot/gitHelper_main")
from niceplot import *
from nicecontour import *

def main():
	#sweep_fitting_1D_individual(preview_only=True)
	#sweep_fitting_1D_individual()
	#sweep_fitting_1D_simultaneous()
	#I_crit_from_2D()
	#minima_from_1D()
	analytical_crossover_fitting()
	#visualize_fitted()
	#DQCM()
	#measureLacyRotation()
	#measureEllipseRotation()
	#compareLacyRotations()
	test_crossovers()


def sweep_fitting_1D_individual(microscope_name="macstem",preview_only=False):
	# LOAD MICROSCOPE (built by builder.py), CROP TO POST-SAMPLE, ADD SOURCE WITH ALMOST-ONLY DIFFRACTION RAYS
	microscope = mic_load(microscope_name)
	microscope = microscope["sample":]
	microscope.insert( 0., Source(size=(2e-4/100,0),np_xy=(3,1),angle=(.0001,0),na_xy=(3,0),name="gun") )

	#style = "OL;A+B*x" ; guesses = [1,0,1]
	#style = "OL;C*x^2" ; guesses = [1,1]
	style = "theta;OL;B*x" ; guesses = [1,1,2]

	def setter(vals):
		if style == "OL;A+B*x":
			microscope["OL2"].calibration = vals[0]				# OL2, affects angles entering PLs
			microscope["PL"+str(i+1)].calibration = vals[1:]	# first two of A + B*x^(1/1) + C*x^(1/2) + ....
		if style == "OL;C*x^2":
			microscope["OL2"].calibration = vals[0]				# OL2, affects angles entering PLs
			microscope["PL"+str(i+1)].calibration = [0,0,vals[1]]	# "C" only, of A + B*x^(1/1) + C*x^(1/2) + ....
		if style == "theta;OL;B*x":
			microscope["gun"].angle=(vals[0],0)
			microscope["OL2"].calibration = vals[1]
			microscope["PL"+str(i+1)].calibration = vals[2]	# "B" only, of A + B*x^(1/1) + C*x^(1/2) + ....
	def getter():
		if style == "OL;A+B*x":
			return [ microscope["OL2"].calibration ] + microscope["PL"+str(i+1)].calibration
		if style == "OL;C*x^2":
			return [ microscope["OL2"].calibration , microscope["PL"+str(i+1)].calibration[2] ]
		if style == "theta;OL;B*x":
			return [ microscope["gun"].angle[0], microscope["OL2"].calibration, microscope["PL"+str(i+1)].calibration ]

	# READ IN DATA FROM CSV (csv generated via sea_pearl/tools/scratch.py as of 2026-05-10, eventually moving to its own script)
	#data = np.loadtxt("linear_sweeps.csv",delimiter=",") # colmuns are PL strength / measured beam diameter
	data = np.load("linear_sweeps.npy")
	strengths = [] ; diameters = [] ; modeled=[] ; calibrations=[] ; labels=['']*8 ; markers=['']*8
	# loop through each PL to load
	for i in range(4):
		# read data from columns in CSV
		strengths.append( data[:,i*4] )
		diameters.append( data[:,i*4+1] )
		labels[i]="PL"+str(i) ; markers[i]=''

	# FITTING OF 1D SWEEPS, ONE AT A TIME
	# loop through each PL individually
	for i in range(4):
		# define an error function for scipy-minimize: given a calibration: loop strengths, propagate rays, measure each diameter
		def dz(vals,ret="dz"):
			# update all model parameters we're fitting for
			setter(vals)
			# zero-out everyone else
			for j in range(4):
				microscope["PL"+str(j+1)].strength = 0
			# loop through all strengths, propagate, measure beam diameter
			ds = []
			for s in strengths[-1]:
				microscope["PL"+str(i+1)].strength = s
				r1 = microscope.propagate_ray()
				x,y,xt,yt,R,I = measureAtZ(microscope["projector"].position+microscope["CCD"].position,rays=r1)
				ds.append(np.sqrt(x**2+y**2))
			if ret=="dz":
				return np.sqrt(np.sum((np.asarray(diameters[-1])-np.absolute(ds))**2))
			return ds
		# optionally, just plot the scope state
		if preview_only:
			microscope["gun"].angle=(.2,0)
			x = [ v if v is not None else 1 for v in getter() ]; print(x)
		else:
			# feed our error function to scipy minimize
			x = minimize(dz,x0=guesses)['x']
		# getting ready to plot the result
		markers[i+4]='-'
		if style == "OL;A+B*x":
			labels[i+4]="OL="+str(np.round(x[0],4))+",K="+str(np.round(x[1],4))+"+"+str(np.round(x[2],4))+"*S"
		if style == "OL;C*x^2":
			labels[i+4]="OL="+str(np.round(x[0],4))+",K="+str(np.round(x[1],4))+"*sqrt(S)"
		if style == "theta;OL;B*x":
			labels[i+4]="theta="+str(np.round(x[0],4))+",OL="+str(np.round(x[1],4))+",K="+str(np.round(x[2],4))+"*S"
		modeled.append( dz(x,ret="full") )
	# PLOTTING
	plot(strengths+strengths, diameters+modeled, xlabel="strength", ylabel="diameter", labels=labels, markers=markers)# filename=style.replace(";","").replace("+","").replace('*',"")+".png")

def sweep_fitting_1D_simultaneous(microscope_name="U100"):
	# LOAD MICROSCOPE (built by builder.py), CROP TO POST-SAMPLE, ADD SOURCE WITH ALMOST-ONLY DIFFRACTION RAYS
	microscope = mic_load(microscope_name)
	microscope = microscope["sample":]
	microscope.insert( 0., Source(size=(2e-4/100,0),np_xy=(3,1),angle=(.0001,0),na_xy=(3,0),name="gun") )

	#style = "D;theta;OL;B*x" ; guesses = [2e-4/100,.0001,1,1]
	#style = "theta;OL;B*x" ; guesses = [.2,1,2]
	style = "theta;OL;B;B;B;B" ; guesses = [.2,1,np.sqrt(.1),np.sqrt(.2),np.sqrt(.3),np.sqrt(.4)]

	# READ IN DATA FROM CSV (csv generated via sea_pearl/tools/scratch.py as of 2026-05-10, eventually moving to its own script)
	data = np.loadtxt("linear_sweeps.csv",delimiter=",") # colmuns are PL strength / measured beam diameter
	strengths = [] ; diameters = [] ; modeled=[] ; calibrations=[] ; labels=['']*8 ; markers=['']*8
	# loop through each PL to load
	for i in range(4):
		# read data from columns in CSV
		strengths.append( data[:,i*2] )
		diameters.append( data[:,i*2+1] )
		labels[i]="PL"+str(i) ; markers[i]=''

	# FITTING OF 1D SWEEPS, ALL TOGETHER NOW
	# defined error function for scipy-minimize will loop through all PLs: given a calibration: loop PLs, loop strengths, propagate rays, measure each diameter, concatenate lists to compare all datasets (PL1 vs diameter, PL2 vs diameter, etc) simultaneously
	diameters = [ list(dia) for dia in diameters ] # required for list concatenation later on
	def dz(vals,ret="dz"):
		# update all model parameters we're fitting for
		if style == "D;theta;OL;B*x":
			microscope["gun"].size=(vals[0],0)
			microscope["gun"].angle=(vals[1],0)
			microscope["OL2"].calibration = vals[2]
			for i in range(4):
				microscope["PL"+str(i+1)].calibration = [0,vals[3]]	# "B" only, of A + B*x^(1/1) + C*x^(1/2) + ....
		if style == "theta;OL;B*x":
			microscope["gun"].angle=(vals[0],0)
			microscope["OL2"].calibration = vals[1]
			for i in range(4):
				microscope["PL"+str(i+1)].calibration = [0,vals[2]]	# "B" only, of A + B*x^(1/1) + C*x^(1/2) + ....
		if style == "theta;OL;B;B;B;B":
			microscope["gun"].angle=(vals[0],0)
			microscope["OL2"].calibration = vals[1]
			for i in range(4):
				microscope["PL"+str(i+1)].calibration = [0,vals[2+i]]	# "B" only, of A + B*x^(1/1) + C*x^(1/2) + ....
		# loop through each PL, concegating model list
		ds = []
		for i in range(4):
			ds.append([])
			# zero-out everyone
			for j in range(4):
				microscope["PL"+str(j+1)].strength = 0
			# loop through all strengths, propagate, measure beam diameter
			for s in strengths[-1]:
				microscope["PL"+str(i+1)].strength = s
				r1 = microscope.propagate_ray()
				x,y,xt,yt,R,I = measureAtZ(microscope["projector"].position+microscope["CCD"].position,rays=r1)
				ds[-1].append(np.absolute(x))
			#if i==0:
			#	microscope["PL"+str(i+1)].strength = strengths[-1][10]
			#	microscope.show()
		if ret=="dz":
			ds = sum(ds,[])
			dia = sum(diameters,[])
			return np.sqrt(np.sum((np.asarray(dia)-np.asarray(ds))**2))
		return ds
	# feed our error function to scipy minimize
	x = minimize(dz,x0=guesses)['x']
	# getting ready to plot the result
	modeled = dz(x,ret="full") ; markers = markers + ['-']*4
	if style == "theta;OL;B*x":
		labels = labels + [ "theta="+str(np.round(x[0],4))+",OL="+str(np.round(x[1],4))+",K="+str(np.round(x[2],4))+"*S" ]*4
	if style == "theta;OL;B;B;B;B":
		for i in range(4):
			labels.append( "theta="+str(np.round(x[0],4))+",OL="+str(np.round(x[1],4))+",K="+str(np.round(x[2+i],4))+"*S" )

	# PLOTTING
	plot(strengths+strengths, diameters+modeled, xlabel="strength", ylabel="diameter", labels=labels, markers=markers)# filename=style.replace(";","").replace("+","").replace('*',"")+".png")
	#	print("current:",x)
	#	c=input(style+": ")
	#	if "q" in c:
	#		break
	#	x=[ float(v) for v in c.split(",") ]

#def sweep_fitting_2D():

# FOR EACH LENS PAIR SWEEP, LOAD DATA, FIND VERTICAL AND HORIZONTAL "ASYMTOTES": critical current where dDiameter/dPL=0, in other words, the critical current for lens 1 which makes the beam diameter insensitive to lens 2, implying 1 has placed a crossover at 2. OR, upstream, a critical current for lens 2 which makes the beam diameter insensitive to lens 1
def I_crit_from_2D():
	log = open("Icrit_from_2D.csv",'w')
	direc = "20260619/" ; lines = open(direc+"index.txt",'r').readlines()[1:]
	#for PL in ["PL2","PL3","PL4"]:
	#	#data = np.load("20260429/PL1_"+PL+"_diameters.npy")
	#	#data = np.load("20260517/P1_"+PL.replace("PL","P")+".npy") ; data[1::2,:] = data[1::2,::-1]
	#	data = np.load("/mnt/macstem/USERS/Thomas Pfeifer/TWP20260517/tools/calsweeps_20260518_2d/P1_"+PL.replace("PL","P")+".npy") ; data[1::2,:] = data[1::2,::-1] # flip alternating rows, because we did serpentine acquire but didn't flip it while saving
	#	na,nb = data.shape
	#	dim_a = np.arange(na)*.05 ; dim_b = np.arange(nb)*.05
	for l in lines:
		cols = l.split("#")[0].split()
		letter,what,vswhat = cols[:3]
		extras = ""
		if len(cols)==4:
			extras = cols[3]
		f = direc+letter+"/"+what+"_"+vswhat+"_diameters.npy"
		if os.path.exists(f):
			data = np.load(f)
		else:
			f = direc+letter+"/"+vswhat+"_"+what+"_diameters.npy"
			data = np.load(f).T
			#what,vswhat=vswhat,what
		dim_a = np.load(direc+letter+"/"+what+".npy")
		dim_b = np.load(direc+letter+"/"+vswhat+".npy")
		interp = RegularGridInterpolator((dim_a,dim_b),data)
		int_a = np.linspace(min(dim_a),max(dim_a),1000)
		int_b = np.linspace(min(dim_b),max(dim_b),1002)


		int_aa,int_bb = np.meshgrid(int_a,int_b,indexing='ij')
		interpolated = interp((int_aa,int_bb)) ; print(np.shape(interpolated))
		dzdb=interpolated[:,1:]-interpolated[:,:-1]
		a = np.argmin( np.amax( np.absolute(dzdb), axis=1 ) ) # which "a" (PL1) value has all-low slopes (invariant wrt) "b" (PL3)
		print("sweep",letter,"has",what,"focused into",vswhat,"at",int_a[a],"mA")

		#magic_PL1s[PL]=int_a[a]
		#dzda=interpolated[1:,:]-interpolated[:-1,:]
		#b = np.argmin( np.amax( np.absolute(dzda), axis=0 ) ) # which "b" (PL3) value has all-low slopes (invariant wrt) "a" (PL1)
		overplot=[{"xs":[int_a[a]]*len(int_b),"ys":int_b,"kind":"line","c":"r"}]#,
		#	{"xs":int_a,"ys":[int_b[b]]*len(int_a),"kind":"line","c":"b"}]
		#contour(interpolated.T,int_a,int_b,heatOrContour="pix",overplot=overplot,xlabel=what,ylabel=vswhat,title="2D sweeps",zlabel="diameter")
		log.write(what+","+vswhat+","+str(int_a[a])+","+extras+"\n")
		#log.write(PL+",PL1,"+str(int_b[b])+"\n")

# FOR EACH 1D LENS SWEEP, FIT A QUADRATIC TO THE MINIMUM TO FIND THE TRUE INTERPOLATED STRENGTH AT MINIMUM DIAMETER
def minima_from_1D():
	def quad(xs,a,b,c):
		return a*(xs-b)**2+c
	log = open("minima_from_linear.csv",'w')
	# dataset from.....
	#data = np.loadtxt("linear_sweeps.csv",delimiter=",") # columns are: strength, diameter, wobble dx, dy
	# dataset from 2026-05-18. CHECK TEXT FILES IN SUBFOLDERS (DQCM should be off for 1D sweeps)
	data = np.zeros((31,4*4))
	files = list(sorted(glob.glob("/mnt/macstem/USERS/Thomas Pfeifer qwe/TWP20260517/tools/calsweeps_20260518_2d/*.npy")))
	strengths = np.arange(25)*.03
	data = np.load("linear_sweeps.npy")
	for i,s in enumerate(strengths):
		break
		for n in range(1,5):
			print("looking for","P"+str(n)+"_"+str(np.round(s,3)))
			for f in files:
				if "P"+str(n)+"_"+str(np.round(s,3)) in f:
					print("found",f)
					im = np.load(f)**2 # squaring helps get rid of diffraction spots?
					sy,sx = np.shape(im) ; xs = np.arange(sx) ; ys = np.arange(sy)
					#im[im>0] = np.sqrt(im[im>0])
					caching = ".ellipses/"+f.replace("/","_").replace(".npy",".json")
					(x,y),(cxe,cye,ae,be,thetae) = findEllipse(im,xs,ys,caching=caching)
					#contour(im, xs, ys, overplot=[{"xs":x,"ys":y,"kind":"line","c":"b"}], filename=caching.replace(".json",".png"))
					data[i,(n-1)*4]=s ; data[i,(n-1)*4+1]=np.sqrt(ae**2+be**2)
					break
	#np.save("linear_sweeps.npy",data)

	for i,PL in enumerate([ "PL"+str(n) for n in range(1,5) ]):
		diameter = data[:,i*4+1]
		strength = data[:,i*4]
		#plot([strength],[diameter])
		n = np.argmin(diameter)
		n1 = max(n-3,0) ; n2=n+5
		xs = strength[n1:n2]
		ys = diameter[n1:n2]
		#plot([strength],[diameter])
		guesses = [10*((max(ys)-min(ys))/(xs[3]-xs[0]))**2,xs[3],ys[3]]
		plot([strength,strength,xs],[diameter,quad(strength,*guesses),ys])

		x,_ = curve_fit(quad,xs,ys,p0=guesses)
		plot([strength,strength],[diameter,quad(strength,*x)],markers=['.','-'],ylim=[0,diameter[0]*2])
		log.write(PL+","+str(x[1])+"\n")

# CRITICAL CURRENTS, FORWARDS:

#      x.'\      |x₂|=|  1  0 | |x₁| lens
#     .' | \θ₂   |θ₂| |-1/f 1 | |θ₁|
# θ₁.'   |  \    |x₂|=| 1 L | |x₁| drift
# .'_____|_l_\   |θ₂| | 0 1 | |θ₁|
#z₀     z₁   z₂
#
# x₁ = (z₁-z₀)*θ₁ if x₀=0	(1) drift 0-1
# θ₂ = θ₁-x₁/f				(2) lens at 1
# x₂ = x₁+(z₂-z₁)*θ₂ = 0	(3) drift 1-2
# θ₂ = -x₁/(z₂-z₁)			(4) rearrangement of (3)
# θ₁ = x₁/(z₁-z₀)			(5) rearrangement of (1)
# -x₁/(z₂-z₁) = x₁/(z₁-z₀)-x₁/f	(6) plugging (4) and (5) into (2)
# 1/f = 1/(z₁-z₀)+1/(z₂-z₁)	(7) simplifying/rearranging (6)
# TROUBLE: we don't actually know that the rays entering plane 1 are divergent.
# (z₁-z₀) = x₁/θ₁			(8) rearrangement of (1)
# 1/f = θ₁/x₁+1/(z₂-z₁)		(9) plug (8) into (7) to generalize, which allows convergent beam entering 1.
# OR, recognize (9) is still equivalent with negative (z₁-z₀) (non-physical but mathematically fine)
# downstream lenses: triangle goes from z₀ (imaginary preceeding image plane) to z₁, to z₂,z₃, or z₄
# PL1 v *:	1/f₁₂ = 1/(z₁-z₀)+1/(z₂-z₁)
#			1/f₁₃ = 1/(z₁-z₀)+1/(z₃-z₁)
#			1/f₁₄ = 1/(z₁-z₀)+1/(z₄-z₁)
# upstream lens: triangle goes from z₁ (preceeding lens), to z₂, to z₅ (CCD location)
# PL1 v 2: 	1/f₂₁ = 1/(z₂-z₁)+1/(z₅-z₂)
# PL1 v 3: 	1/f₃₁ = 1/(z₃-z₁)+1/(z₅-z₃)
# PL1 v 4: 	1/f₄₁ = 1/(z₄-z₁)+1/(z₅-z₄)
# 1D lens sweeps: triangle goes from z₀ (imaginary preceeding image plane) to z₁, to z₅ (CCD location)
# PL1: 		1/f₁ = 1/(z₁-z₀)+1/(z₅-z₁)
# PL2: 		1/f₂ = 1/(z₂-z₀)+1/(z₅-z₂)
# PL3: 		1/f₃ = 1/(z₃-z₀)+1/(z₅-z₃)
# PL4: 		1/f₄ = 1/(z₄-z₀)+1/(z₅-z₄)
# unknowns and uncertains:
# positions: we *think* we know z₁,z₂,z₃,z₄,z₅.
# we measured Iᵢ and Iᵢⱼ but do not know the function which relates Iᵢ and Iᵢⱼ to fᵢ and fᵢⱼ. but we hope it is I²∝K²≈1/f
# (but we can be confident the function relating Iᵢ to fᵢ is the same as that which relates Iᵢⱼ to fᵢⱼ)

# USE BOTH OF THESE ERROR FUNCTIONS TO BUILD OUT OUR FULL ERROR FUNCTION BELOW, COMPRISED OF ALL LENS CROSSOVERS
def dz_single_lens(d1,d2,iF): # 1/f=1/d1+1/d2 where 1/f = C*I^2
	if d1==0 or d2==0: # special case: avoid div/by/zero error
		return np.inf
	return 1/d1+1/d2-iF
def dz_double_lens(d1,d2,d3,iF1,iF2): # 1/f1=1/d1+1/d2a ; 1/f2=1/d2b+1/d3 ; da=d2a+d2b, where 1/f = C*I^2
	if d1==0 or d3==0:
		return np.inf
	if iF1==1/d1: # special case, will throw divide by zero error. paralel beam between lens 1 and 2, d2 is arbitrary
		return 1/d3-iF2
	d2a = 1/(iF1-1/d1)
	d2b = d2-d2a
	return 1/d2b+1/d3-iF2

dz = None
def analytical_crossover_fitting():
	# LOAD CRITICAL CURRENTS: minima from 1D sweeps (lens focuses to detector), asymtotes from 2D sweeps (lens focuses to another lens)
	states = []
	lines = open("minima_from_linear.csv").readlines()
	#for l in lines: # e.g. "PL1,-0.012345\n"
	#	k,v = l.split(",")
	#	vals = [0,0,0,0] ; vals[int(k.replace("PL",""))-1] = float(v)
	#	states.append( vals+["0/"+k.replace("PL","P")+"/CCD"] )	# -0.012345,0,0,0,0/P1/CCD --> PL1-PL4 set to...yields triangle from z0 to P1 to CCD

	lines = open("Icrit_from_2D.csv").readlines()
	for l in lines:	# e.g. "P2,P4,567.89,P1=500"
		k1,k2,v,extra = l.split(",")
		vals = [0,0,0,0] ; vals[int(k1.replace("P",""))-1] = float(v)/1000	# 0,0.56789,0,0
		if "=" in extra:
			k,v = extra.split("=")
			vals[int(k.replace("P",""))-1] = float(v)/1000	# 0.5,0.56789,0,0
		if int(k1.replace("P","")) > int(k2.replace("P","")): # e.g. back propagation focusing P4 into P2 forms triangle P2/P4/CCD
			vals.append(k2+"/"+k1+"/CCD")
		else:
			vals.append("0/"+k1+"/"+k2) # forwards propagation focusing P2 into P4 forms triangle z0/P2/P4
		if "=" in extra:				# only forwards propagation cares if there is a preceding lens
			vals[-1] = vals[-1].replace("0/","0/P1/")
		states.append( vals )

	for s in states:
		print(s)

	# LOAD POSITIONS FROM MICROSCOPE (built by builder.py)
	microscope = mic_load("macstem")
	z = { e.replace("PL","P"):
			(microscope["projector"].position+microscope[e].position)-(microscope["objective"].position+microscope["sample"].position)
				for e in [ "PL1","PL2","PL3","PL4","CCD" ] }

	microscope.adjust_length("PL1", L=8)
	microscope.adjust_length("PL2", L=8)
	microscope.adjust_length("PL3", L=8)
	microscope.adjust_length("PL4", L=8)

	# IF YOU MOVE AN ELEMENT IN THE ANALYTICAL MATH, YOU MUST MOVE IT IN THE MICROSCOPE TOO, OR ELSE test_crossovers() AND FOLLOWING WILL BE INCORRECT
	#z["P2"]+=1 ; microscope["projector"].move("PL2",dz=+1)
	#z["P3"]+=4 ; microscope["projector"].move("PL3",dz=+4)
	z["P4"]+=0 ; microscope["projector"].move("PL4",dz=0)

	# ERROR FUNCTION FOR SCIPY-MINIMIZE: residual from all analytical equations. note: equations are non-linear, so we can't matrix solve
	# Historical calibration guesses removed; use generic values such as [.1, .2, .3, .4] when testing fitting strategies.

	# ITERATIVE STRATEGY: FIRST, ONLY USE P1/P2/CCD P1/P3/CCD P1/P4/CCD P2/P3/CCD P2/P4/CCD P3/P4/CCD (ignore C1 for now)
	#states = [ s for s in states if "CCD" in s[4] and len(s[4].split("/"))==3 ]
	# Generic iterative example: fit_vars = ["C2","C3","C4"], guesses = [.1,.2,.3], bounds = [[0,1],[0,1],[0,1]].
	# THIS YIELDS DEAD-ON CALS (once we moved P4 slightly). SET VALUES TO CALIBRATIONS SO THEY GET INHERITED ON NEXT ITERATION
	#PL234 = [1.1, 1.2, 1.3]
	#PL234 = [.12345, .23456, .34567]
	PL234 = [.12345, .23456, .34567]
	for i,v in enumerate(PL234):
		microscope["PL"+str(i+2)].calibration = v
	#test_crossovers(microscope,states)
	states = [ s for s in states if "P1" ==s[4].split("/")[1] ]
	# Generic C1 example: fit_vars = ["C1"], guesses = [.1], bounds = [[0,1]].
	fit_vars = ["C1"] ; guesses = [.1] ; bounds = [[.05,.2]] # anonymized calibration example

	microscope["OL2"].calibration = 1.0 #; microscope["PL1"].strength = 0 ; microscope["PL2"].strength = 0 ; microscope["PL3"].strength = 0 ; microscope["PL4"].strength = 0 ; scope = microscope["sample":] ; scope.insert(0,Source()) ; scope.show()

	global dz
	# ANALYTICAL ERROR FUNCTION ASSUMING 1/f=(C*I)^2*L, WHICH FAILS FOR THICK LENSES
	def dz(vals,z0=None,printed=[False]): # vals are our calibration factors C
		# make use of:
		# dz_single_lens(d1,d2,C,I) # 1/f=1/d1+1/d2 where 1/f = C*I^2
		# dz_double_lens(d1,d2,d3,C1,I1,C2,I2)
		vals = {"dz1":0,"dz2":0,"dz3":0,"dz4":0,
				"B1":0,"B2":0,"B3":0,"B4":0, "z0":z0,
				"C1":microscope["PL1"].calibration,"C2":microscope["PL2"].calibration,
				"C3":microscope["PL3"].calibration,"C4":microscope["PL4"].calibration,
				} | { k:vals[i] for i,k in enumerate(fit_vars) } # map list into a dict, defaults for dz
		#print(vals)
		deltas = []
		z_copy = { k:v for k,v in z.items() }
		if vals["z0"] is not None:
			z_copy["0"] = vals["z0"]
		# parse our "states" lists
		for state in states: # e.g. [-0.012345, 0, 0, 0, '0/P1/CCD'] or [0.5, 0, 0.56789, 0, '0/P1/P3/P4']
			if not printed[0]:
				print("state",state)
			zs = [ z_copy[e]+vals.get(e.replace("P","dz").replace("PCCD","CCD"),0) for e in state[-1].split("/") ]
			#if not printed[0]:
			#	print(zs)
			if len(zs) == 3:
				d1 = zs[1]-zs[0] ; d2=zs[2]-zs[1]
				lens = state[-1].split("/")[1]	# '0/P1/CCD' --> 'P1'
				i = int(lens.replace("P",""))-1	# 'P1' --> 1 --> index 0
				I = state[i] ; C = vals[ lens.replace("P","C") ] ; B = vals[ lens.replace("P","B") ]
				if not printed[0]:
					print("zs",zs,"d1,d2",d1,d2,"P"+str(i+1),"C,I",C,I)
				L = microscope["PL"+str(i+1)].length
				d = dz_single_lens(d1,d2,(C*I)**2*L+B*I) # 1/f=1/d1+1/d2 where 1/f = (C*I)^2*L
			else:
				d1 = zs[1]-zs[0] ; d2=zs[2]-zs[1] ; d3=zs[3]-zs[2]
				lens1 = state[-1].split("/")[1]		# '0/P1/P3/P4' --> 'P1'
				i1 = int(lens1.replace("P",""))-1	# 'P1' --> 1 --> index 0
				I1=state[i1] ; C1 = vals[ lens1.replace("P","C") ] ; B1 = vals[ lens1.replace("P","B") ]
				lens2 = state[-1].split("/")[2]		# '0/P1/P3/P4' --> 'P3'
				i2 = int(lens2.replace("P",""))-1	# 'P3' --> 3 --> index 2
				I2=state[i2] ; C2 = vals[ lens2.replace("P","C") ] ; B2 = vals[ lens2.replace("P","B") ]
				if not printed[0]:
					print("zs",zs,"d1,d2,d3",d1,d2,d3,"P"+str(i1+1),"C1,I1",C1,I1,"P"+str(i2+1),"C2,I2",C2,I2)
				L1 = microscope["PL"+str(i1+1)].length
				L2 = microscope["PL"+str(i2+1)].length
				d = dz_double_lens(d1,d2,d3,(C1*I1)**2*L1+B1*I1,(C2*I2)**2*L2+B2*I2)
			deltas.append(d)
		if not printed[0]:
			print("deltas",deltas)
			printed[0]=True
		return np.sum(np.asarray(deltas)**2)

	# FULLY MATRIX MATH ERROR FUNCTION, USES test_crossovers TO RUN FULL PROPAGATION
	def dz(vals):
		for k,v in zip(fit_vars,vals):
			print(k,v)
			if "dz" in k:
				PL = k.replace("dz","PL")
				microscope.move(PL,dz=v)
			if "C" in k:
				PL = k.replace("C","PL")
				microscope[PL].calibration = v
			if "OL" in k:
				microscope["OL2"].calibration = v
		deltas = np.asarray( test_crossovers(microscope,states,noplot=True) )
		return np.sum(deltas**2)

	x=minimize(dz,x0=guesses,bounds=bounds)
	#x={'x':brute(dz,bounds,Ns=40,finish=None)} ; dz(x['x'])
	#print("MATRIX BASED DZ FOUND",[v for v in x['x']])
	#print(repr(microscope))
	#test_crossovers(microscope,states)
	#OLs = np.linspace(.95,1.05,100) ; funs=[]
	#for OL in tqdm(OLs):
	#	microscope["OL2"].calibration = OL
	#	x = minimize(dz,x0=guesses,bounds=bounds) ; f = x["fun"]
	#	funs.append(f)
	#plot([OLs],[funs],ylim=[0,np.mean(funs)],xlabel="OL",ylabel="residual",title="OL2 calibration indicates convergence into PLs")


	#z["0"]=65+z["P1"] ; print(z) #; sys.exit()
	#x=minimize(dz,x0=guesses,args=(156.62),bounds=bounds)
	#x = brute(dz, bounds, Ns=20, workers=1)
	#print(x) ; sys.exit()

	# FOR A RANGE OF TRIAL z0 (since fitting seems to be bad at finding z0)
	#z0s = np.linspace(5,200,1000)
	#funs = []
	#for z0 in tqdm(z0s):
	#	#continue
	#	x = minimize(dz,x0=guesses,args=(z0),bounds=bounds) ; f = x["fun"]
	#	#x = brute(dz, bounds, Ns=20, workers=8, args=(z0)) ; f = dz(x,z0)
	#	funs.append(f)

	# plot residual vs v0
	#plot([z0s],[funs],ylim=[0,np.mean(funs)],xlabel="z0",ylabel="residual",title="position of z0 indicates convergence into PLs")

	# WHAT ABOUT 2D? WHAT IF OUR BACK-PROJECTION STARTING POINT ISN'T THE CCD, BECAUSE THERE IS A DQCM?
	#z0s = np.linspace(140,170,100)
	#z5s = np.linspace(z["P4"],z["CCD"],101)
	#funs = np.zeros((100,101))
	#for i,z0 in enumerate(tqdm(z0s)):
	#	for j,z5 in enumerate(z5s):
	#		z["CCD"]=z5
	#		x = minimize(dz,x0=guesses,args=(z0),bounds=bounds) ; f = x["fun"]
	#		funs[i,j]=f
	#plot([z5s],[funs[0,:]],xlabel="z5",ylabel="fun")
	#contour(np.log(funs.T),z0s,z5s,xlabel="z0",ylabel="z5",heatOrContour="pix")

	#print("POST FIT")
	#print(repr(microscope))

	# RERUN FITTING AT BEST z0
	#z0 = z0s[np.argmin(funs)]
	#x_PLs=minimize(dz,x0=guesses,args=(z0),bounds=bounds)
	#print(x_PLs,z0)

	#x_PLs['x']=[1.1, 1.2, 1.3]


	#for P,C in zip(fit_vars,x_PLs['x']):
	#	if "C" in P:
	#		print(f)
	#		# 1/f = (I*C)^2*L --> sqrt(1/f/L) = I*C,
	#		microscope[P.replace("C","PL")].calibration = C # /microscope["PL"+str(n+1)].length #	 np.sqrt(1/f/microscope["PL"+str(n+1)].length)
	#	if "dz" in P:
	#		microscope["projector"].move(P.replace("dz","PL"),dz=C)
	#	#microscope["PL"+str(n+1)].length = 1
	#	if P == "z0":
	#		z0 = C

	# DO FITTED CALS FOR FOCAL LENGTHS (AT CURRENT LENS LENGTHS) CHECK OUT?
	#test_crossovers(microscope)
	# OOPS, DON'T DO THIS YET. FIT OL BASED ON z0 FIRST

	# NEXT, FIT FOR OL2 CALIBRATION BASED ON z0
	#microscope = microscope["sample":]
	#microscope.insert( 0., Source(size=(2e-4/100,0),np_xy=(3,1),angle=(.0001,0),na_xy=(3,0),name="gun") )

	#z_OLPL = microscope["projector"].position + ( microscope["OL2"].position + microscope["PL1"].position )/2
	#print(z_OLPL)
	#def dz(vals):
	#	microscope["OL2"].calibration = vals[0]
	#	r1 = microscope.propagate_ray()
	#	x,y,xt,yt,R,I = measureAtZ(z_OLPL,rays=r1)
	#	#print(x,y,xt,yt,x/xt,z_OLPL-x/xt)
	#	#microscope.show() ; sys.exit()
	#	return (z_OLPL-x/xt-z0)**2 # x=l*theta, z0+l=here
	#x_OL=minimize(dz,x0=[1])
	#print(x_OL)

	#test_crossovers(microscope,states)

	#microscope.show(title="does it look like convergence into PLs, from OL cal, comes from z0 of "+str(z0)+"?")

	# RELOAD, UPDATE MODEL, SAVE OFF
	microscope2 = mic_load("macstem")
	microscope2["projector"].elements = microscope["projector"].elements
	microscope2["OL1"].calibration =microscope["OL2"].calibration
	microscope2["OL2"].calibration = microscope["OL2"].calibration

	# OPTION 1 TO HANDLE ROTATION: use ignore_length to simplify math, simply calculate Cnew Lnew from Ccurrent Lcurrent
	for n in range(1,5):
		continue
		#microscope2["PL"+str(n)].ignore_length = True
		if os.path.exists("rotpermA.npy"):
			rotationPerAmp = -np.load("rotpermA.npy")[n-1]*1000 # per mA --> per A
			# (I*C_0)^2*L_0 = (I*C_f)^2*L_f, C_f*L_f = RPA
			# C_0*C_0*L_0 = C_f*RPA
			#C0,L0 = microscope2["PL"+str(n)].calibration,microscope2["PL"+str(n)].length
			#microscope2["PL"+str(n)].calibration = -C0**2*L0/rotationPerAmp
			#microscope2["PL"+str(n)].length = -rotationPerAmp/microscope2["PL"+str(n)].calibration
			#i,j = microscope2.index("PL"+str(n))
			#microscope2[i][j+1].length+=L0-microscope2[i][j].length
			#microscope2[i][j+1]._position-=L0-microscope2[i][j].length
			microscope2["PL"+str(n)].enforced_rotation = rotationPerAmp/microscope2["PL"+str(n)].calibration
	#print("PRESAVED")
	#print(repr(microscope2))
	#microscope2.show()
	#microscope2.save("macstem_calibratedPL")
	#return

	# OPTION 2 TO HANDLE ROTATION, USE calibration_from_f_and_I, BUT THIS ASSUMES THIN LENS f = 1/((C*I)**2*L) APPROX
	for i in range(1,5):
		continue
		print("PL"+str(i)+", current calibration",microscope["PL"+str(i)].calibration,"and length",microscope["PL"+str(i)].length)
		if os.path.exists("rotpermA.npy"):
			rotationPerAmp = -np.load("rotpermA.npy")[i-1]*1000 # per mA --> per A
		else:
			rotationPerAmp = None
		cstrength = .23456
		L0 = microscope2["PL"+str(i)].length ; print(L0)
		C = microscope["PL"+str(i)].calibration ; I = cstrength ; L = microscope2["PL"+str(i)].length
		f = 1/((C*I)**2*L) # 1/f = (C*I)^2*L
		microscope2["PL"+str(i)].calibration_from_f_and_I(f,I,rotationPerAmp=rotationPerAmp)
		i,j = microscope2.index("PL"+str(i))
		microscope2[i][j+1].length+=L0-microscope2[i][j].length
		microscope2[i][j+1]._position-=L0-microscope2[i][j].length


	# OPTION 3, MANUALLY CHECK FOCAL LENGTH WHILE CHECKING ROTATION??? similar to the code in
	print("PRE ROTATION ADJUSTMENT")
	#test_crossovers(microscope2)
	for n in range(1,5):
		rotationPerAmp = -np.load("rotpermA.npy")[n-1]*1000
		C,L = microscope2["PL"+str(n)].get_C_L_from_rotation(.23456,rotationPerAmp)
		microscope2["PL"+str(n)].calibration = C ; print("SET","PL"+str(n),"C TO",C)
		microscope2.adjust_length("PL"+str(n),L=L) ; print("SET","PL"+str(n),"L TO",L)
		#print(repr(microscope2))
	print(repr(microscope2))
	print("POST ROTATION ADJUSTMENT")
	test_crossovers(microscope2)


	#	for i,z in enumerate(zs):
	#		microscope2.move("PL"+str(i+1),z=z)
	#	deltas = np.asarray( test_crossovers(microscope2,noplot=True) )
	#	return np.sum(deltas**2)
	#x0 = [ microscope2.get_element_position("PL"+str(n)) for n in range(1,5) ]
	#bounds = [ [x-10,x+10] for x in x0 ]
	#print(x0,bounds) #; sys.exit()
	#x_OL=minimize(dz,x0=x0,bounds=bounds)

	#microscope2["OL1"].calibration = x_OL['x'][0]
	#microscope2["OL2"].calibration = x_OL['x'][0]
	print("PRESAVED")
	print(repr(microscope2))
	microscope2.show()
	microscope2.save("macstem_calibratedPL")

def test_crossovers(microscope = None, states = None, noplot = False):

	if states is None:
		states = []

		lines = open("Icrit_from_2D.csv").readlines()
		for l in lines:	# e.g. "P2,P4,567.89,P1=500"
			k1,k2,v,extra = l.split(",")
			vals = [0,0,0,0] ; vals[int(k1.replace("P",""))-1] = float(v)/1000	# 0,0.56789,0,0
			if "=" in extra:
				k,v = extra.split("=")
				vals[int(k.replace("P",""))-1] = float(v)/1000	# 0.5,0.56789,0,0
			if int(k1.replace("P","")) > int(k2.replace("P","")): # e.g. back propagation focusing P4 into P2 forms triangle P2/P4/CCD
				vals.append(k2+"/"+k1+"/CCD")
			else:
				vals.append("0/"+k1+"/"+k2) # forwards propagation focusing P2 into P4 forms triangle z0/P2/P4
			if "=" in extra:				# only forwards propagation cares if there is a preceding lens
				vals[-1] = vals[-1].replace("0/","0/P1/")
			states.append( vals )

	print(states)
	deltas = []
	for s in states:
		print("CHECK STATE",s)
		#if "P3" not in s[4]:# or "P3" in s[4]:
		#	print("SKIP")
		#	continue
		if microscope is None:
			scope = mic_load("macstem_calibratedPL") ; print("RELOAD FROM calibratedPL")
			# sanity check to make sure OL.py didn't mess up our projector entrance state
			#scope = mic_load("macstem_calibratedFull") ; print("RELOAD FROM calibratedFull") ; scope["DQCM"].strength=0
		else:
			scope = microscope.copy()
		if "CCD" in s[4]:
			PL0 = s[4].split("/")[0] ; PL0 = PL0.replace("P","PL")
			scope = scope[PL0:] ; print("INITIALIZE BEAM AT",PL0)
			# We should start the rays in the *middle* of the "focused back to" element?? or nah
			# scope[0][0] = Drift(length=scope[PL0].length,position=scope[PL0].position) # REPLACE LENS WITH DRIFT
			scope[0].elements.insert(0,Drift(length=0.01,position=scope[PL0].position)) # OR, KEEP THE LENS?
			scope[0][1]._position+=.01 ; scope[0][1].length-=.01
			scope.insert(0,Source())
			for n,v in enumerate(s[:4]):
				try:
					scope["PL"+str(n+1)].strength = v
				except:
					pass
			sc = [ v for v in s ]
			sc[:4] = [ float(np.round(v,4)) for v in sc[:4] ]
		else:
			scope = scope["sample":] ; print("INITIALIZE BEAM AT SAMPLE")
			scope.insert(0,Source())
			for n,v in enumerate(s[:4]):
				scope["PL"+str(n+1)].strength = v
			sc = [ v for v in s ]
			sc[:4] = [ float(np.round(v,4)) for v in sc[:4] ]
		print(repr(scope))
		scope.propagate_ray()
		diameters = []
		for element_name in sc[4].split("/"):
			element_name = element_name.replace("P","PL")
			try:
				z = scope.get_element_position(element_name)
			except:
				continue
			if element_name != "CCD":
				z += scope[element_name].length/2
			x,y,xt,yt,R,I = measureAtZ(z,rays=scope.rays)
			diameters.append( np.sqrt( x**2+y**2 ) )
		delta = diameters[-1]/np.amax(diameters)
		deltas.append( delta )
		if noplot:
			continue
		scope.show(title=str(sc)+" \\Delta"+str(np.round(delta,4)))
	print(deltas)
	return deltas


def visualize_fitted():
	sweep_fitting_1D_individual(microscope_name="macstem_calibrated",preview_only=True)

	# LOAD MICROSCOPE (built by builder.py), CROP TO POST-SAMPLE, ADD SOURCE WITH ALMOST-ONLY DIFFRACTION RAYS
	microscope = mic_load("macstem_calibrated")
	microscope = microscope["sample":]
	microscope.insert( 0., Source(size=(2e-4/100,0),np_xy=(3,1),angle=(.0001,0),na_xy=(3,0),name="gun") )

	#microscope["PL1"].strength = .23456 ; microscope["PL2"].strength = 0 ; microscope["PL3"].strength = .23456 ; #microscope["PL4"].strength = 0 ; microscope.show()

	# LOAD CRITICAL CURRENTS, SHOW MICROSCOPE TO CHECK PROXIMITY
	lines = open("Icrit_from_2D.csv",'r').readlines()
	for n in range(2,5):
		microscope["PL"+str(n)].strength = 0
	for n in range(2,5):
		continue
		for l in lines:
			if "PL1,PL"+str(n)+"," in l:
				microscope["PL1"].strength = float(l.split(",")[-1])
				microscope.show(title="is there a crossover at PL"+str(n)+"?")

	# GRID OF PLs TO COMPARE TO 2D SWEEPS
	for n in range(2,5):
		for i in range(1,5):
			microscope["PL"+str(i)].strength = 0
		diameters = np.zeros((100,101)) ; thetas = np.zeros(diameters.shape) #; thetas2 = np.zeros(diameters.shape)
		PL1s = np.linspace(0,.8,100) ; PLZs = np.linspace(0,.8,101)
		dist_to_diff = np.zeros(diameters.shape) ; dist_to_image = np.zeros(diameters.shape)

		for i,PL1 in enumerate(tqdm(PL1s)):
			for j,PL in enumerate(PLZs):
				microscope["PL1"].strength = PL1
				microscope["PL"+str(n)].strength = PL
				r1 = microscope.propagate_ray()
				#microscope.show()
				#print(r1[:,0,columnByName("R")])
				x,y,xt,yt,R,I = measureAtZ(microscope["projector"].position+microscope["CCD"].position,rays=r1)
				#print(x,y,xt,yt,R)
				#microscope.show()
				#R2=np.arctan2(y,x)
				diameters[i,j]=(x**2+y**2)**.5
				thetas[i,j]=R+(np.sign(x)-1)/2*np.pi # if x<0, flip by pi
				#thetas2[i,j]=R2
				planes = findPlanes(r1,"x") #['x']['diff' or 'image']['z' or 'M' or 'R' or 'p']
				zp = planes['x']['diff']['z']	# findPlanes returns fractional coordinated. 1.4 is 40% of the way through element 1
				zp = [ zFromFractional(r1[:,0,columnByName('z')],z) for z in zp ]
				zp=np.asarray(zp)
				k = np.argmin(np.absolute(zp-microscope.get_element_position("CCD")))
				dist_to_diff[i,j] = abs(zp[k]-microscope.get_element_position("CCD"))
				zp = planes['x']['image']['z']	# findPlanes returns fractional coordinated. 1.4 is 40% of the way through element 1
				zp = [ zFromFractional(r1[:,0,columnByName('z')],z) for z in zp ]
				zp=np.asarray(zp)
				k = np.argmin(np.absolute(zp-microscope.get_element_position("CCD")))
				dist_to_image[i,j] = abs(zp[k]-microscope.get_element_position("CCD"))


		overplot = []
		for l in lines:
			if "PL1,PL"+str(n)+"," in l:
				v = float(l.split(",")[-1])
				overplot.append({"xs":[v,v],"ys":[min(PLZs),max(PLZs)],"kind":"line","c":"r"})
			if "PL"+str(n)+",PL1," in l:
				v = float(l.split(",")[-1])
				overplot.append({"xs":[min(PL1s),max(PL1s)],"ys":[v,v],"kind":"line","c":"b"})
		where = np.asarray(np.where(dist_to_diff < 50 ))	# all values on grid below threshold
		diff_1s=[] ; diff_Zs =[]
		for i,j in where.T:									# i,j pairs [0,0,0,1,1,2,2,3,4,4...],[5,6,7,..]
			x = PL1s[i]
			if x in diff_1s:
				continue
			mask = np.zeros(len(where[0])) ; mask[where[0]==i]=1	# values in "where" with this i value (this column)
			diff_1s.append(x)
			diff_Zs.append(np.mean(PLZs[where[1][mask==1]]))		# row values (j) with this i value (this column)
		overplot.append({"xs":diff_1s,"ys":diff_Zs,"kind":"line","c":"g"})
		where = np.asarray(np.where(dist_to_image < 70 ))	# all values on grid below threshold
		image_1s=[] ; image_Zs =[]
		for i,j in where.T:									# i,j pairs [0,0,0,1,1,2,2,3,4,4...],[5,6,7,..]
			x = PL1s[i]
			if x in diff_1s:
				continue
			mask = np.zeros(len(where[0])) ; mask[where[0]==i]=1	# values in "where" with this i value (this column)
			image_1s.append(x)
			image_Zs.append(np.mean(PLZs[where[1][mask==1]]))		# row values (j) with this i value (this column)
		overplot.append({"xs":image_1s,"ys":image_Zs,"kind":"line","c":"c"})


		contour(diameters.T, PL1s, PLZs, xlabel="PL1", ylabel="PL"+str(n), heatOrContour="pix,contour", linecolor='w', zlabel="diameter", title="PL"+str(n), overplot=overplot)

		thetas-=2 ; thetas[thetas<-np.pi]+=2*np.pi ; thetas*=-1
		contour(thetas.T,PL1s,PLZs, xlabel="PL1",ylabel="PL"+str(n), heatOrContour="pix,contour",linecolor='w', zlabel="thetas",title="PL"+str(n),cmap='hsv',zlim=[-np.pi,np.pi])

		#thetas2-=2 ; thetas2[thetas2<-np.pi]+=2*np.pi ; thetas2*=-1
		#contour(thetas2.T,PL1s,PLZs, xlabel="PL1",ylabel="PL"+str(n), heatOrContour="pix,contour",linecolor='w', #zlabel="thetas2",title="PL"+str(n),cmap='hsv',zlim=[-np.pi,np.pi])

		contour(dist_to_diff.T,PL1s,PLZs, xlabel="PL1",ylabel="PL"+str(n), heatOrContour="pix,contour",linecolor='w', zlabel="dist_to_diff",title="PL"+str(n))


def DQCM():
	# LOAD MICROSCOPE (built by builder.py), CROP TO POST-SAMPLE, ADD SOURCE WITH ALMOST-ONLY DIFFRACTION RAYS
	microscope = mic_load("U100_calibrated")
	microscope = microscope["sample":]
	microscope.insert( 0., Source(size=(2e-4/100,0),np_xy=(3,1),angle=(.0001,0),na_xy=(3,0),name="gun") )

	# HYPOTHESIS, DQCM CAN BE TREATED AS A ROUND LENS, PROJECTING SOME UNKNOWN POST-PL PLANE ONTO THE CCD.
	# So where is this plane? check trusted reference PL settings now that we think the PLs are U100_calibrated
	xml_file = "../MACSTEM/AS2restore_20260103.xml"
	# uncomment to explore settings tree
	#print(rootControlSettingValue(level="R",path="",filename=xml_file))
	#print(rootControlSettingValue(level="C",path="S_Projectors",filename=xml_file))
	#print(rootControlSettingValue(level="S",path="S_Projectors/_Diffn 20mm  (ref)",filename=xml_file))
	#print(rootControlSettingValue(level="D",path="S_Projectors/_Diffn 20mm  (ref)/PL1",filename=xml_file))

	whitelist = ['_Diffn 20mm  (ref)','Diffn 100mm', 'Diffn 80mm', '_Diffn 60mm', '_Diffn 40mm' ]
	for setting in whitelist:
		for i in range(1,5):
			v = lookupStrengthsXML("S_Projectors/"+setting+"/PL"+str(i),filename=xml_file)
			microscope["PL"+str(i)].strength = v
		microscope.show()

# given some ronchigram images of the same lacy carbon, use openCV to do an alignment, thus inferring the relative rotation of images introduced by the lenses. following along with: https://learnopencv.com/image-alignment-feature-based-using-opencv-c-python/
def lacyRotationOpenCV():
	import cv2
	from scipy.ndimage import gaussian_filter
	a = np.load('/mnt/U100/Users/Thomas Pfeifer/2026_04_03_U100/TWP_code/sea-pearl/TWP20260402/tools/calsweeps/frame_PL1_0.0_PL2_700.0.npy')
	b = np.load('/mnt/U100/Users/Thomas Pfeifer/2026_04_03_U100/TWP_code/sea-pearl/TWP20260402/tools/calsweeps/frame_PL1_0.0_PL2_0.0.npy')
	orb = cv2.ORB_create(1000)
	a = gaussian_filter(a,10) ; b = gaussian_filter(b,10)
	a=a/np.amax(a)*255 ; a=a.astype(np.uint8) # 2D numpy arrays are allowed as cv2 images, but they appear to need to be 8-bit ints
	b=b/np.amax(b)*255 ; b=b.astype(np.uint8)
	kp1,d1 = orb.detectAndCompute(a,None)
	kp2,d2 = orb.detectAndCompute(b,None)
	matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
	matches = matcher.match(d1,d2,None)
	print([ m.distance for m in matches])
	matches = sorted(matches,key=lambda x: x.distance)
	print([ m.distance for m in matches])
	#numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
	#matches = matches[:numGoodMatches]
	#print([ m.distance for m in matches])
	imMatches = cv2.drawMatches(a, kp1, b, kp2, matches, None)
	cv2.imwrite("matches.jpg", imMatches)

# given a ronchigram image containing an arbitraryly stretched, rotated, etc, VOA full of lacy carbon: infer VOA (findEllipse), rotate to a,b axes, stretch x,y to make VOA round
def orthogonalize(f):
	# LOAD IMAGE
	print("load")
	im = np.load(f)
	sy,sx = np.shape(im) ; xs = np.arange(sx) ; ys = np.arange(sy)
	# ELLIPSE FITTING
	f_ellipse = ".ellipses/"+f.replace(".npy","_ellipse.json").replace("/","_")
	print("ellipse")
	if os.path.exists(f_ellipse):
		with open(f_ellipse, 'r') as fo:
			dic = json.load(fo)
		x=np.asarray(dic['x']) ; y=np.asarray(dic['y'])
		cxe=dic['cxe'] ; cye=dic['cye'] ; ae=dic['ae'] ; be=dic['be'] ; thetae=dic['thetae']
	else:
		(x,y),(cxe,cye,ae,be,thetae) = findEllipse(im,xs,ys)
		with open(f_ellipse,'w') as fo:
			json.dump({ "x":list(x), "y":list(y), "cxe":cxe, "cye":cye, "ae":ae, "be":be, "thetae":thetae },fo)
	# EXCLUDE ELLIPSES OUT OF BOUNDS
	#if np.amax(x_1)>sx or np.amin(x_1)<0 or np.amax(y_1)>sy or np.amin(y_1)<0:
	#	print("WARNING, ELLIPSE OOB")
	print("roll")
	dx = int(round(sx/2-cxe)) ; dy = int(round(sy/2-cye))
	im = np.roll(np.roll(im,dx,axis=1),dy,axis=0)
	x+=dx ; y+=dy
	#b = np.roll(np.roll(b,int(round(sx/2-cxe_b)),axis=1),sy//2-cye_b,axis=0)
	print("contour")
	#contour(im,xs,ys,heatOrContour="pix",overplot=[{"xs":x,"ys":y,"kind":"line","c":"r"}])
	print("rotate")
	im = Image.fromarray(im)
	im = im.rotate(thetae*180/np.pi)
	# TODO: expand-then-crop is terrrrible on ram if the ellipse is small. consider an initial pre-crop?
	im = im.resize((int(round(1024**2/ae)),int(round(1024**2/be))),resample=Image.Resampling.LANCZOS)
	ary = np.asarray(im)
	sy,sx = ary.shape
	i1=int(sx/2-1024/2) ; i2=i1+1024
	j1=int(sy/2-1024/2) ; j2=j1+1024
	ary = ary[j1:j2,i1:i2]
	ary = ary/np.sum(ary) # normalize intensity??
	return ary,thetae

def measureEllipseRotation():
	files = glob.glob("/media/qwe/Data/Various Code/rayTEM_resources/TWP20260506/MACSTEM Lens Sweeps/.ellipses/*.json")
	print(files)
	for PL in ["P1","P2","P3","P4"]:
		strengths = np.arange(0,16)*40.
		thetas = []
		for PZ in strengths:
			candidates = [ f for f in files  if PL+"_"+str(PZ/1000) in f ]
			print(PZ/1000)
			f = candidates[0]
			print(f)
			ellipse = json.loads("\n".join(open(f).readlines()))
			thetas.append(ellipse["thetae"])
		plot([strengths],[thetas],title=PL)
		out = np.zeros((2,len(strengths))) ; out[0,:] = strengths ; out[1,:] = thetas
		np.save("ellipse_rotation_"+PL+".npy",out)


# LOAD RONCHIGRAM IMAGES, PERFORM ALIGNMENT (recenter, rotate ellipse, anisotropic scale ellipse to circle), MEASURE ROTATION
def measureLacyRotation():
	from scipy.signal import correlate2d
	PL="P4"
	if PL=="P1":
		f1 = "/mnt/emmadrive/Staff/Hoglund/Pfeifer/MACSTEM/TWP20260517/tools/calsweeps_normal/P1_0.0.npy"
	else:
		f1 = "/mnt/emmadrive/Staff/Hoglund/Pfeifer/MACSTEM/TWP20260517/tools/calsweeps_PL1400mA/"+PL+"_0.0.npy"

	ary_1,theta_1 = orthogonalize(f1)

	strengths = np.arange(1,16)*40.
	ANGLES = []
	for PZ in strengths:
		#PZ=700.0
		#f2 = '/mnt/U100/Users/Thomas Pfeifer/2026_04_03_U100/TWP_code/sea-pearl/TWP20260402/tools/calsweeps/frame_PL1_0.0_PL2_700.0.npy'
		f2 = f1.replace(PL+"_0.0",PL+"_"+str(PZ/1000))
		if not os.path.exists(f2):
			files = glob.glob(f1.replace(f1.split("/")[-1],"*"))
			for f in files:
				if PL+"_"+str(PZ/1000) in f:
					f2 = f
					break
		print(f2)
		try:
			ary_2,theta_2 = orthogonalize(f2)
			assert ary_2.shape == (1024,1024)
		except:
			ANGLES.append(0)
			continue

		#stitched = np.zeros((1024,1024*2))
		#stitched[:,:1024]=np.load(f1)[::2,::2] ; stitched[:,1024:]=np.load(f2)[::2,::2]
		#contour(stitched, np.arange(1024*2), np.arange(1024), heatOrContour="pix", title="pre")# filename=".ellipses/"+f2.replace(".npy","_2.png").replace("/","_"),aspect=1,figsize=(16,7))


		theta_3s = np.linspace(0,2*np.pi,360*3)
		residual = []
		for theta_3 in tqdm(theta_3s):
			im = Image.fromarray(ary_2)
			im = im.rotate(theta_3*180/np.pi)
			residual.append( np.sum( (ary_1 - np.asarray(im))**2 ) )

		plot([theta_3s],[residual],filename=".ellipses/"+f2.replace(".npy","_1.png").replace("/","_"))
		theta_3 = theta_3s[ np.argmin(residual) ]
		im = Image.fromarray(ary_2)
		im = im.rotate(theta_3*180/np.pi)
		ary_3 = np.asarray(im)

		stitched = np.zeros((1024,1024*2))
		stitched[:,:1024]=ary_1 ; stitched[:,1024:]=ary_3

		#    15          15+20         -5
		#  ^ θ₁ ^    ^          ^       ^   ^  means original delta was:
		#  |   /      \   θ₂  .'      .'   /   Δθ = θ₂+θ₃-θ₁
		#  |  /     vs \    .'  +   .' θ₃ /
		#  | /          \ .'      .'     /

		theta = (theta_2+theta_3-theta_1)%(2*np.pi)
		contour(stitched, np.arange(1024*2), np.arange(1024), heatOrContour="pix", title=str(theta*180/np.pi), filename=".ellipses/"+f2.replace(".npy","_2.png").replace("/","_"),aspect=1,figsize=(16,7))
		ANGLES.append(theta)
	out = np.zeros((len(strengths),2))
	out[:,0]=strengths ; out[:,1]=ANGLES
	np.save(".ellipses/"+f1.split("/frame_")[0].replace("/","_")+"_"+PL+"_angles.npy",out)
	plot([strengths],[ANGLES],xlabel="lens strength",ylabel="lens rotation",title=PL,filename=".ellipses/"+f1.split("/frame_")[0].replace("/","_")+"_"+PL+"_angles.png")

def compareLacyRotations():
	xs = [] ; ys = []
	files = glob.glob("/media/qwe/Data/Various Code/rayTEM_resources/TWP20260506/MACSTEM Lens Sweeps/.ellipses/*.npy")
	# LOAD RAW DATA AS PROCESSED BY measureLacyRotation
	for PL in ["P1","P2","P3","P4"]:
		candidates = [ f for f in files if PL+"_angles" in f ]
		print(candidates)
		f = candidates[0]
		data = np.load(f) ; print(data.shape)
		x = data[:,0] ; y = data[:,1] ; x=x[y!=0] ; y=y[y!=0]
		xs.append(x) ; ys.append(y)
	mkrs = ['']*4 ; lbls = ["PL1","PL2","PL3","PL4"]
	#plot(xs,ys,labels=lbls,xlabel="strength",ylabel="rotation",markers=mkrs)
	# CURVE_FIT FITTING WITH A STEPPED LINEAR FUNCTION?
	# linear + errorfunction = linear with a jump
	if False:
		def lin(xs,m,x):
			return m*xs+(erf(xs-x)+1)/2*np.pi
		for j in range(4):
			x=xs[j] ; y=ys[j]
			# look at slope to find the pi jump
			i = np.argmax(y[1:]-y[:-1])
			guesses = ( y[i]/x[i], (x[i+1]+x[i])/2 )
			# delete datapoints near pi jump (small beam = difficult to measure rotation accurately)
			x=list(x) ; y=list(y)
			for k in range(2,-1,-1):
				del x[i+k] ; del y[i+k]
			x=np.asarray(x) ; y=np.asarray(y)
			# scipy curve_fit
			res,err = curve_fit(lin,x,y,p0=guesses)
			xs.append(x) ; ys.append(lin(x,*res)) ; mkrs.append('-') ; lbls.append(str(res[0]))
		plot(xs,ys,labels=lbls,xlabel="strength",ylabel="rotation",markers=mkrs)
		np.save("rotpermA.npy",[float(v) for v in lbls[-4:]])

	# LOAD MICROSCOPE (built by builder.py), CROP TO POST-SAMPLE, ADD SOURCE WITH ALMOST-ONLY DIFFRACTION RAYS
	microscope = mic_load("macstem_calibratedFull")
	microscope = microscope["sample":]
	microscope.insert( 0., Source(size=(2e-4/100,0),np_xy=(3,1),angle=(.0001,0),na_xy=(3,0),name="gun") )
	for n in range(1,5):
		for m in range(1,5):
			microscope["PL"+str(m)].strength=0
		xs.append([]) ; ys.append([]) ; mkrs.append("-") ; lbls.append("PL"+str(n))
		for s in [0]+list(xs[n-1]):
			microscope["PL"+str(n)].strength = s/1000 # mA to A
			r1 = microscope.propagate_ray()
			R = -r1[-1,-1,columnByName('R')]
			if r1[-1,-1,columnByName('x')]<0:
				R+=np.pi
			xs[-1].append(s) ; ys[-1].append(R)
		ys[-1]=np.asarray(ys[-1])-ys[-1][0] # ensure y-intercept is zero (this is *relative* rotation, so if OL gave us some rotation beforehand, ignore it)
		out = np.zeros((2,len(xs[-1]))) ; out[0,:] = xs[-1] ; out[1,:] = ys[-1]
		np.save("lacy_rotation_PL"+str(n)+".npy",out)

	plot(xs,ys,labels=lbls,xlabel="strength",ylabel="rotation",markers=mkrs)


main()
