import sys,glob,json
import numpy as np
from tqdm import *

sys.path.insert(1,"../../../../../../../niceplot")
from niceplot import *
from nicecontour import *

from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit,minimize

sys.path.insert(1,"../../../../")
from pySEA.rayTEM.assemblies import load_microscope as mic_load
from pySEA.rayTEM import MicroscopeSection,Source,Lens,Drift,columnByName,Aperture,measureAtZ
from pySEA.rayTEM.xmlNion import lookupStrengthsXML

def main():
	#I_crit_from_2D()
	#analytical_crossover_fitting()
	#visualize_fitted()
	analytical_crossover_fitting2()
	#visualize_fitted()


def I_crit_from_2D():
	# 2D LENS SWEEPS (higher res than with PLs, but zoomed-in on where i manually found the cross-overs via wobbling)
	path = '/mnt/emmadrive/Instruments/MACSTEM/USERS/Thomas Pfeifer/TWP20260517/tools/calsweeps_CLs_v04/'
	rundict = { "A":["CL2","CL3",0], "B":["CL2","CL3",.3], # which CL, is used to focus to which other CL, and what's the 3rd's value
			"C":["CL1","CL2",0], "D":["CL1","CL3",0],
			"E":["CL3","CL2",0], "F":["CL3","CL1",0], "G":["CL2","CL1",0] }
	# SURFACE HEATMAPS (diameter vs lens A vs lens B), INTERPOLATE/SMOOTH, FIND WHERE dD/dB==0
	for k,(xlens,ylens,zval) in rundict.items():
		diameters = np.load(path+k+"_diameters.npy")
		dim_a = np.load(path+k+"_"+xlens+".npy")
		dim_b = np.load(path+k+"_"+ylens+".npy")
		#contour(diameters.T,dim_a,dim_b,heatOrContour="pix",xlabel=xlens,ylabel=ylens)
		interp = RegularGridInterpolator((dim_a,dim_b),diameters)
		int_a = np.linspace(min(dim_a),max(dim_a),1000)
		int_b = np.linspace(min(dim_b),max(dim_b),1002)
		int_aa,int_bb = np.meshgrid(int_a,int_b,indexing='ij')
		interpolated = interp((int_aa,int_bb)) #; print(np.shape(interpolated))
		dzdb=interpolated[:,1:]-interpolated[:,:-1]
		a = np.argmin( np.amax( np.absolute(dzdb), axis=1 ) ) # which "a" (PL1) value has all-low slopes (invariant wrt) "b" (PL3)
		rundict[k].append(int_a[a])
		overplot=[{"xs":[int_a[a]]*len(int_b),"ys":int_b,"kind":"line","c":"r"}]
		#	{"xs":int_a,"ys":[int_b[b]]*len(int_a),"kind":"line","c":"b"}]
		#contour(interpolated.T,int_a,int_b,heatOrContour="pix",xlabel=xlens,ylabel=ylens,title="I_crit =

	# C1 VOA SWEEP: BEAM CURRENT MAXIMIZES WHERE C1 FOCUSES TO VOA, BUT TECHNICALLY MIDDLE OF THE MAXIMA IS NOT CROSSOVER
	if os.path.exists("C1_vs_beamcurrent.npy"):
		C1s,beamcurrent = np.load("C1_vs_beamcurrent.npy").T
	else:
		files = list(sorted(glob.glob("/mnt/macstem/USERS/Thomas Pfeifer/TWP20260517/tools/calsweeps_CLs_v02/*.npy")))
		C1s = np.arange(51)*.01 ; beamcurrent = []
		for C1 in tqdm(C1s):
			for f in files:
				if "C1_"+str(C1) in f:
					print("found",C1,"in file",f.split("/")[-1])
					beamcurrent.append( np.sum( np.load(f) ) )
					break
		out = np.zeros((len(C1s),2)) ; out[:,0]=C1s ; out[:,1]=beamcurrent
		np.save("C1_vs_beamcurrent.npy",out)
	#plot([C1s],[beamcurrent])
	# θ₀  _________r₁  |		as focus is adjusted, beam edge ray is "swept"
	# .-'|'-.          |		inwards across the edge of the aperture.
	#    | \'.'-.      |r		θ₁ = θ₀ - 1/f (and recall, 1/f = K² = S I²)
	#    |  \ '.  '-.			r₁ = r₀ + θ₁ l
	#    |   \  '.    '-.		r₁ = r₀ + l θ₀ - l/f = r₀ + l θ₀ - l S I²
	# 	lens       aperture		∫ CCD dA ∝ rᵥₒₐ²/r₁²
	# |∫       ||				f(x) = A/(B-C*x²)² recognizing we can't differentiate r₀ θ₀ S
	# |       /  \				B = r₀ + l θ₀ (horizontal offset), C = l S (width), A = rᵥₒₐ² (height)
	# |__...-'    '-...__		and r₁ = 0 when r₀ + l θ₀ - l C I² = 0 or B+C*x²=0
	# |___________________K		x where r₁ is at √(B/c)
	def fun(xs,A,B,C):
		return A/(B-C*xs**2)**2
	cx = C1s[np.argmax(beamcurrent)] ; cy = np.amax(beamcurrent)
	C=.5 ; B=cx**2*C ; A=np.sqrt(cy)
	guesses = (A,B,C)
	mask = np.zeros(len(C1s))
	mask[beamcurrent>np.amax(beamcurrent)/2]=1		# FWHM
	#mask[beamcurrent>np.mean(beamcurrent)]=1
	for i in range(len(mask)):						# FILL IN HOLES (if i'm below FWHM, but i have points above and below me which are, i'm probably the noise in the middle of the peak)
		if mask[i]==0 and np.sum(mask[:i])>0 and np.sum(mask[i:])>0:
			mask[i]=1
	C1s = C1s[mask==0] ; beamcurrent = beamcurrent[mask==0]
	res,err = curve_fit(fun,C1s,beamcurrent,p0=guesses)
	A,B,C=res ; I_crit = np.sqrt(B/C)
	np.save("C1_vs_beamcurrent_fit.npy",res)
	plot( [C1s,C1s,[I_crit,I_crit]], [beamcurrent,fun(C1s,*res),[0,np.amax(beamcurrent)]], markers=['','','-'], title="I_crit="+str(np.round(I_crit,6)))
	rundict["H"]=["CL1","VOA",0,I_crit]
	with open("CLs_I_crit.json",'w') as fo:
		json.dump(rundict,fo)

# USE BOTH OF THESE ERROR FUNCTIONS TO BUILD OUT OUR FULL ERROR FUNCTION BELOW, COMPRISED OF ALL LENS CROSSOVERS
def dz_single_lens(d1,d2,C,I): # 1/f=1/d1+1/d2 where 1/f = C*I^2
	if d1==0 or d2==0: # special case: avoid div/by/zero error
		return np.inf
	return 1/d1+1/d2-C*I**2
def dz_double_lens(d1,d2,d3,C1,I1,C2,I2): # 1/f1=1/d1+1/d2a ; 1/f2=1/d2b+1/d3 ; da=d2a+d2b, where 1/f = C*I^2
	if d1==0 or d3==0:
		return np.inf
	if C1*I1**2==1/d1: # special case, will throw divide by zero error. paralel beam between lens 1 and 2, d2 is arbitrary
		return 1/d3-C2*I2**2
	d2a = 1/(C1*I1**2-1/d1)
	d2b = d2-d2a
	return 1/d2b+1/d3-C2*I2**2


def analytical_crossover_fitting():
	# LOAD CRITICAL CURRENTS: asymtotes from 2D sweeps (lens focuses to another lens)
	with open("CLs_I_crit.json",'r') as f:
		rundict = json.load(f)
	print(rundict)

	# LOAD POSITIONS FROM MICROSCOPE (built by builder.py)
	microscope = mic_load("macstem")
	element_positions = { e:microscope["condenser"].position+microscope[e].position for e in [ "CL1","VOA","CL2","CL3" ] }
	#z={ k:element_positions[kk] for k,kk in {"1":"CL1","2":"VOA","3":"CL2","4":"CL3"}.items() }
	#z1 = z["1"] ; z = { k:v-z1 for k,v in z.items() } # put CL1 at z=0, so z0 can be +/- ahead or behind
	microscope = microscope[:2] ; microscope["objective"].insert(-1,Drift(length=100))
	#microscope.show()

	# ERROR FUNCTION FOR SCIPY-MINIMIZE: residual from all analytical equations. note: equations are non-linear, so we can't matrix solve
	#                 .-|.
	#             .-'   | '.
	#         .-'       |   '.
	#     .-'           |     '.
	# .-'      d1       |  d2   '.
	# r₁ = d₁ θ₁				(1) propagation
	# θ₂ = θ₁ - r₁/f₁			(2) lens
	# r₂ = r₁ + d₂ θ₂ = 0		(3) propagation
	# θ₁ = r₁/d₁				(4) rearrange 1
	# θ₂ = -r₁/d₂				(5) rearrange 3 if r₂=0
	# -r₁/d₂ = r₁/d₁ - r₁/f₁ 	(6) plug 4,5 into 2
	# 1/f₁ = 1/d₁ + 1/d₂		(7) rerrange 6
	# B       .-'. |V     .\
	#     .-'     '|    .'  \  /
	#_.-'______1____'..'__2__\3__
	# '-.           .''.     /\
	#     '-.     .|    '.  /  \
	#         '-.  |      '/ ₁₂₃
	# r₁ = d₁ θ₁						(1) propagation 0-1
	# θ₂ = θ₁ - r₁/f₁					(2) lens 1
	# r₂ = r₁ + d₂ θ₂					(3) propagation 1-2
	# θ₃ = θ₂ - r₂/f₂					(4) lens 2
	# r₃ = r₂ + d₃ θ₃ = 0				(5) propagation 2-3
	# θ₁ = r₁/d₁						(6) rearrange 1
	# θ₂ = r₁/d₁ - r₁/f₁				(7) plug 6 into 2
	# θ₃ = -r₂/d₃						(8) rearrange 5 if r₃=0
	# -r₂/d₃ = r₁/d₁ - r₁/f₁ - r₂/f₂ 	(9) plug 8 and 7 into 4
	# r₂ = r₁ + d₂ θ₁ - d₂ r₁/f₁		(10) plug 2 into 3
	# r₂ = r₁ + d₂ r₁/d₁ - d₂ r₁/f₁		(11) plug 6 into 10
	# -r₁/d₃ - d₂ r₁/d₁/d₃ + d₂ r₁/f₁/d₃ = r₁/d₁ - r₁/f₁ - r₁/f₂ - d₂ r₁/d₁/f₂ + d₂ r₁/f₁/f₂ (12) plug 11 into 9
	# 1/f₁ + 1/f₂ + d₂/d₃/f₁ + d₂/d₁/f₂ = 1/d₁ + 1/d₃ + d₂/f₁/f₂ + d₂/d₁/d₃ (13) rearrange/simplfying 12
	# sanity check: what if d₂=0 and 1/f₂=0, i.e., delete one drift?
	# simplfies to 1/f₁ = 1/d₁ + 1/d₃
	# what if 1/f₁ =0?
	# simplifies to: 1/f₂ + d₂/d₁/f₂ = 1/d₁ + 1/d₃ + + d₂/d₁/d₃
	# OR SIMPLY:
	# 1/f₁ = 1/d₁ + 1/d₂ᵢ ; 1/f₂ = 1/d₂ⱼ + 1/d₃ ; /d₂ᵢ+d₂ⱼ=d₂

	# TESTING OUR ERROR FUNCTIONS ABOVE. SIMPLE TWO LENS SYSTEM, DETECT BOTH CROSSOVERS
	if False:
		microscope = MicroscopeSection(elements=[Source(),Lens(name="L1",strength=1.,position=1,length=0),Lens(name="L2",strength=1,position=3,length=0),Drift(length=5)])

		def dz2(d3):
			return dz_double_lens(d1=microscope["L1"].position, d2=microscope["L2"].position-microscope["L1"].position, d3=d3,
							C1=1,I1=microscope["L1"].strength,C2=1,I2=microscope["L2"].strength)**2
		def dz1(d2):
			return dz_single_lens(d1=microscope["L1"].position, d2=d2,
							C=1,I=microscope["L1"].strength)**2

		x1 = minimize(dz1,x0=-1)
		x2 = minimize(dz2,x0=1)
		microscope.show(title=str(microscope["L1"].position+x1['x'][0])+","+str(microscope["L2"].position+x2['x'][0]))

	# PARTIAL ERROR FUNCTION, CL1 FOCUSING ONLY (since it is unclear whether dDiameter/dCL1 is because CL2 or CL3 are focused into CL1, or the VOA...., but at least CL1 focusing to VOA, CL2, CL3, should be well-defined)
	def dz(C1,z0):
		deltas = []
		for k,(lensA,lensB,zval,Icrit) in rundict.items(): # A,(CL2,CL3,0,0.09) --> "CL2 focuses into CL3 at 90mA when CL1 is zero"
			if lensA!="CL1":
				continue
			d1 = element_positions["CL1"]-z0 ; d2 = element_positions[lensB]-element_positions["CL1"]
			d = dz_single_lens(d1,d2,C1,Icrit) #; print(d1,d2)
			deltas.append(d)
		return np.sum(np.asarray(deltas)**2)
	z0s = np.linspace(150,300,1000) ; funs = []
	for z0 in z0s:
		x=minimize(dz,x0=[1],args=(z0))
		funs.append(x['fun'])
	#plot([z0s],[funs])
	z0 = z0s[np.argmin(funs)]
	x=minimize(dz,x0=[1],args=(z0))
	C1 = x['x']
	# RECALL: we fitted for C in "1/f = C*I^2", not "1/f = L*(C*I)^2", so the latterC = sqrt(formerC/L)
	#microscope["CL1"].calibration = np.sqrt(C1/microscope["CL1"].length)
	microscope["CL1"].calibration_from_f_and_I(1/C1/(.25)**2,.25)

	# INSERT A GUN LENS, CALIBRATED, TO GIVE US REQUIRED CONVERGENCE INTO CL1. (gun is parallel beam)
	z_CL1 = microscope["CL1"].position
	#f_GL = 1/(1/(z_CL1/2)+1/(z0-z_CL1/2)) # 1/f = 1/d1 + 1/d2, where GL is halfway between source and CL1, and focuses to z0 THIS WOULD BE TRUE FOR DIVERGENT SOURCE, BUT WE HAVE PARALLEL
	f_GL = z0-z_CL1/2 # for a starting-parallel beam, f is just where we want it to focus to
	microscope.insert(z_CL1/2,Lens(name="GL",strength=1,length=0))
	microscope["GL"].calibration_from_f_and_I(f_GL,1)

	#for k,(lensA,lensB,zval,Icrit) in rundict.items():
	#	if lensA!="CL1":
	#		continue
	#	microscope["CL1"].strength=Icrit ; microscope["CL2"].strength = 0 ; microscope["CL3"].strength = 0
	#	microscope.show(title=k+", CL1 focuses to "+lensB)

	# PARTIAL ERROR FUNCTION FOR CL2 (use z0 for forwards-propagation, and check the backwards propagation into CL1)
	def dz(C2):
		deltas = []
		for k,(lensA,lensB,zval,Icrit) in rundict.items(): # A,(CL2,CL3,0,0.09) --> "CL2 focuses into CL3 at 90mA when CL1 is zero"
			if lensA!="CL2" or lensB!="CL3":
				continue
			z1,z2,z3 = element_positions["CL1"],element_positions["CL2"],element_positions["CL3"]
			d1 = z1-z0 ; d2 = z2-z1 ; d3 = z3-z2
			d = dz_double_lens(d1,d2,d3,C1,zval,C2,Icrit)
			deltas.append(d)
		return np.sum(np.asarray(deltas)**2)
	x=minimize(dz,x0=[1])
	C2 = x['x']
	microscope["CL2"].calibration_from_f_and_I(1/C2/(.25)**2,.25)

	#for k,(lensA,lensB,zval,Icrit) in rundict.items():
	#	if lensA!="CL2" or lensB!="CL3":
	#		continue
	#	microscope["CL1"].strength=zval ; microscope["CL2"].strength = Icrit ; microscope["CL3"].strength = 0
	#	microscope.show(title=k+", CL2 focuses to "+lensB)

	# IN THEORY, WE NOW KNOW C1, C2, WE DO NOT KNOW C3, OR z5. BUT WE HAVE CL2>CL1 (TELLS z5), AND CL3>CL2 AND CL3>CL1 (TELLS C3 and Z5)

	# PARTIAL ERROR FUNCTION FOR CL3
	def dz(C3,z5):
		deltas = []
		for k,(lensA,lensB,zval,Icrit) in rundict.items(): # A,(CL2,CL3,0,0.09) --> "CL2 focuses into CL3 at 90mA when CL1 is zero"
			if not ( (lensA=="CL3" and lensB in ["CL1","CL2"]) or (lensA=="CL2" and lensB=="CL1") ):
				continue
			if lensB=="CL1": # TWO ORDERS OF MAGNITUDE LOWER RESIDUAL (see plotted z5s vs funs below) WHEN WE FOCUS BACK TO VOA INSTEAD OF CL1.
				lensB="VOA"
			z1,z2 = element_positions[lensB],element_positions[lensA]
			d1 = z2-z1 ; d2 = z5-z2
			C = {"CL2":C2,"CL3":C3}[lensA]
			d = dz_single_lens(d1,d2,C,Icrit)
			deltas.append(d)
		return np.sum(np.asarray(deltas)**2)

	z5s = np.linspace(150,500,1000) ; funs = []
	for z5 in z5s:
		x=minimize(dz,x0=[1],args=(z5))
		funs.append(x['fun'])
	plot([z5s],[funs],ylim=[0,.0001])
	z5 = z5s[np.argmin(funs)]
	x=minimize(dz,x0=[1],args=(z5))
	C3 = x['x']
	microscope["CL3"].calibration_from_f_and_I(1/C3/(.25)**2,.25)

	# RELOAD, UPDATE MODEL, SAVE OFF
	microscope2 = mic_load("macstem")

	for n in range(1,4):
		microscope2["CL"+str(n)].calibration = microscope["CL"+str(n)].calibration
	microscope2["OL1"].calibration = microscope["OL1"].calibration
	microscope2.insert(microscope["GL"].position,microscope["GL"])
	microscope2["VOA"].name = ""
	microscope2.insert(microscope["VOA"].position,microscope["VOA"])
	microscope2.save("macstem_calibratedCL")

def visualize_fitted():
	with open("CLs_I_crit.json",'r') as f:
		rundict = json.load(f)
	print(rundict)

	microscope = mic_load("macstem_calibratedCLPL")[:"PL1"]

	for k,(lensA,lensB,zval,Icrit) in rundict.items(): # A,(CL2,CL3,0,0.09) --> "CL2 focuses into CL3 at 90mA when CL1 is zero"
		#continue
		if microscope.get_element_position(lensA) > microscope.get_element_position(lensB):
			continue
		microscope[lensA].strength = Icrit
		microscope[lensB].strength = 0
		lensC = ["CL1","VOA","CL2","CL3"] ; del lensC[lensC.index(lensA)] ; del lensC[lensC.index(lensB)] ; lensC=lensC[0]
		microscope[lensC].strength = zval
		microscope.show(title=k+", "+lensA+" focuses into "+lensB)

	C1s,beamcurrent = np.load("C1_vs_beamcurrent.npy").T
	beamcurrent_2 = beamcurrent[C1s<.375] ; C1s_2=C1s[C1s<.375]
	beamcurrent/=np.amax(beamcurrent_2) ; beamcurrent_2/=np.amax(beamcurrent_2)
	modelcurrent = []
	for c1 in C1s_2:
		microscope["CL1"].strength = c1
		r1 = microscope.propagate_ray()
		modelcurrent.append( r1[-1,-1,columnByName('I')] )
	plot([C1s*1000,C1s_2*1000],[beamcurrent,modelcurrent],xlabel="C1 current (mA)",ylabel="beam current (a.u.)",labels=["measured","modeled"],title="",filename="CL_VOA.svg",ylim=[0,None],xlim=[0,500])

# analytical_crossover_fitting did individual fitting: CL1 first, then CL2, then CL3. HERE, we do simultaneous?
def analytical_crossover_fitting2():

	# LOAD CRITICAL CURRENTS: asymtotes from 2D sweeps (lens focuses to another lens)
	with open("CLs_I_crit.json",'r') as f:
		rundict = json.load(f)
	print(rundict)

	# LOAD POSITIONS FROM MICROSCOPE (built by builder.py)
	microscope = mic_load("macstem")[:"PL1"]
	element_positions = { e:microscope["condenser"].position+microscope[e].position for e in [ "CL1","VOA","CL2","CL3" ] }

	# FULL ERROR FUNCTION, COMPRISED OF ALL LENS CROSSOVERS
	def dz(vals,z0,z5):
		if len(vals)==5:
			Cs = vals[:3] ; z0=1/vals[3] ; z5 = 1/vals[4]
		else:
			Cs = vals
		deltas = []
		for k,(lensA,lensB,zval,Icrit) in rundict.items(): # A,(CL2,CL3,0,0.09) --> "CL2 focuses into CL3 at 90mA when CL1 is zero"
			#microscope[lensA].strength = Icrit		# lens being used to focus into....
			#microscope[lensB].strength = 0			# lens being focused into (strength doesn't matter)
			#lensC = ["CL1","VOA","CL2","CL3"]		# remaining lens
			#del lensC[lensC.index(lensA)] ; del lensC[lensC.index(lensB)] ; lensC=lensC[0]
			#microscope[lensC].strength = zval
			if lensA=="CL2" and lensB=="CL3": # CL1/CL2 --> CL3
				C1,C2,C3 = Cs
				I1 = zval ; I2 = Icrit
				d1 = element_positions["CL1"] - z0
				d2 = element_positions["CL2"] - element_positions["CL1"]
				d3 = element_positions["CL3"] - element_positions["CL2"]
				d = dz_double_lens(d1,d2,d3,C1,I1,C2,I2)
			if lensB == "CL1": # CL1 <-- CL2 or CL3. we should actually use VOA for the position
				C = vals[ ["CL1","CL2","CL3"].index(lensA) ]
				I = Icrit
				d1 = element_positions[lensA] - element_positions["VOA"]
				d2 = z5 - element_positions[lensA]
				d = dz_single_lens(d1,d2,C,I)
			if lensA == "CL1": # simple forward-propagation
				C=vals[0] ; I=Icrit
				d1 = element_positions["CL1"] - z0
				d2 = element_positions[lensB]-element_positions["CL1"]
				d = dz_single_lens(d1,d2,C,I)
			if lensA == "CL3" and lensB == "CL2": # simple backwards propagation
				C=vals[2] ; I=Icrit
				d1 = element_positions[lensA] - element_positions[lensB]
				d2 = z5 - element_positions[lensA]
				d = dz_single_lens(d1,d2,C,I)
			deltas.append( d )
		return np.sum(np.asarray(deltas)**2)

	# FOR A RANGE OF TRIAL z0 z5 (since fitting seemed to be bad at finding z0 in the PL section)
	z0 = 218.1818181818182 ; z5 = 312.0
	if False:
		z0s = np.linspace(150,400,100)
		z5s = np.linspace(200,1000,101)
		#z0s = np.concatenate((np.linspace(-1000,-100,50),np.linspace(100,1000,50)))
		#z5s = np.concatenate((np.linspace(-1000,-100,50),np.linspace(100,1000,51)))
		funs = np.zeros((len(z0s),len(z5s)))
		for i,z0 in enumerate(tqdm(z0s)):
			for j,z5 in enumerate(z5s):
				x=minimize(dz,x0=[1,1,1],args=(z0,z5))
				fun=x['fun']
				if not np.isnan(fun) and np.isfinite(fun):
					funs[i,j]=fun
				else:
					funs[i,j]=-1
		funs[funs==-1] = np.nanmax(funs)

		contour(np.log(funs.T),z0s,z5s,heatOrContour="pix")

		where = np.where(funs==np.amin(funs))
		z0 = z0s[where[0][0]] ; z5 = z5s[where[1][0]]
		print(where,z0,z5)

	x=minimize(dz,x0=[1,1,1],args=(z0,z5))
	print(x)
	#x=minimize(dz,x0=[1,1,1,1/z0,1/z5],args=(220,275))
	#print(x['x'][:3],1/x['x'][3:],x['fun'])

	# UPDATE MICROSCOPE OBJECT WITH FITTED
	for n in range(3):
		# RECALL: we fitted for C in "1/f = C*I^2", not "1/f = L*(C*I)^2", so the latterC = sqrt(formerC/L)
		microscope["CL"+str(n+1)].calibration = np.sqrt(x['x'][n]/microscope["CL"+str(n+1)].length)
		#microscope["PL"+str(i)].calibration = [0,np.sqrt(x_PLs['x'][i-1]/microscope["PL"+str(i)].length)]

	# INSERT A GUN LENS, CALIBRATED, TO GIVE US REQUIRED CONVERGENCE INTO CL1. (gun is parallel beam)
	z_CL1 = microscope["CL1"].position
	#f_GL = 1/(1/(z_CL1/2)+1/(z0-z_CL1/2)) # 1/f = 1/d1 + 1/d2, where GL is halfway between source and CL1, and focuses to z0 THIS WOULD BE TRUE FOR DIVERGENT SOURCE, BUT WE HAVE PARALLEL
	f_GL = z0-z_CL1/2 # for a starting-parallel beam, f is just where we want it to focus to
	microscope.insert(z_CL1/2,Lens(name="GL",strength=1,length=0))
	microscope["GL"].calibration_from_f_and_I(f_GL,1)

	# NEXT, FIT FOR OL1 CALIBRATION BASED ON z5 AND SAMPLE POSITION
	z_OL = microscope["objective"].position + microscope["OL1"].position
	z_sample = microscope["objective"].position + microscope["sample"].position
	f_OL = 1/(1/(z_OL-z5)+1/(z_sample-z_OL))
	microscope["OL1"].calibration_from_f_and_I(f_OL,microscope["OL1"].strength)


	for k,(lensA,lensB,zval,Icrit) in rundict.items(): # A,(CL2,CL3,0,0.09) --> "CL2 focuses into CL3 at 90mA when CL1 is zero"
		continue
		if microscope.get_element_position(lensA) > microscope.get_element_position(lensB):
			continue
		microscope[lensA].strength = Icrit
		microscope[lensB].strength = 0
		lensC = ["CL1","VOA","CL2","CL3"] ; del lensC[lensC.index(lensA)] ; del lensC[lensC.index(lensB)] ; lensC=lensC[0]
		microscope[lensC].strength = zval
		microscope.show(title=k+", "+lensA+" focuses into "+lensB)

	#sys.exit()

	# AND FINALLY, FIT FOR APERTURE DIAMETER (relative to initial beam parameters) BASED ON CURRENT SWEEP
	A,B,C = np.load("C1_vs_beamcurrent_fit.npy")
	C1s,beamcurrent = np.load("C1_vs_beamcurrent.npy").T
	#beamcurrent = beamcurrent[C1s<.375] ; C1s=C1s[C1s<.375]
	mask = np.zeros(len(C1s)) ; mask[C1s<.335]=1 ; mask[C1s>.445]=1
	beamcurrent = beamcurrent[mask==1] ; C1s=C1s[mask==1]
	beamcurrent/=np.amax(beamcurrent) #beamcurrent[np.argmin(np.absolute(C1s-.338))]
	#mask = np.zeros(len(C1s))
	#mask[beamcurrent>np.amax(beamcurrent)/2]=1		# FWHM
	#mask[beamcurrent>np.mean(beamcurrent)]=1
	#for i in range(len(mask)):						# FILL IN HOLES (if i'm below FWHM, but i have points above and below me which are, i'm probably the noise in the middle of the peak)
	#	if mask[i]==0 and np.sum(mask[:i])>0 and np.sum(mask[i:])>0:
	#		mask[i]=1
	#C1s = C1s[mask==0] ; beamcurrent = beamcurrent[mask==0]
	#beamcurrent/=np.amax(beamcurrent)
	#plot([C1s],[beamcurrent])
	# recall, in I_crit_from_2D() we said:		  r₀'-.θ₀  |
	# r₁ = r₀ + l θ₀ - l/f = r₀ + l θ₀ - l S I²		|   '-.|r₁
	# ∫ CCD(z,A) dA ∝ rᵥₒₐ²/r₁(z)²					|       '-.
	# f(z) = A/(B-C*z²)²							|  l        '-.z0
	# --> A = rᵥₒₐ², B --> r₀ + l θ₀, but now we think we know θ₀?
	# but we said "prortional to": ∫ CCD(z,A) dA ∝ rᵥₒₐ²/r₁(z)²: don't know scaling of CCD intensity between A and true rᵥₒₐ²
	# instead, let's fix our "15i" reference setting to max current?
	microscope["CL1"].strength = lookupStrengthsXML("S_Condensers/30mrad15iRef/C1 ConstW",filename="AS2restore_20260103.xml")
	print(microscope["CL1"].strength)
	r1 = microscope.propagate_ray()
	x,y,xt,yt,R,I = measureAtZ(microscope["condenser"].position+microscope["VOA"].position,rays=r1)
	microscope["VOA"].name = "VOA_d"
	def dz(C1s,radius,dz,scaling):
		new = microscope.copy()
		new.insert(microscope["VOA_d"].position+dz,Aperture(name="VOA",radius=radius))
		modelcurrent = []
		#microscope.show()
		for c1 in C1s:
			new["CL1"].strength = c1
			r1 = new.propagate_ray()
			modelcurrent.append( r1[-1,-1,columnByName('I')] )
		modelcurrent = np.asarray(modelcurrent)
		return modelcurrent/scaling #np.amax(modelcurrent[C1s<.365])
	guesses = ( abs(x)*.5, .78,.2 )
	#guesses = (9.966e-02 , 1.898e-01, .09)
	plot([C1s,C1s],[beamcurrent,dz(C1s,*guesses)],title="guesses")
	print("MSE",np.sum((beamcurrent-dz(C1s,*guesses))**2))
	res,err = curve_fit(dz,C1s,beamcurrent,p0=guesses)
	print("MSE",np.sum((beamcurrent-dz(C1s,*res))**2))

	C1s,beamcurrent = np.load("C1_vs_beamcurrent.npy").T
	beamcurrent/=np.amax(beamcurrent[mask==1]) #beamcurrent[np.argmin(np.absolute(C1s-.338))]
	modelcurrent = dz(C1s,*res)
	norm = beamcurrent[np.argmin(np.absolute(C1s-.338))]
	beamcurrent/=norm ; modelcurrent/=norm
	plot([C1s,C1s,[0,.338]],[beamcurrent,modelcurrent,[1,1]],title="fit"+str(res),markers=['','','-'],xlim=[0,.5],ylim=[0,4],filename="CLs.svg")
	microscope.insert(microscope["VOA_d"].position+res[1],Aperture(name="VOA",radius=res[0]))

	# RELOAD, UPDATE MODEL, SAVE OFF
	microscope2 = mic_load("macstem")

	for n in range(1,4):
		microscope2["CL"+str(n)].calibration = microscope["CL"+str(n)].calibration
	microscope2["OL1"].calibration = microscope["OL1"].calibration
	microscope2.insert(microscope["GL"].position,microscope["GL"])
	microscope2["VOA"].name = ""
	microscope2.insert(microscope["VOA"].position,microscope["VOA"])
	microscope2.save("macstem_calibratedCL")

main()
