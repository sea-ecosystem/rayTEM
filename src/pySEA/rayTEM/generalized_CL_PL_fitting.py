import sys
sys.path.insert(1,"../TWP20260820/src/")
from pySEA.rayTEM import Source, Lens, Drift, Aperture, MicroscopeSection, Microscope, load_microscope, dz_focus_to, check_lengths, setkeys_to_settables_dict, load_guesses, columnByName
from scipy.optimize import minimize,curve_fit
import numpy as np
import matplotlib.pyplot as plt

microscope_file = "microscope"
#microscope_file = "microscope_bettersolution_4"

# creates a completely generic microscope model, consisting of basic sections and elements (every STEM has a condenser section, comprised of lenses, etc), but positions, strengths, calibrations will not be correct
def assemble():
	condenser = [ Source(np_xy=(2,2),na_xy=(0,0),angle=(1e-3,1e-3)), # rays over a grid of positions (linspace "size" and "np_xy") for a range of angles ("angle" and "na_xy")
					Lens(name="GL",position=50,strength=0,length=.1),
					Lens(name="C1",position=80, strength=.3,length=.1),
					Aperture(name="VOA",position=120, radius=.1),
					Lens(name="C2",position=150, strength=.5,length=.1),
					Lens(name="C3",position=190, strength=.5,length=.1) ]
	objective = [ Lens(name="O1",position=0, strength=.1,length=.1),	# positions restart, referenced to beginning of section
					Drift(name="sample",position=15),
					Lens(name="O2",position=30, strength=.1,length=.1) ]
	projector = [ Lens(name="P1",position=0, strength=.1,length=.1),
					Lens(name="P2",position=70, strength=.1,length=.1),
					Lens(name="P3",position=110, strength=.1,length=.1),
					Lens(name="P4",position=160, strength=.1,length=.1),
					Drift(name="CCD",position=350,length=100) ]
	condenser = MicroscopeSection(name="condenser",elements=condenser)
	objective = MicroscopeSection(name="objective",elements=objective,position=450)
	projector = MicroscopeSection(name="projector",elements=projector,position=500)

	microscope = Microscope(name="generic",sections=[condenser,objective,projector])
	#microscope.show()
	microscope.save("microscope")

# Loads real positions from a text file, re-saves microscope object. file should be rows of "lensname \t value \n" pairs
def adjust_element_positions():
	lines = open("positions.txt").readlines() # ASSUMED TO BE: lensname \t value \n
	positions = { l.split()[0] : float(l.split()[1]) for l in lines if len(l)>0 and l[0]!="#" }
	microscope = load_microscope(microscope_file)
	first = None
	for lens in microscope.keys():
		if lens in positions.keys():
			print(lens)
			if first is None:
				first = lens
				continue
			z = microscope.get_element_position(first)+positions[lens]-positions[first]
			microscope.move_element(lens,z=z,allow_unsafe=True)
	check_lengths(microscope)
	#microscope.show()
	microscope.save("microscope")
	#sys.exit()

# General premise for fitting CLs: C1 / VOA / C2 / C3 means you can't necessarily forwards-focus C1 into C2/C3 or back-focus C2/C3 into C1 directly. Instead, focus C2 into C3 and vice versa at a range of C1 values. this controls the convergence/divergence into C2, and the system of equations can be solved. We'll adjust GL.strength, C1.calibration, C2.calibration, C3.calibration, until all focusing conditions are met.
# Input file convention: rows of "whichLens","focusedToWhichLens","atValue","additionalLens=Value". e.g. "P2,P4,.56789,P1=500" uses P2 to focus to P4, at 567.89 mA, with P1 also set to 500 mA.
def fit_CLs(file_in,mode="fit"):
	lines = open(file_in).readlines()
	states = [ l.split(",") for l in lines if len(l)>0 and l[0]!="#" ]
	microscope = load_microscope(microscope_file)

	# FULL, USING GUESSES FROM ABOVE
	setKeys = {"GL":"strength", "C1":"calibration", "C2":"calibration", "C3":"calibration", "O1":"strength"}
	guesses = load_guesses("guesses.txt")
	x0 = [ guesses[k][v] for k,v in setKeys.items() ] ; bounds=None ; method="BFGS"
	if "iterative" in mode:
		x0 = [ getattr(microscope[k],v) for k,v in setKeys.items() ]
		bounds = [ [v*.99,v*1.01] for v in x0 ] ; method="Powell"

	cases = []
	for s in states: # e.g. C2,C3,0.45654321,C1=0.3
		target = {"name":str(s), "settables":{ "C"+str(n):{"strength":0} for n in range(1,4)}}
		if len(s)==3:
			L1,L2,v = s
		else:
			L1,L2,v,p = s
		# in all cases, L1's strength is a settable:
		target["settables"][L1] = { "strength":float(v) }
		# simple "backwards focus" case: "C3,C2,0.7654567654321" --> beam originating at C2 focuses to z5, and again to sample
		if int(L1[-1]) > int(L2[-1]):
			target["from"] = L2 ; target["to"] = "sample"
		# simple forwards case: "C2,C3,0.123454321" --> beam enters C2, focuses to C3, at 123.454321 mA
		else:
			target["from"] = 0 ; target["to"] = L2
		if len(s)==4:
			L3,v3 = p.split("=") ; v3 = float(v3)
			target["settables"][L3]={"strength":v3}
		# SPECIAL CASE: wobbling C1 controls angles (but not size) through VOA, so if wobbling C1 does not show a change in beam size at a particular C2 or C3, this actually means the VOA plane is being projected to the detector
		if target["from"] == "C1":
			target["from"] = "VOA"
		cases.append(target)
	print(cases)
	if "loaded" in mode:
		deltas = dz_focus_to([],{},cases,microscope[:"P1"],plotting=True)
		print("loaded deltas",np.sum(deltas))
	if "guesses" in mode:
		deltas = dz_focus_to(x0,setKeys,cases,microscope[:"P1"],plotting=True)
	if "fit" in mode or "iterative" in mode:
		#print(deltas)
		res = minimize(dz_focus_to,x0=x0,args=(setKeys,cases,microscope),bounds=bounds,method=method,options={"xtol":1e-10})#,method='L-BFGS-B')#,method='Nelder-Mead')
		print(res)
		#dz_focus_to(res['x'],setKeys,cases,microscope,plotting=True)

		microscope = load_microscope(microscope_file)
		settings = setkeys_to_settables_dict(res['x'],setKeys)
		microscope.update_with_settings(settings)
		microscope.save("microscope")

def fit_VOA(file_in,mode="fit"):
	# load data from CSV
	data = np.loadtxt(file_in,delimiter=",")
	C1s,Is = data.T ; Is/=np.amax(Is)
	# thresholding with window. Any values above threshold (and any spurious datapointss between) are excluded
	I_threshold = .6 ; mask = np.zeros(len(C1s))
	mask[Is>I_threshold]=1
	for i in range(len(C1s)):
		if mask[i]==0 and sum(mask[:i])>0 and sum(mask[i:])>0:
			mask[i]=1
	C1s = C1s[mask==0] ; Is = Is[mask==0] ; Is/=np.amax(Is)
	# Beam current vs lens current function
	microscope = load_microscope(microscope_file) ; z0 = microscope.get_element_position("VOA")
	#microscope["GL"].strength/=1.05 ; microscope["C1"].calibration*=1.05
	def I(C1s,r_VOA,dz_VOA):
		# the traced current through a MASKING aperture is a staircase in C1
		# (quantized by ray count), which starves curve_fit's gradients; the
		# smooth continuum estimate Aperture.transmitted_fraction is the
		# fitting surface instead, evaluated on the rays ARRIVING at the VOA
		microscope["VOA"].radius = r_VOA
		microscope.move_element("VOA",z=z0+dz_VOA)
		Is = []
		for C1 in C1s:
			microscope["C1"].strength = C1
			r1 = np.asarray(microscope.propagate_ray())
			at_voa = r1[np.argmin(np.abs(r1[:,0,columnByName("z")] - (z0+dz_VOA)))]
			Is.append(microscope["VOA"].transmitted_fraction(at_voa))
		Is = np.asarray(Is)/np.amax(Is)
		return Is
	# fitting
	if mode == "fit" or mode == "iterative":
		guesses = load_guesses("guesses.txt")
		guess = guesses["VOA"]["radius"] if mode=="fit" else microscope["VOA"].radius
		res,err = curve_fit(I,C1s,Is,p0=(guess,0))
		print(res,err)
	else:
		res = [microscope["VOA"].radius,0]
	I2 = I(C1s,*res)
	plt.plot(C1s,Is)
	plt.plot(C1s,I2)
	plt.title(str(res)+" "+str(np.sqrt(np.sum((Is-I2)**2))))
	plt.show()
	microscope.save("microscope")

# General premise for fitting PLs: P1 / P2 / P3 / P4, focusing each to each from CCD.
# Input file convention: rows of "whichLens","focusedToWhichLens","atValue","additionalLens=Value". e.g. "P2,P4,.56789,P1=500" uses P2 to focus to P4, at 567.89 mA, with P1 also set to 500 mA.
def fit_PLs(file_in,mode="fit"):
	lines = open(file_in).readlines()
	states = [ l.split(",") for l in lines if len(l)>0 and l[0]!="#" ]
	microscope = load_microscope(microscope_file)

	# FULL, USING GUESSES FROM ABOVE
	setKeys = {"O2":"strength", "P1":"calibration", "P2":"calibration", "P3":"calibration", "P4":"calibration"}
	guesses = load_guesses("guesses.txt")
	if mode == "fit":
		x0 = [ guesses[k][v] for k,v in setKeys.items() ]
		bounds=None ; method="BFGS"
	else:
		x0 = [ getattr(microscope[k],v) for k,v in setKeys.items() ]
		bounds = [ [v*.99,v*1.01] for v in x0 ] ; method="Powell"

	# length also matters for PLs, so initialize to something sane (and we'll fit for it later)
	if mode == "fit":
		for n in range(1,5):
			PL = "P"+str(n)
			L = microscope[PL].length ; L_new = guesses[PL]["length"]
			microscope.adjust_element_length(PL,L_new)
			microscope.move_element(PL,dz=-guesses[PL]["length"]/2+L/2) # LENS POSITION SHOULD BE OPTICAL CENTER?

	cases = []
	for s in states: # e.g. P2,P4,0.56789,P1=0.500
		target = {"name":str(s), "settables":{ "P"+str(n):{"strength":0} for n in range(1,5)}}
		if len(s)==3:
			L1,L2,v = s
		else:
			L1,L2,v,p = s
		# in all cases, L1's strength is a settable:
		target["settables"][L1] = { "strength":float(v) }
		# simple "backwards focus" case: "C3,C2,0.6789876" --> beam originating at C2 focuses to z5, and again to sample
		if int(L1[-1]) > int(L2[-1]):
			target["from"] = L2 ; target["to"] = "CCD"
		# simple forwards case: "C2,C3,0.123454321" --> beam enters C2, focuses to C3, at 123.454321 mA
		else:
			target["from"] = 0 ; target["to"] = L2
		if len(s)==4:
			L3,v3 = p.split("=") ; v3 = float(v3)
			target["settables"][L3]={"strength":v3}
		cases.append(target)
	print(cases)

	# use truncated scope for all previewing and fitting
	if "loaded" in mode:
		deltas = dz_focus_to([],{},cases,microscope["sample":],plotting=True)
		print("loaded deltas",np.sum(deltas))
	if "guesses" in mode:
		deltas = dz_focus_to(x0,setKeys,cases,microscope["sample":],plotting=True)
	if "fit" in mode or "iterative" in mode:
		res = minimize(dz_focus_to,x0=x0,args=(setKeys,cases,microscope["sample":]),bounds=bounds,method=method)#,method='L-BFGS-B')#,method='Nelder-Mead')
		print(res)
		#dz_focus_to(res['x'],setKeys,cases,microscope,plotting=True)

		#microscope = load_microscope("microscope")
		settings = setkeys_to_settables_dict(res['x'],setKeys)
		microscope.update_with_settings(settings)
		microscope.save("microscope")

def fit_rotation(file_in,mode="fit"):
	lines = open(file_in).readlines()
	RpA = { l.split(',')[0] : float(l.split(",")[1]) for l in lines  if len(l)>0 and l[0]!="#" } # radians per amp
	microscope = load_microscope(microscope_file)
	for n in range(1,5):
		PL = "P"+str(n)
		C_new,L_new = microscope[PL].get_C_L_from_rotation_at_I(.25,RpA[PL]*.25)
		L = microscope[PL].length
		microscope.adjust_element_length(PL,L_new)
		microscope.move_element(PL,dz=-L_new/2+L/2) # LENS POSITION SHOULD BE OPTICAL CENTER
		microscope[PL].calibration = C_new

	microscope.save("microscope")


assemble()
adjust_element_positions()
#fit_CLs("CLs_critical.csv",mode="guesses")

# ITERATIVE FITTING OF CL CALIBRATIONS AND VOA:
# BFGS does well for course refinement of CL calibrations, but a bad job dialing it in. bounded Powell does a better job dialing it in, but needs tight bounds.
# moving the VOA (as possibly required for VOA fitting) means CL calibrations need to be re-checked, since rays propagated from the VOA are used for calibrating CLs.
# SO: BFGS (mode=fit), then a few Powells (mode=iterative), then VOA, and repeat. view the result with mode=loaded

# ITERATIVE FIT
fit_CLs("CLs_critical.csv",mode="fit")
fit_VOA("C1_vs_beamcurrent.csv",mode="fit")
for i in range(10):
	fit_CLs("CLs_critical.csv",mode="iterative")
fit_VOA("C1_vs_beamcurrent.csv",mode="iterative")
for i in range(10):
	fit_CLs("CLs_critical.csv",mode="iterative")
# PREVIEWS
fit_VOA("C1_vs_beamcurrent.csv",mode="loaded")
fit_CLs("CLs_critical.csv",mode="loaded")

# feedback from codex review: if focus-position is wrong (but we're measuring focus at a fixed position z), then should we be finding the crossover and using that for our delta?

# PROJECTOR FITTING: similarly iterative: fit calibrations using assumed sane lengths, actually fit lengths based on rotations which might fork up the focus calibrations slightly, repeat calibrations using updated lengths
#fit_PLs("PLs_critical.csv",mode="guesses")
fit_PLs("PLs_critical.csv",mode="fit")
#fit_PLs("PLs_critical.csv",mode="loaded")
for i in range(10):
	fit_rotation("rotations.csv",mode="fit")
	for j in range(20):
		fit_PLs("PLs_critical.csv",mode="iterative")
fit_PLs("PLs_critical.csv",mode="loaded")

#fit_CLs("CLs_critical.csv",mode="loaded")
#fit_PLs("PLs_critical.csv",mode="loaded")

# SAVE OFF CALIBRATIONS
microscope = load_microscope(microscope_file)
microscope.save_as_calibration("condensers", {"C1":"calibration", "VOA":["position","radius"], "C2":"calibration", "C3":"calibration","GL":"strength","O1":"strength"})
microscope.save_as_calibration("projectors", {"P1":["calibration","length"], "P2":["calibration","length"], "P3":["calibration","length"], "P4":["calibration","length"],"O2":"strength"})

# CHECK RELOADING:
#assemble()
#adjust_element_positions()
#microscope = load_microscope("microscope")
#microscope.load_calibration("condensers")
#microscope.load_calibration("projectors")
#microscope.save("microscope")
#fit_CLs("CLs_critical.csv",mode="loaded")
#fit_PLs("PLs_critical.csv",mode="loaded")

