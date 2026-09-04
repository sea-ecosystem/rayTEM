import sys
sys.path.insert(1,"../dev/src/")
from pySEA.rayTEM import Source, Lens, Drift, Aperture, MicroscopeSection, Microscope, load_microscope, dz_focus_to, check_lengths, setkeys_to_settables_dict, load_guesses, columnByName, closest_plane
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
		print(microscope[:"P1"].propagate_ray())
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

# General premise for fitting VOA: measure and fit a beam current vs C1 curve
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
		microscope["VOA"].radius = r_VOA
		microscope.move_element("VOA",z=z0+dz_VOA)
		Is = []
		for C1 in C1s:
			microscope["C1"].strength = C1
			r1 = microscope.propagate_ray()
			I = microscope["VOA"].transmitted_fraction(r1.at_z(microscope.get_element_position("VOA")))
			Is.append(I)
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

# General premise for fitting PL lengths: measure beam rotation per amp for each lens. if focusing follows K^2 L and rotation follows K L, and K = I C, then you can adjust L and C simultaneously to preserve focal lengths at a given current while adjusting beam rotation
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

# fitting for PLs and O2 establishes an image plane position (switching to the spectroscopy DQCM mode (instead of 4DSTEM) focuses an image plane onto the detector along the dispersive axis (tightly-focused zero-loss beam came from a tightly-focused probe at the sample plane).
# this image plane can be preserved if the position and strength of O2 are simultaneously adjusted, but this will move the post-OL diffraction plane.
# How do we know where the post-OL diffraction plane should be? pick two projector settings which a different number of subsequent diffraction planes before the CCD. adjust OL until the positions of these settings' diffraction planes line up.
def adjustOL(settings_list):
	microscope = load_microscope(microscope_file)["sample":]
	microscope.insert(0,Source(angle=(1e-3,1e-3)))

	# infer current image plane (multiple settings ought to agree, but might not necessarily, so take their mean)
	z_images = []
	#z_P1 = microscope.get_element_position("P1") # nope. DON'T use this for image plane, because it's mid-PLs
	z_P4 = microscope.get_element_position("P4")/2+microscope.get_element_position("CCD")/2
	for s in settings_list:
		microscope.load_setting(s)
		#microscope.show()
		z_images.append( closest_plane(microscope,z_P4,"image")['z'] )
	z_image = np.mean(z_images)

	# error function: pass O2.calibration and a dz (from initial O2.position), calculate how far image plane has moved (should be zero), and distance between two settings' diffraction planes (should be the same)
	z0 = microscope.get_element_position("O2")
	def dz(vals):
		C,dz = vals
		microscope.move_element("O2",z=z0+dz)
		microscope["O2"].calibration = C
		z_images = [] ; z_diffs = []
		for s in settings_list:
			microscope.load_setting(s)
			#microscope.show()
			z_images.append( closest_plane(microscope,z_P4,"image")['z'] )
			z_diffs.append( closest_plane(microscope,z_P4,"diff")['z'] )
		# z0-mean(zi). and use std for diff: std is just dz/2 for 2 entries, but arbitrarily expandable for many settings
		return (z_image-np.mean(z_images))**2 + (2*np.std(z_diffs))**2

	x0 = minimize(dz,x0=(1,0))
	print(x0)

	if "DQCM" not in microscope.keys():
		microscope.insert(z_P4,Lens(name="DQCM",strength=0,length=.1))

	while True:
		for s in settings_list:
			microscope.load_setting(s)
			microscope.show()
		c = input("DQCM dz,K: ")
		if "q" in c:
			break
		dz,K = c.split(",") ; dz=float(dz) ; K=float(K)
		microscope.move_element("DQCM",dz=dz)
		microscope["DQCM"].strength=K

	microscope.save("microscope_OL")

#step = "fresh,fitCL,viewCL,guessPL,fitPL,viewPL,fitOL"
step = "viewCL,viewPL"
#step = "fitPL,viewPL"
#step = "fitOL"


if "fresh" in step:
	assemble()
	adjust_element_positions()

# ITERATIVE FITTING OF CL CALIBRATIONS AND VOA:
# BFGS does well for course refinement of CL calibrations, but a bad job dialing it in. bounded Powell does a better job dialing it in, but needs tight bounds.
# moving the VOA (as possibly required for VOA fitting) means CL calibrations need to be re-checked, since rays propagated from the VOA are used for calibrating CLs.
# SO: BFGS (mode=fit), then a few Powells (mode=iterative), then VOA, and repeat. view the result with mode=loaded
if "guessCL" in step:
	fit_CLs("CLs_critical.csv",mode="guesses")
if "fitCL" in step:
	# ITERATIVE FIT
	fit_CLs("CLs_critical.csv",mode="fit")
	fit_CLs("CLs_critical.csv",mode="loaded")
	fit_VOA("C1_vs_beamcurrent.csv",mode="loaded")
	fit_VOA("C1_vs_beamcurrent.csv",mode="fit")
	for i in range(10):
		fit_CLs("CLs_critical.csv",mode="iterative")
	fit_VOA("C1_vs_beamcurrent.csv",mode="iterative")
	for i in range(10):
		fit_CLs("CLs_critical.csv",mode="iterative")
if "viewCL" in step:
	# PREVIEWS
	fit_VOA("C1_vs_beamcurrent.csv",mode="loaded")
	fit_CLs("CLs_critical.csv",mode="loaded")

# feedback from codex review: if focus-position is wrong (but we're measuring focus at a fixed position z), then should we be finding the crossover and using that for our delta?

# PROJECTOR FITTING: similarly iterative: fit calibrations using assumed sane lengths, actually fit lengths based on rotations which might fork up the focus calibrations slightly, repeat calibrations using updated lengths
if "guessPL" in step:
	fit_PLs("PLs_critical.csv",mode="guesses")
if "fitPL" in step:
	fit_PLs("PLs_critical.csv",mode="fit")
	#fit_PLs("PLs_critical.csv",mode="loaded")
	for i in range(10):
		fit_rotation("rotations.csv",mode="fit")
		for j in range(20):
			fit_PLs("PLs_critical.csv",mode="iterative")
if "viewPL" in step:
	fit_PLs("PLs_critical.csv",mode="loaded")

# SAVE OFF CALIBRATIONS
#microscope = load_microscope(microscope_file)
#microscope.save_as_calibration("condensers", {"C1":"calibration", "VOA":["position","radius"], "C2":"calibration", "C3":"calibration","GL":"strength","O1":"strength"})
#microscope.save_as_calibration("projectors", {"P1":["calibration","length"], "P2":["calibration","length"], "P3":["calibration","length"], "P4":["calibration","length"],"O2":"strength"})

# CHECK RELOADING:
#assemble()
#adjust_element_positions()
#microscope = load_microscope("microscope")
#microscope.load_calibration("condensers")
#microscope.load_calibration("projectors")
#microscope.save("microscope")
#fit_CLs("CLs_critical.csv",mode="loaded")
#fit_PLs("PLs_critical.csv",mode="loaded")

if "fitOL" in step: # BEWARE, THIS BREAKS GENERALIZATION BECAUSE YOU MUST SPECIFY TWO MEASURED PROJECTOR SETTINGS
	adjustOL(["projectors_A","projectors_C"]) # see /media/qwe/Data/Various Code/rayTEM/TWP20260810/src/pySEA/rayTEM/microscopes/MACSTEM/OLs.py for the settings we used. also referred to as manul104 and manual165





