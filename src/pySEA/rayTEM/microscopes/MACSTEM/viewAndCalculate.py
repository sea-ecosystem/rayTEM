import sys,os,time,glob
sys.path.insert(1,"../dev/src/")
from pySEA.rayTEM.elements import Source,columnByName,Lens
from pySEA.rayTEM.assemblies import load_microscope as mic_load
from pySEA.rayTEM.postprocessing import plot2D, measureAtZ, findPlanes, error_at_plane, error_at_position, error_dz, zFromFractional, closest_plane, diffraction_bundles_at_z
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tqdm import tqdm

import numpy as np
from scipy.optimize import minimize,brute

CUBE_PATH = "/media/qwe/Data/Various Code/rayTEM_resources/TWP20260506/MACSTEM Lens Sweeps/20260702/"
CUBE_PATH = "./"
SEA_PEARL_PATH = "../../../../../../../sea-pearl/TWP20260530/"

# LOAD THE CALIBRATED MICROSCOPE
f = "microscope"
#f = "codex_attempts_01/01_midpoint_best_3xO2"
#f = "codex_attempts_01/02_midpoint_closest_ccd_3xO2"
#f = "codex_attempts_01/03_midpoint_distributed_2xO2"
#f = "codex_attempts_01/04_wobble_best_common_plane_2xO2"
#f = "codex_attempts_01/05_wobble_near_ccd_parallel_2xO2"
#f = "codex_attempts_01/06_parallel_camera_1xO2"
#f = "codex_attempts_01/07_wobble_focal_priority_3xO2"
#f = "codex_attempts_01/08_balanced_compact_1xO2"
#f = "codex_attempts_01/09_common_plane_distributed_3xO2"
#f = "codex_attempts_01/10_empirical_75pct_3xO2"

#f = "codex_attempts/01_best_absolute_2xO2"
#f = "codex_attempts/02_best_bundle_3xO2"
#f = "codex_attempts/03_focus_priority_2xO2"
#f = "codex_attempts/04_focus_priority_3xO2"
##f = "codex_attempts/05_tight_bundle_2xO2"
#f = "codex_attempts/06_tight_bundle_3xO2"
#f = "codex_attempts/07_camera_priority_1xO2"
#f = "codex_attempts/08_camera_priority_3xO2"
#f = "codex_attempts/09_focal_control_1xO2"
#f = "codex_attempts/10_diffraction_control_2xO2"


microscope = mic_load(f) #; print(microscope)
# FRONT HALF / BACK HALF
micro = microscope[:"P1"] #; micro["gun"].size=(10,10) ; micro["VOA"].radius
scope = microscope["sample":] ; scope[0].insert(0,Source(size=(1e-4,1e-4),angle=(1e-3,1e-3),na_xy=(3,1),np_xy=(3,1)))

z_P4 = scope.get_element_position("P4")
z_CCD = scope.get_element_position("CCD")
#scope.insert(z_P4*3/4+z_CCD/4,Lens(name="DQCM",length=0,focal_length=22))
#print(repr(microscope))

# intended as a replacement for scipy.optimize.minimize, where we use guesses x0 AND some more randomly-generated guesses within bounds
def bminimize(fun,x0,bounds,attempts=10):
	best = minimize(fun,x0=x0,bounds=bounds) # try with passed bounds
	bounds = np.asarray(bounds) # bounds is [[minA,maxA],[minB,maxB],[minC,maxC]...]
	x0s = np.random.random(size=(attempts,len(x0))) * (bounds[:,1]-bounds[:,0]) + bounds[:,0]
	for x0 in tqdm(x0s):
		x0 = minimize(fun,x0=x0,bounds=bounds)
		if x0['fun'] < best['fun']:
			best = x0
	return best

# FUNCTION TO ESTABLISH HARDWARE CONNECTION, QUERY PHYSICAL MICROSCOPE FOR LENS PARAMETERS
hardware = None
def readFromHardware(event):
	global hardware
	if hardware is None:
		try:
			sys.path.insert(1,SEA_PEARL_PATH+"src")
			from pySEA.sea_pearl.microscope import Microscope
			hardware = Microscope(name="U100", vendor="nion")
			hardware.connect()
			#hardware.acquire_frame("ronchigram")
		except Exception as e:
			print("WARNING: No hardware connection established:",e)
			class fake:
				def __getitem__(self,key):
					return np.random.random()
				def __setitem__(self,key,val):
					pass
			hardware=fake()


	if hardware is not None:
		#for element in microscope.named_positions:
			#if hasattr(microscope[element],"strength") and "lens" in microscope[element].kind.lower():
			#if hasattr(hardware,element):
		for element in ["C"+str(n) for n in range(1,4)]+["P"+str(n) for n in range(1,5)]:
			value = hardware[element[0]+element[-1]] # "CL1" --> "C1" in sea-pearl
			print("query",element,"found",value)
			microscope[element].strength = value
			if "C" in element:
				n = int(element.replace("C",""))
				CL_stringvars[n-1].set(str(value))
			if "P" in element:
				n = int(element.replace("P",""))
				PL_stringvars[n-1].set(str(value))
		update_ray_diagram(event)

history = []
def pushToHardware(event):
	global history
	last = {}
	for element in ["C"+str(n) for n in range(1,4)]+["P"+str(n) for n in range(1,5)]:
		if "C" in element:
			n = int(element.replace("C",""))
			value = float(CL_stringvars[n-1].get())
		if "P" in element:
			n = int(element.replace("P",""))
			value = float(PL_stringvars[n-1].get())

		last[element[0]+element[-1]] = hardware[element[0]+element[-1]]
		hardware[element[0]+element[-1]] = value # "CL1" --> "C1" in sea-pearl
	history.append(last)

def unpush(event):
	global history
	for e,v in history[-1].items():
		hardware[e] = v
	del history[-1]

CSettings = {} ; PSettings={} ; CSetting = None ; PSetting = None
def collectPresets():
	from pySEA.rayTEM.xmlNion import lookupStrengthsXML,rootControlSettingValue
	xml_file = "AS2restore_20260103.xml"
	global CSettings,PSettings,CSetting,PSetting

	CondSettings,CSetting = rootControlSettingValue(level="C",path="S_Condensers",filename=xml_file)
	for CS in CondSettings:
		setting = {}
		controls = rootControlSettingValue(level="S",path="S_Condensers/"+CS,filename=xml_file)
		for n in range(1,4):
			if "C"+str(n)+" ConstW" not in controls:
				break
			setting["C"+str(n)] = lookupStrengthsXML("S_Condensers/"+CS+"/C"+str(n)+" ConstW",xml_file)
		else:
			CSettings[CS] = setting

	ProjSettings,PSetting = rootControlSettingValue(level="C",path="S_Projectors",filename=xml_file)
	for PS in ProjSettings:
		setting = {}
		controls = rootControlSettingValue(level="S",path="S_Projectors/"+PS,filename=xml_file)
		for n in range(1,5):
			if "P"+str(n) not in controls:
				break
			setting["P"+str(n)] = lookupStrengthsXML("S_Projectors/"+PS+"/PL"+str(n),xml_file)
		else:
			PSettings[PS] = setting

	setting_files = glob.glob("settings/*.json")
	for f in setting_files:
		name = f.replace(".json","").replace("settings/","")
		microscope.load_setting(name)
		if "condenser" in name:
			CSettings[name] = { "C"+str(n):microscope["C"+str(n)].strength for n in range(1,4) }
		if "projector" in name:
			PSettings[name] = { "P"+str(n):microscope["P"+str(n)].strength for n in range(1,5) }

	manual = CUBE_PATH+"PLs_manual.txt"
	if os.path.exists(manual):
		lines = [ l for l in open(manual,'r').readlines() if len(l)>0 and l[0]!="#" ]
		for l in lines:
			vals = [ float(c) for c in l.split() ]
			mag = (vals[-2]+vals[-1])/2
			PSettings["MANUAL_"+str(np.round(mag,1))] = { "P"+str(n+1):v/1000 for n,v in enumerate(vals[:4]) }

collectPresets()

def loadPresetCS(event):
	CS = dropdown_CS.get()
	for k,v in CSettings[CS].items():
		print(k,v)
		CL_stringvars[int(k.replace("C",""))-1].set(str(v))
		micro[k].strength = v
	update_ray_diagram(event)

def loadPresetPS(event):
	PS = dropdown_PS.get()
	for k,v in PSettings[PS].items():
		PL_stringvars[int(k.replace("P",""))-1].set(str(v))
		scope[k].strength = v
	update_ray_diagram(event)

lines = [] #; xlim = [0,1] ; ylim = [-1,1]
# FUNCTION FOR GUI PLOTTING
def update_ray_diagram(event):
	xlim = ax.get_xlim() ; ylim = ax.get_ylim() ; print(xlim,ylim)
	# read from GUI entry fields, these are source for truth!
	for n,CL in enumerate(CL_stringvars):
		micro["C"+str(n+1)].strength = float(CL.get())
	for n,PL in enumerate(PL_stringvars):
		scope["P"+str(n+1)].strength = float(PL.get())
	for var,key,attr in bonus_stringvars.values():
		if attr == "position":
			scope[-1].move(key,float(var.get()))
		else:
			setattr(scope[key],attr,float(var.get()))
	# first half: re-propagate, replot
	ax.cla()
	r1 = micro.propagate_ray()
	# use rotated reference frame with measureAtZ or you get hypotenuse effects!
	x,y,xt,yt,R,I = measureAtZ("sample",section=micro)#,reference_frame="rotated") #; xt=micro.convergence_angle()
	#x,y,_,_,R,I = measureAtZ("O1",section=micro,reference_frame="rotated")
	ele =  micro.named_positions ; ele["VOA"+"\nI="+str(I)]=ele["VOA"] ; del ele["VOA"] # manually add current to VOA annotation
	ele["O1"+"\nxt="+str(xt)]=ele["O1"] ; del ele["O1"] # manually add angle to CL1. uncomment O1 measureAtZ and add +"\nx="+str(x)

	print("ALPHA",micro.convergence_angle)
	print("dF",micro.focus_error(240,regenerate=False))
	print("I",micro.beam_current)


	plot2D(r1, zpts = ele, plt_ax=ax, sections=micro.named_sections)
	# use diffraction-rays angle at sample (first half) for sample source starting rays (second half)
	#scope["gun"].angle=(-xt/100,xt/100)
	# second half: re-propagate, replot. edit element/section annotation positions, since this section thinks it starts at zero!
	r2 = scope.propagate_ray() ; dz = microscope["objective"].position+microscope["sample"].position
	ele = scope.named_positions ; sec = scope.named_sections
	ele = { k+"\n"+str(getattr(scope[k],"strength","")):v+dz for k,v in ele.items() } ; sec = { k:[v[0]+dz,v[1]+dz] for k,v in sec.items() }
	zi = columnByName('z')
	r2[:,:,zi] += dz
	r2.boundary_ray[...,zi] += dz
	z_CCD = scope.get_element_position("CCD") + dz
	plot2D(r2, zpts = ele,plt_ax=ax,sections=sec)
	# infer new xlim,ylim, this is what the "home button" should return us to: https://stackoverflow.com/questions/70336467/keep-zoom-and-ability-to-zoom-out-to-current-data-extent-in-matplotlib-pyplot
	ax.relim() ; ax.autoscale() ; toolbar.update() ; toolbar.push_current()
	# then set xlim/ylim back to where they were
	ax.set_xlim(xlim) ; ax.set_ylim(ylim)
	#plt.xlim(xlim) ; plt.ylim(ylim)
	ret = diffraction_bundles_at_z(z_CCD,r2)
	x,y,xt,yt,R,I = measureAtZ(z_CCD,rays=scope.rays)
	label = "L="+str(np.round(ret["bundle_spread"]['x'],5))+\
		"\ndefocus="+str(np.round(ret["bundle_spread"]['x'],5))+\
			"\nR="+str(np.round(R*180/np.pi,5))
	ax.annotate(label,(z_CCD,0))
	# update canvas
	print("MICRO\n",repr(micro))
	print("SCOPE\n",repr(scope))

	fig.canvas.draw_idle()

def live_worker(stop_event):
	while not stop_event.is_set():
		thinking(text="LIVE")
		readFromHardware(None)
		update_ray_diagram(None)
# https://alexandra-zaharia.github.io/posts/how-to-stop-a-python-thread-cleanly/
def live(event,thread=[None,None]): # lists in function default args are persistent between executions of the function
	import threading
	print("THREAD STATE",thread)
	if thread[0] is None:
		print("STARTING LIVE THREAD")
		thread[1] = threading.Event()
		thread[0] = threading.Thread(target=live_worker, args=(thread[1],))
		thread[0].start()
	else:
		print("INTERRUPTING LIVE THREAD")
		thread[1].set()
		#thread[0].join() # don't join, or we hang. idk why
		thread[0]=None ; thread[1]=None

def thinking(text="THINKING"):
	ax.annotate(text,(ax.get_xlim()[0],ax.get_ylim()[1]),color="red",verticalalignment="top",fontsize=100)
	fig.canvas.draw() ; fig.canvas.flush_events()

# PROJECTOR_MODE: DETERMINES HOW WE DO OUR PROJECTOR FITTING FOR PLANE POSITIONS AND PLANE MAGNITUDES (SCALING)
# "diff"	- locks diff plane position, fits diff plane mag
# "image"	- locks image plane position, fits image plane mag
# "both"	- locks diff and image plane positions, fits diff plane mag
# "bundles"	- locks image plane position, fits diffraction ray bundle mag, which allows defocus of diffraction plane at CCD
PROJECTOR_MODE = "bundles"

prefit_params = {} ; last_fit = None
def log_prefit(func):
	global prefit_params,last_fit
	last_fit = func
	prefit_params = {"C"+str(n):micro["C"+str(n)].strength for n in range(1,4)} | {"P"+str(n):scope["P"+str(n)].strength for n in range(1,5)}

def retry_last_fit(event):
	for n in range(1,4):
		micro["C"+str(n)].strength = prefit_params["C"+str(n)]
	for n in range(1,5):
		scope["P"+str(n)].strength = prefit_params["P"+str(n)]
	last_fit(event,use_brute=True)

def update_rotation(event,use_brute=False):
	thinking()
	log_prefit(update_rotation)
	#global value_rotation
	degrees = float(gui_vars["rotation"].get())
	# WHAT DO WE WANT?
	# 1) a diffraction plane (where it currently is, in case our model calibration is slightly off)
	# 2) the rotation of that diffraction plane updated slightly
	# 3) the magnification of that diffraction plane held constant
	# CURRENT STATE, SO WE CAN HOLD PLANE POSITIONS, ROTATIONS, AND APPLY dMAG
	r1 = scope.propagate_ray()
	z_CCD = scope.get_element_position("CCD")
	planes = findPlanes(r1,"x") #['x']['diff' or 'image']['z' or 'M' or 'R' or 'p']
	z_im = planes['x']["image"]['z']	# findPlanes returns fractional coordinated. 1.4 is 40% of the way through element 1
	z_im = [ zFromFractional(r1[:,0,columnByName('z')],z) for z in z_im ] ; z_im = np.asarray(z_im)
	i = np.argmin(np.absolute(z_im-z_CCD)) ; z_im = z_im[i]
	p_im = { k:planes['x']["image"][k][i] for k in planes['x']["image"].keys() }
	z_df = planes['x']["diff"]['z']
	z_df = [ zFromFractional(r1[:,0,columnByName('z')],z) for z in z_df ] ; z_df = np.asarray(z_df)
	i = np.argmin(np.absolute(z_df-z_CCD)) ; z_df = z_df[i]
	p_df = { k:planes['x']["diff"][k][i] for k in planes['x']["diff"].keys() }
	bundles = diffraction_bundles_at_z(z_CCD, r1)
	_,_,_,_,R_init,_ = measureAtZ(z_CCD,section=scope)

	# ERROR FUNCTION FOR MINIMIZE: depends on "PLANE" functionality
	def dz(vals):
		# regardless of mode, we're fitting for PLs
		settings = { "P"+str(n+1):{"strength":v} for n,v in enumerate(vals) }
		deltas = []
		# PLANE POSITIONS
		if PROJECTOR_MODE in [ "diff", "both" ]: # "diff" - hard plane mode, required fixed plane location
			deltas += error_dz(scope,settings,{"diff":z_df})
		if PROJECTOR_MODE in [ "image", "both", "bundles" ]: # locks image plane, since this is what matters for the spectrometer
			deltas += error_dz(scope,settings,{"image":z_im})
		# LENS ROTATION (OPTIONAL)
		_,_,_,_,R,_ = measureAtZ(z_CCD,section=scope)
		deltas.append(R-(R_init+degrees*np.pi/180))
		# MAGNIFICATION CALCULATION (OPTIONAL)
		if lock_associated_params.get():
			if PROJECTOR_MODE == "image":
				planes = findPlanes(scope.rays,"x")
				zp = planes['x']["image"]['z']
				zp = [ zFromFractional(r1[:,0,columnByName('z')],z) for z in zp ]
				i = np.argmin(np.absolute(zp-z_CCD))
				deltas.append( p_im["M"] - planes['x']["image"]['M'][i] )
			if PROJECTOR_MODE in [ "diff", "both" ]:
				planes = findPlanes(scope.rays,"x")
				zp = planes['x']["diff"]['z']
				zp = [ zFromFractional(r1[:,0,columnByName('z')],z) for z in zp ]
				i = np.argmin(np.absolute(zp-z_CCD))
				deltas.append( p_df["M"] - planes['x']["diff"]['M'][i] )
			if PROJECTOR_MODE == "bundles":
				bun = diffraction_bundles_at_z(z_CCD, scope.rays)
				deltas.append( bundles["bundle_spread"]['x'] - bun["bundle_spread"]['x'] )
		return np.sqrt(np.sum(np.asarray(deltas)**2))

	# INITIAL GUESSES: current lens values
	x0 = [ scope["P"+str(n)].strength for n in range(1,5) ]
	bounds = [ [ 0,1.5] for n in range(4) ]
	x0 = minimize(dz,x0=x0,bounds=bounds)['x']
	print(x0)
	# update the GUI entry fields, these are source for truth!
	for n,PL in enumerate(PL_stringvars):
		PL.set(str(x0[n]))
	update_ray_diagram(event)


def update_magnification(event,use_brute=False):
	thinking()
	log_prefit(update_magnification)

	dmag = float(gui_vars["magnification"].get())
	# WHAT DO WE WANT?
	# 1) a diffraction plane (where it currently is, in case our model calibration is slightly off)
	# 2) the magnification of that diffraction plane, changed by the percentage given
	# 3) the rotation of that diffraction plane held constant
	# MORE DISCUSSION:
	# What "magnification of the diffraction plane" mean? If you go to the reference setting and wobble majorOL, you'll see the diffraction image and bragg disk sizes shrink and grow, suggesting the diffraction plane is not focused onto the CCD. Loosening the restriction on the position of the diffraction plane, we care about the size of the psuedo-diff-plane on the CCD.
	# meanwhile, an image plane is focused to the spectrometer detector, so we likely care about locking the position of the image plane

	# CURRENT STATE, SO WE CAN HOLD PLANE POSITIONS, ROTATIONS, AND APPLY dMAG
	r1 = scope.propagate_ray()
	z_CCD = scope.get_element_position("CCD")
	planes = findPlanes(r1,"x") #['x']['diff' or 'image']['z' or 'M' or 'R' or 'p']
	z_im = planes['x']["image"]['z']	# findPlanes returns fractional coordinated. 1.4 is 40% of the way through element 1
	z_im = [ zFromFractional(r1[:,0,columnByName('z')],z) for z in z_im ] ; z_im = np.asarray(z_im)
	i = np.argmin(np.absolute(z_im-z_CCD)) ; z_im = z_im[i]
	p_im = { k:planes['x']["image"][k][i] for k in planes['x']["image"].keys() }
	z_df = planes['x']["diff"]['z']
	z_df = [ zFromFractional(r1[:,0,columnByName('z')],z) for z in z_df ] ; z_df = np.asarray(z_df)
	i = np.argmin(np.absolute(z_df-z_CCD)) ; z_df = z_df[i]
	p_df = { k:planes['x']["diff"][k][i] for k in planes['x']["diff"].keys() }
	bundles = diffraction_bundles_at_z(z_CCD, r1)
	_,_,_,_,R_init,_ = measureAtZ(z_CCD,section=scope)

	# ERROR FUNCTION FOR MINIMIZE: depends on "PLANE" functionality
	def dz(vals):
		# regardless of mode, we're fitting for PLs
		settings = { "P"+str(n+1):{"strength":v} for n,v in enumerate(vals) }
		deltas = []
		# PLANE POSITIONS
		if PROJECTOR_MODE in [ "diff", "both" ]: # "diff" - hard plane mode, required fixed plane location
			deltas += error_dz(scope,settings,{"diff":z_df})
		if PROJECTOR_MODE in [ "image", "both", "bundles" ]: # locks image plane, since this is what matters for the spectrometer
			deltas += error_dz(scope,settings,{"image":z_im})
		# MAGNIFICATION CALCULATION
		if PROJECTOR_MODE == "image":
			planes = findPlanes(scope.rays,"x")
			zp = planes['x']["image"]['z']
			zp = [ zFromFractional(r1[:,0,columnByName('z')],z) for z in zp ]
			i = np.argmin(np.absolute(zp-z_CCD))
			deltas.append( p_im["M"]*(1+dmag/100) - planes['x']["image"]['M'][i] )
		if PROJECTOR_MODE in [ "diff", "both" ]:
			planes = findPlanes(scope.rays,"x")
			zp = planes['x']["diff"]['z']
			zp = [ zFromFractional(r1[:,0,columnByName('z')],z) for z in zp ]
			i = np.argmin(np.absolute(zp-z_CCD))
			deltas.append( p_df["M"]*(1+dmag/100) - planes['x']["diff"]['M'][i] )
		if PROJECTOR_MODE == "bundles":
			bun = diffraction_bundles_at_z(z_CCD, scope.rays)
			deltas.append( bundles["bundle_spread"]['x']*(1+dmag/100) - bun["bundle_spread"]['x'] )
		# LENS ROTATION (OPTIONAL)
		if lock_associated_params.get():
			_,_,_,_,R,_ = measureAtZ(z_CCD,section=scope)
			deltas.append(R-R_init)
		return np.sqrt(np.sum(np.asarray(deltas)**2))

	# INITIAL GUESSES: current lens values
	x0 = [ scope["P"+str(n)].strength for n in range(1,5) ]
	bounds = [ [ 0,1.5] for n in range(4) ]
	x0 = minimize(dz,x0=x0,bounds=bounds)['x']
	print(x0)
	# update the GUI entry fields, these are source for truth!
	for n,PL in enumerate(PL_stringvars):
		PL.set(str(x0[n]))
	update_ray_diagram(event)

def update_current(event,use_brute=False):
	thinking()
	log_prefit(update_current)

	#global value_current
	dcurrent = float(gui_vars["current"].get())
	# WHAT DO WE WANT?
	# 1) a post-CLs crossover (where it currently is, in case our model calibration is slightly off). this maintains OL focus
	# 2) the angles into this plane held constant
	# 2b) optionally: use position entering OL instead?
	# 3) the beam current (diameter passing through VOA, or just last ray's current) changed by the percentage requested

	# POSITION OF CROSSOVER AFTER CL3
	z_CL3 = micro.get_element_position("C3")
	z_OL1 = micro.get_element_position("O1")
	plane = closest_plane(micro,z_CL3+(z_OL1-z_CL3)/4,"diff")
	z = plane['z']
	# CONVERGENCE ANGLE EXITING OL, AND CURRENT
	z_pOL = z_OL1+micro["O1"].length+.001
	x,y,xt,yt,R,I = measureAtZ(z_pOL,section=micro)
	I = micro.rays.I_per_plane[-1] # 'I' from measureAtZ is (as of recently) the non-blocked rays' intensity. I_per_plane is calculated from smooth function transmitted_fraction during propagation

	targets = { z_pOL:{"xt":xt,"I":I*(1+dcurrent/100) } } ; was=I

	# ERROR FUNCTION FOR MINIMIZE, USE PRE-BUILT: checks for angle and current at plane closest to z, and checks distance from plane to desired z
	def dz(vals):
		settings = { "C"+str(n+1):{"strength":v} for n,v in enumerate(vals) }
		deltas = error_at_position(micro,settings,targets,absolute=True) + error_dz(micro,settings,{"diff":z})
		#deltas[0]*=scale_xt ; print(deltas)
		if not lock_associated_params.get(): # optionally ignore dx (may be higher current available if we allow changed conv angle)
			del deltas[0]
		#print(deltas)
		return np.sqrt(np.sum(np.asarray(deltas)**2))

	# INITIAL GUESSES: current lens values, or grid from CLcube
	bounds = [ [ 0,upper] for upper in [.4,1,1.5] ]
	tg = { 'z':z, "xt":xt, "I":I*(1+dcurrent/100) }
	if not lock_associated_params.get():
		del tg["xt"]
	x0 = condenser_guesses(tg)

	x0 = bminimize(dz,x0=x0,bounds=bounds)['x']

	x,y,xt,yt,R,I = measureAtZ(z,section=micro)
	I = micro.rays.I_per_plane[-1]
	print("was",was,"now",I,"targets",targets)

	# UPDATE GUI AND MODEL: update the GUI entry fields, these are source for truth!
	for n,CL in enumerate(CL_stringvars):
		CL.set(str(x0[n]))
	update_ray_diagram(event)

def update_convergence(event,use_brute=False):
	thinking()
	log_prefit(update_convergence)

	#global value_current
	dconvergence = float(gui_vars["convergence"].get())
	# WHAT DO WE WANT?
	# 1) a post-CLs crossover (where it currently is, in case our model calibration is slightly off). this maintains OL focus
	# 2) the beam current (diameter passing through VOA, or just last ray's current) held constant
	# 3) the angles into this plane adjusted
	# 3b) optionally: use position entering OL instead?

	# POSITION OF CROSSOVER AFTER CL3
	z_CL3 = micro.get_element_position("C3")
	z_OL1 = micro.get_element_position("O1")
	plane = closest_plane(micro,z_CL3+(z_OL1-z_CL3)/4,"diff")
	z = plane['z']
	# CONVERGENCE ANGLE EXITING OL, AND CURRENT
	z_pOL = z_OL1+micro["O1"].length+.001
	x,y,xt,yt,R,I = measureAtZ(z_pOL,section=micro)
	I = micro.rays.I_per_plane[-1] # 'I' from measureAtZ is (as of recently) the non-blocked rays' intensity. I_per_plane is calculated from smooth function transmitted_fraction during propagation
	targets = { z_pOL:{"xt":xt*(1+dconvergence/100),"I":I } } ; was=x

	# ERROR FUNCTION FOR MINIMIZE, USE PRE-BUILT: checks for angle and current at plane closest to z, and checks distance from plane to desired z
	def dz(vals):
		settings = { "C"+str(n+1):{"strength":v} for n,v in enumerate(vals) }
		deltas = error_at_position(micro,settings,targets,absolute=True) + error_dz(micro,settings,{"diff":z})
		# optionally ignore dI (may be new conv angle available if we allow changing current)
		if not lock_associated_params.get():
			del deltas[1]
		return np.sqrt(np.sum(np.asarray(deltas)**2))

	# INITIAL GUESSES: current lens values, or grid from CLcube
	bounds = [ [ 0,upper] for upper in [.4,1,1.5] ]
	tg = { 'z':z, "xt":xt*(1+dconvergence/100), "I":I }
	if not lock_associated_params.get():
		del tg["I"]
	x0 = condenser_guesses(tg)

	# MINIMIZATION
	x0 = bminimize(dz,x0=x0,bounds=bounds,attempts=40)['x']
	x,y,xt,yt,R,I = measureAtZ(z_pOL,section=micro)
	I = micro.rays.I_per_plane[-1] # 'I' from measureAtZ is (as of recently) the non-blocked rays' intensity. I_per_plane is calculated from smooth function transmitted_fraction during propagation
	print("was",was,"now",xt,"targets",targets)

	# UPDATE GUI AND MODEL: update the GUI entry fields, these are source for truth!
	for n,CL in enumerate(CL_stringvars):
		CL.set(str(x0[n]))
	update_ray_diagram(event)

last_projector_targets = None ; PLcube_err = []
def projector_guesses(targets):
	if not os.path.exists(CUBE_PATH+"diffM_mmp.npy"):
		return [ scope["P"+str(n)].strength for n in range(1,5) ]
	PL1 = np.load(CUBE_PATH+"PL1.npy") ; PL2 = np.load(CUBE_PATH+"PL2.npy")
	PL3 = np.load(CUBE_PATH+"PL3.npy") ; PL4 = np.load(CUBE_PATH+"PL4.npy")
	global last_projector_targets,PLcube_err
	if len(PLcube_err) == 0:
		if os.path.exists("tmp_memmap.npy"): # TODO and # all current keys are in last keys and all last keys are in current keys:
			PLcube_err = np.memmap("tmp_memmap.npy",mode='r',shape=(len(PL1),len(PL2),len(PL3),len(PL4)),dtype=np.float64)
		else:
			for k,v in targets.items(): # e.g. "diffz":239
				print("GETTING GUESS FROM CUBE. TARGETTING",k,v)
				print("load memmap")
				dz = np.memmap(CUBE_PATH+k+"_mmp.npy",mode="r",shape=(len(PL1),len(PL2),len(PL3),len(PL4)),dtype=np.float64)
				if len(PLcube_err)==0:
					print("create error memmap")
					PLcube_err = np.memmap("tmp_memmap.npy",mode='w+',shape=dz.shape,dtype=dz.dtype)
				if "M" in k:
					print("M!")
					for i in tqdm(range(len(PL1))):
						PLcube_err[i,:,:,:] += ((np.absolute(dz[i,:,:,:])-np.absolute(v))/v)**2
				elif "z" in k:
					print("z")
					for i in tqdm(range(len(PL1))):
						PLcube_err[i,:,:,:] += (dz[i,:,:,:]-v)**2
				else:
					print("else")
					for i in tqdm(range(len(PL1))):
						PLcube_err[i,:,:,:] += ((dz[i,:,:,:]-v)/v)**2
	i,j,k,l = np.where(PLcube_err == np.amin(PLcube_err)) ; i=i[0] ; j=j[0] ; k=k[0] ; l=l[0]
	print("FOUND PLS:",PL1[i], PL2[j], PL3[k], PL4[l])
	#print("WHICH YIELDS:","xt",convang[i,j,k],"z",dfocus[i,j,k],"I",current[i,j,k])
	return PL1[i], PL2[j], PL3[k], PL4[l]

def condenser_guesses(targets):
	if not os.path.exists(CUBE_PATH+"dfocus.npy"):
		return [ micro["C"+str(n)].strength for n in range(1,4) ]
	xt,z,I = [ targets.get(k,None) for k in ["xt","z","I"] ]
	print("GET GUESSES FROM CUBE. TARGETTING:","xt",xt,"z",z,"I",I)
	CL1 = np.load(CUBE_PATH+"CL1.npy") ; CL2 = np.load(CUBE_PATH+"CL2.npy") ; CL3 = np.load(CUBE_PATH+"CL3.npy")
	dfocus = np.load(CUBE_PATH+"dfocus.npy")+240
	convang = np.load(CUBE_PATH+"convang.npy")
	current = np.load(CUBE_PATH+"current.npy")
	err = (dfocus-z)**2
	if xt is not None:
		err += ((np.absolute(convang)-abs(xt))/xt)**2
	if I is not None:
		err += ((current-I)/I)**2
	i,j,k = np.where(err == np.amin(err)) ; i=i[0] ; j=j[0] ; k=k[0]
	print("FOUND CLS:",CL1[i], CL2[j], CL3[k])
	print("WHICH YIELDS:","xt",convang[i,j,k],"z",dfocus[i,j,k],"I",current[i,j,k])
	return CL1[i], CL2[j], CL3[k]

scoot_direction = -1
def scootOL2(event):
	global scoot_direction
	for n in range(1,5):
		scope["P"+str(n)].strength = 0
	r1 = scope.propagate_ray()
	planes = findPlanes(r1,"x") #['x']['diff' or 'image']['z' or 'M' or 'R' or 'p']
	z_OL2 = scope.get_element_position("O2")
	# FIRST, INFER FIRST SUBSEQUENT IMAGE PLANE POSITION
	zp = planes['x']["image"]['z']
	zp = [ zFromFractional(r1[:,0,columnByName('z')],z) for z in zp ] # findPlanes returns fractional coordinated. 1.4 is 40% of the way through element 1
	zp = np.asarray(zp) ; z_image = zp[zp>z_OL2][0] ; print("scootOL2, z_image",z_image)
	if z_OL2 < 1:
		scoot_direction = 1
	if z_OL2 > 17:
		scoot_direction = -1
	scope["objective"].move_element("O2",dz=1*scoot_direction)
	def dz(vals):
		settings = {"O2":{"calibration":vals[0]}} | {"P"+str(n):{"strength":0} for n in range(1,5) }
		targets = {"image":z_image}
		print("dz","settings",settings,"targets",targets)
		return np.asarray(error_dz(scope,settings,targets))**2
	x = minimize(dz,x0=scope["O2"].calibration)['x']
	update_ray_diagram(event)


#scope["gun"].size = np.asarray(scope["gun"].size)/20
#scope["gun"].angle = np.asarray(scope["gun"].angle)*10000
z_sample = microscope.get_element_position("sample")
z_OL2 = microscope.get_element_position("O2")
C,I,L = microscope["O2"].calibration , microscope["O2"].strength, microscope["O2"].length
if C is None:
	C=1
print(C,I,L)
K=I*C ; S = np.sin(K*L)
if_OL = K*S
#dz = -5
id2 = if_OL-1/(z_OL2-z_sample) # 1/d1 + 1/d2 = 1/f
print(z_OL2+1/id2)
#if_OL = id2+1/(z_OL2+dz-z_sample)
#C = np.sqrt(if_OL/L)/I # iF = (I*C)*(I*C)*L --> (I*C)^2 = iF/L --> C = sqrt(iF/L)/I
#scope["objective"].move("OL2",dz=dz)
#scope["OL2"].calibration = C
#print(C)



#scope["objective"].move("OL2",dz=-10)


def wobble_worker(stop_event):
	global scope
	t0 = time.time() ; period = 2 ; strength = .02
	#s0 = scope["OL2"].strength
	s0 = float(bonus_stringvars["O2"][0].get())
	while not stop_event.is_set():
		thinking(text="WOBBLING")
		dt = ((time.time()-t0)%period)/period # normalize 0-1
		#scope["OL2"].strength = s0+np.sin(dt*2*np.pi)*strength
		bonus_stringvars["O2"][0].set(str(s0+np.sin(dt*2*np.pi)*strength))
		#print("SCOPE",repr(scope))
		update_ray_diagram(stop_event)
	#scope["OL2"].strength = s0
	bonus_stringvars["O2"][0].set(str(s0))
# https://alexandra-zaharia.github.io/posts/how-to-stop-a-python-thread-cleanly/
def wobbler(event,thread=[None,None]): # lists in function default args are persistent between executions of the function
	import threading
	print("THREAD STATE",thread)
	if thread[0] is None:
		print("STARTING WOBBLER")
		thread[1] = threading.Event()
		thread[0] = threading.Thread(target=wobble_worker, args=(thread[1],))
		thread[0].start()
	else:
		print("INTERRUPTING LIVE THREAD")
		thread[1].set()
		thread[0]=None ; thread[1]=None



# SET UP THE GUI:
#  __________________
# |  ______________  |
# | | rays diagram | |
# | |______________| |
# |  [] [] [] [] []  | <-- buttons for syncing with scope, calculating lens
# |__________________|     parameters for rotation, convergence angle, etc


from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.pyplot as plt

window = tk.Tk()											# main GUI window
# FIGURE PANEL FOR RAY DIAGRAM
fig,ax=plt.subplots()										# fig object for ray diagram
canvas = FigureCanvasTkAgg(fig, master=window)				# tk canvas for fig
canvas.get_tk_widget().grid(row=0,column=0,sticky='nsew')
frame_tb = Frame(master=window) # frame required to "grid" toolbar: https://stackoverflow.com/questions/12913854/displaying-matplotlib-navigation-toolbar-in-tkinter-via-grid
frame_tb.grid(row=1,column=0,sticky="NSEW")
toolbar = NavigationToolbar2Tk(canvas,frame_tb)				# matplotib navigation toolbar: allows zoom etc
toolbar.update()

# BUTTONS:
frame_buttons = Frame(master=window)
frame_buttons.grid(row=2,column=0,sticky="NSEW")

for lbl,xy in [["CLs",(1,5)],["PLs",(2,5)]]+[ [str(n+1),(0,n+6)] for n in range(4) ]:
	lb = Label(text=lbl,master=frame_buttons)
	lb.grid(row=xy[0],column=xy[1])


# helper function: adds a button, calling "func", at grid position "xy", with text "text"
def add_button(func,xy,text):
	button = Button(master=frame_buttons,text=text)
	button.bind("<Button-1>",func)
	button.grid(row=xy[0],column=xy[1],sticky="NSEW")

# helper function: creates a tk.StringVar, links it to an entry field, places it at grid position "xy", accessible via gui_vars["name"]
gui_vars = {}
def add_entry(name,xy,default):
	value = StringVar(window) ; value.set(default)
	entry = Entry(master=frame_buttons,textvariable=value)
	entry.grid(row=xy[0],column=xy[1],sticky="NSEW")
	gui_vars[name]=value

add_button(update_ray_diagram,(1,0),"replot")
add_button(readFromHardware,(2,0),"update model from microscope")
add_button(pushToHardware,(3,0),"push to hardware")
add_button(unpush,(3,1),"undo push!")
add_button(live,(4,0),"live!")

add_button(loadPresetCS,(3,3),"load preset")
dropdown_CS = ttk.Combobox(frame_buttons,values = list(sorted(CSettings.keys()))) ; dropdown_CS.set("[predefined CS]")
dropdown_CS.grid(row=3,column=4,sticky="NSEW")
add_button(loadPresetPS,(3,5),"load preset")
dropdown_PS = ttk.Combobox(frame_buttons,values = list(sorted(PSettings.keys()))) ; dropdown_PS.set("[predefined PS]")
dropdown_PS.grid(row=3,column=6,sticky="NSEW")


add_button(update_current,(1,1),"dCurrent (%)")
add_entry("current",(1,2),"10.0")

add_button(update_convergence,(1,3),"dconvergence (%)")
add_entry("convergence",(1,4),"10.0")


CL_stringvars = []
for n in range(3):
	CL = StringVar(window)
	CL.set(str(micro["C"+str(n+1)].strength))
	CL_stringvars.append(CL)
	en = Entry(master=frame_buttons,textvariable=CL)
	en.grid(row=1,column=6+n,sticky="NSEW")

add_button(update_rotation,(2,1),"dRotation (degrees)")
add_entry("rotation",(2,2),"5.0")

add_button(update_magnification,(2,3),"dMagnification (%)")
add_entry("magnification",(2,4),"10.0")

lock_associated_params = tk.IntVar()
cbl = tk.Checkbutton(frame_buttons, text="Lock Ass. Param.", variable=lock_associated_params)
cbl.select()
cbl.grid(row=3,column=2)

add_button(retry_last_fit,(3,7),"retry (brute)")
add_button(wobbler,(3,8),"model wobble")

add_button(scootOL2,(4,7),"scootOL2")


PL_stringvars = []
for n in range(4):
	PL = StringVar(window)
	PL.set(str(scope["P"+str(n+1)].strength))
	PL_stringvars.append(PL)
	en = Entry(master=frame_buttons,textvariable=PL)
	en.grid(row=2,column=6+n,sticky="NSEW")

bonus_stringvars = {}
for lbl,key,attr,rc in [["O2","O2","strength",[4,1]]]:
	var = StringVar(frame_buttons)
	var.set(str(getattr(scope[key],attr)))
	en = Entry(master=frame_buttons,textvariable=var)
	lb = Label(text=lbl,master=frame_buttons)
	lb.grid(row=rc[0],column=rc[1])
	en.grid(row=rc[0],column=rc[1]+1,sticky="NSEW")
	bonus_stringvars[lbl]=[var,key,attr]

#if "DQCM" in scope.keys():
#	zDQCM = StringVar(frame_buttons)
#	zDQCM.set(str(scope["DQCM"].strength))
#	en = Entry(master=frame_buttons,textvariable=OL)
#	lb = Label(text="OL",master=frame_buttons)
#	lb.grid(row=3,column=6)
#	en.grid(row=3,column=7,sticky="NSEW")


#opt=tk.StringVar(window)			# value for entry field goes in a stringVar object
#					field=tk.Entry(master=master,textvariable=opt,width=cellWidth)	# text entry field object
#					field.grid(**packkwargs)
#					elementString.split(";")[2]
#					globalName=elementString.split(";")[2]
#					specialFormatHandler=""
def quit_me():
	window.quit()
	window.destroy()

window.rowconfigure(0,weight=1,uniform=str(window))
window.columnconfigure(0,weight=1,uniform=str(window))

window.protocol("WM_DELETE_WINDOW", quit_me)
window.mainloop()
