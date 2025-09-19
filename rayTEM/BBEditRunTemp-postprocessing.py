from .elements import columnByName
import numpy as np
from scipy.optimize import minimize,brute
import matplotlib.pyplot as plt
from matplotlib.cm import plasma as cmap

#import sys
#sys.path.insert(1,"../../../niceplot")
#from nicecontour import *


# Basic 2D plotting (along z, and in whatever axis you have chosen)
def plot2D(r1,axis="x",filename=""):
	plt.clf()
	# add rays to plot, with a range of colors
	linecolors=list( cmap(np.linspace(0,1,len(r1[0]))) )

	# loop through rays
	i,j=columnByName(axis),columnByName("z")
	for ys,xs,c in zip( r1[:,:,i].T , r1[:,:,j].T , linecolors ):
		plt.plot(xs,ys,linestyle="-",color=c,marker='')

	# add all image/diffraction planes
	planes=findPlanes(r1) ; ct=0 ; zs=r1[:,0,j]
	nplanes=len(planes[axis]["diff"]["z"])+len(planes[axis]["image"]["z"])
	for imdiff in ["diff","image"]:
		Z=planes[axis][imdiff]["z"]
		M=planes[axis][imdiff]["M"]
		ylims = [ np.amin(r1[:,:,i]) , np.amax(r1[:,:,i]) ]
		for m,z in zip(M,Z):
			ct+=1
			z=zFromFractional(zs,z)
			label=imdiff+" @ z="+str(np.round(z,3))+"\n M="+str(np.round(m,3))
			ls={"diff":"--","image":":"}[imdiff]
			plt.plot([z,z],ylims,linestyle=ls,color="k",marker='')
			plt.annotate(label,(z,ylims[1]*ct/nplanes))

	# display or save
	if len(filename)>0:
		plt.savefig(filename)
	else:
		plt.show()


# Basic 3D plotting, rays in 3D
def plot3D(r1,filename="",elev=None,azi=None,roll=None):
	plt.clf()
	# add rays to plot, with a range of colors
	linecolors=list( cmap(np.linspace(0,1,len(r1[0]))) )

	# prepare for 3D line-plot
	ax = plt.figure().add_subplot(projection='3d')

	# loop through rays
	i,j,k=columnByName("x"),columnByName("y"),columnByName("z")
	for r in range(len(r1[0])):
		xs,ys,zs=r1[:,r,i].T,r1[:,r,j].T,r1[:,r,k].T
		plt.plot(zs,xs,ys,linestyle="-",color=linecolors[r],marker='')

	# add all image/diffraction planes
	planes=findPlanes(r1) ; zs=r1[:,0,k]
	for imdiff in ["diff","image"]:
		Z=planes["x"][imdiff]["z"] #; print(len(Z),imdiff,"planes at",Z)
		M=planes["x"][imdiff]["M"]
		xlims = [ np.amin(r1[:,:,i]) , np.amax(r1[:,:,i]) ]
		ylims = [ np.amin(r1[:,:,j]) , np.amax(r1[:,:,j]) ]
		xsurf,ysurf=np.meshgrid(xlims,ylims)
		for m,z in zip(M,Z):
			z=zFromFractional(zs,z)
			zsurf=np.asarray([[z,z],[z,z]])
			c={"diff":"r","image":"g"}[imdiff]
			ax.plot_surface(zsurf,xsurf,ysurf,color=c,alpha=.3)

	ax.view_init(elev, azi, roll)

	# display or save off
	if len(filename)>0:
		plt.savefig(filename)
	else:
		plt.show()


# Finds image and diffraction planes based on the crossing of rays: 
# ~ Any two rays originating from the same point form an image plane when they recross
# ~ Any singular ray which began at zero angle finds a diffraction plane when crossing x=0 or y=0
# returns nested dicts with keys: ["x" or "y"]["image" or "diff"]["M" or "z"]
# positions ("z") are stored in fractional coordinates, i.e., "1.2" is "20% of the way between index 1 and 2". use zFromFractional to convert to true z positions
# magnification ("M") in real space is defined as: the ratio of final/original positions of the ray
# magnification ("M") for a diffraction plane is defined as: the ratio of final position vs starting angle of the ray
# this diffraction mag is also known as the "camera length" based on the small angle approximation: dx=L*theta
def findPlanes(rays):
	# Infer which rays we'll use for detecting the planes! we should not require the user to understand the above criteria (and pass them) nor should we make assumptions on how the user constructed their list of rays
	diffRayIndex=None ; imageRayIndices=[]
	x=columnByName("x") ; y=columnByName("y")
	xt=columnByName("xt") ; yt=columnByName("yt")

	# diffraction ray is the first ray emitted at zero angle (nonzero position!)
	for r in range(len(rays[0])):
		if rays[0,r,xt]==0 and rays[0,r,yt]==0 and \
				rays[0,r,x]!=0 and rays[0,r,y]!=0:
			diffRayIndex=r
			break
	else:
		diffRayIndex=None

	# image rays emit from the same point (non-zero position) at differing angles
	for r in range(len(rays[0])):
		if len(imageRayIndices)==1:
			rr=imageRayIndices[0]		# ignore rays where previously-selected ray...
			if rays[0,r,xt] == rays[0,rr,xt] or rays[0,r,yt] == rays[0,rr,yt]: # ...has the same angles
				continue
		imageRayIndices.append(r)
		if len(imageRayIndices)==2:
			break
	else:
		imageRayIndices=None
	#print("using diffRayIndex",diffRayIndex,rays[0,diffRayIndex])
	#print("using imageRayIndices",diffRayIndex,rays[0,imageRayIndices[0]])
	#print("and",diffRayIndex,rays[0,imageRayIndices[1]])

	# loop through elements, get start/end points of each ray, and do some basic "crossing" math		
	for ij,axis in enumerate(["x","y"]):
		Zi=[] ; Zd=[] ; Mi=[] ; Md=[]
		c=columnByName(axis) ; ct=columnByName(axis+"t")
		for i in range(1,len(rays)):
			if not imageRayIndices is None:
				ya0,yb0=rays[i-1,imageRayIndices,c]
				ya1,yb1=rays[i,imageRayIndices,c]
			if not diffRayIndex is None:
				yd0=rays[i-1,diffRayIndex,c]
				yd1=rays[i,diffRayIndex,c]
	
			# if the originally-parallel ray crosses zero, this is a diffraction plane
			if yd1==0:								#'-.   ____m    (y-y0)=m*(x-x0)
				Zd.append(i)						#    '-.   |    solve for x where y=0
			if yd0<0 and yd1>0 or yd0>0 and yd1<0:	#________'-.____x=-y0/m
				m=yd1-yd0 ; dz=-yd0/m				#   dz       '-.
				Zd.append(i-1+dz)					# I'm actually storing the fractional index of the crossover! 
				ma=ya1-ya0 ; mb=yb1-yb0				# "magnification" of the diffraction plane 
				ya=ya0+ma*dz ; yb=yb0+mb*dz			# comes from the *difference in position* 
				Md.append((ya-yb)/max(np.absolute(rays[0,:2,ct])))					# of the perpendicular/angular rays
				#print(axis,"crosses center between",i-1,"and",i,"m",m,"dz",dz)
	
			#print(axis,i,ya0,yb0,ya1,yb1)
			# If rays have crossed in x or y, there is an image plane between i-1 and i. See FultzHowe2013 Fig 2.9
			if ya1==yb1:							#'-.     /  (y-y0)=m*(x-x0)
				Zi.append(i)						#    '-./   solve for where
			if ( ( ya0>yb0 and ya1<yb1 ) or 		#      / '-. ya=yb
					( ya0<yb0 and ya1>yb1 ) ):		#     / ma*(x-xa0)+ya0=mb*(x-xb0)+yb0
				ma=ya1-ya0 ; mb=yb1-yb0				#    / 	https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_line_equations
				dz=(yb0-ya0)/(ma-mb)				# a=ma ; b=mb ; c=xa0 ; d=xb0
				Zi.append(i-1+dz)					# magnification of image plane is
				ya=ya0+ma*dz ; Mi.append(ya/rays[0,0,c]) # ratio of original size to current		
				#print(axis,"rays cross between",i-1,"and",i,"m",m,"dz",dz)

		if axis=="x":
			Zix=Zi ; Zdx=Zd ; Mix=Mi ; Mdx=Md
		else:
			Ziy=Zi ; Zdy=Zd ; Miy=Mi ; Mdy=Md

	return {"x":{"diff":{"z":Zdx,"M":Mdx},"image":{"z":Zix,"M":Mix}},
			"y":{"diff":{"z":Zdy,"M":Mdy},"image":{"z":Ziy,"M":Miy}}}

def zFromFractional(zs,z): # e.g. 1.2 is 20% of the distance through element index 1
	i,di=int(z),z-int(z) # 1.2 --> i=1, and di=0.2
	z0=zs[i] ; z1=zs[i+1]
	return z0+(z1-z0)*di

# Given the ability to 1) generate a section 2) propagate rays and 3) measure attributes of the propagated rays (e.g. location of planes and magnifications), we should be able to fit for variables (like lens strength) to achieve a desired result
# Desired result may be: position of an image/diffraction plane, magnification at that plane, angles coming in, or unbounded desirables like "maximize the magnitude" or "minimize the lens currents"
# 1) create the function "propagateAndCheck" which updates section > element > properties, propagates, and finds the planes
# 2) define an "error" quantity which scipy.optimize minimize or brute can try to minimize by perturbing the properties
# pass:
# r0 : initial list of rays. at least one needs to be normal angle (and not at position 0), and another pair emitted from the same point
# section : a microscope section object
# targets : a list of dicts for what we want, e.g. [{"plane":"image","z":6,"mag":3}] would mean "we want an image plane at z=6 with a magnification of 3"
# modifiable : dict of element index/parameter pairs. e.g. {1:"strength",3:"strength"} if I wish to allow lens at index 1 and 3 to have their strength varied
def fitForCrossover(r0,section,targets=[],modifiable=[],axis="x",prefer={},ignoreSigns=True,filename=""):

	# propoagateAndCheck below takes a list of values, so we need to "map" these to modifiable elements and the parameters within that element
	indices,eleKeys,vals=[],[],[]
	for i in modifiable.keys():
		k=modifiable[i] ; v=section.elements[i].kget(k)
		indices.append(i) ; eleKeys.append(k) ; vals.append(v)
	#print("indices",indices,"eleKeys",eleKeys,"vals",vals)

	# a function which sets passed values, propagates, finds the crossover location, and returns an "error" term to be minimized
	def propagateAndCheck(vals,passback="dz"):
		if isinstance(passback,list):
			passback=passback[0]

		# set each element's parameter based on the list of values passed
		for i,v in enumerate(vals):
			i,kk=indices[i],eleKeys[i]
			section.elements[i].kset(kk,v)

		# propagate the starting array through the (now-updated) section
		r1=section.propagate_ray(r0)

		# inspect the output: find all image and diffraction planes
		planes = findPlanes(r1)
		zs=r1[:,0,columnByName("z")]

		# our "error" defined by each metric in each target (e.g. checking if position z is off, or magnification is off)
		# NOTE: the default is to minimize the sum (rather than a more-conventional "mean squared error")
		deltas=[] ; mags=[]
		for target in targets:

			# SPECIFIED CROSSOVER LOCATION [ REQUIRED ]
			z_desired=target["z"]

			# find closest plane of the correct type
			Zi=planes[axis][ target["plane"] ]["z"]	# "fractional coordinates" of positions of all correct-type planes: target["plane"] is "image" or "diff"

			# plausible there isn't a plane of that type! (maybe the minimization algorith, pushed it out of range)
			if len(Zi)==0:
				deltas.append(10000) ; mags.append(0) #; signs.append(1)
				continue

			# find closest plane
			Zf=[ zFromFractional(zs,z) for z in Zi ]
			n=np.argmin( np.absolute(np.asarray(Zf)-z_desired) )

			# Error in position
			dz=abs(Zf[n]-z_desired)
			deltas.append(dz)

			# LOOP THROUGH THE REST OF THE CRITERIA
			for k in target.keys():
				if k=="z":
					continue

				# SPECIFIED MAGNIFICATION
				if k=="mag":
					Ms=planes[axis][ target["plane"] ]["M"]
					mags.append( Ms[n] )

					# special case, user has asked to maximize the magnitude
					if target["mag"]=="maximize":
						# we want to minimize the negative! and /10 so we don't prefer high mag with plane in the wrong position
						deltas.append(-1*abs(Ms[n])/10)
				
					# traditional: "set mag to a given value"
					else:	
						if ignoreSigns: # + or - is an image flip. we may or may not care
							Ms[n]=abs(Ms[n]) ; target["mag"]=abs(target["mag"])
						dM=abs(Ms[n]-target["mag"])
						deltas.append(dM)				

				# LENS STRENGTH	
				if k=="strength":
					strengths = np.asarray( [ v for v,k in zip(vals,eleKeys) if k=="strength" ] )
					#strength=np.mean(strengths)/10
					if target["strength"]=="minimize":
						strength=max(strengths)/10 #np.sum(strengths**2)
						deltas.append(strength)
					elif target["strength"]=="maximize":
						#strength=np.mean(strengths)/20
						strength=min(strengths)/10
						deltas.append(-1*strength)

			#print("USING",vals,"CROSSOVER",n," IS AT",Zf[n],"DZ",dz,"DM",dM,"mags",mags) #,"deltas",deltas,"signs",signs)
		#plotRays(r1)
		
		deltas=np.asarray(deltas)
		
		if passback=="r1": # allow passback of the raw propagated rays, which is useful for plotting post-fitting (hijack this same function to simply propagate and plot)
			return r1
		if passback=="dz": # raw deviation (ABOVE DELTAS FOR MAXIMIZE MAGNIFICATION ETC HAVE BEEN WRITTEN BASED ON THIS BEING USED)
			return np.sum(deltas)
		if passback=="dz2": # more conventional mean-squared-error (ABOVE DELTAS MAY NOT WORK WITH THIS)
			dzsq=np.sqrt(np.sum(deltas**2))
			return dzsq
		if passback=="M":
			return mags[0]

	# TODOS: THE ERROR DEFINITIONS MAY NOT BE QUITE RIGHT
	# e.g. /10 scaling for maximize/minimize is somewhat arbitrary

	#plotRays( propagateAndCheck(vals,passback="r1") )

	# scipy.optimize.minimize: may fail to converge because of non-linearities in the parameter space
	#x0=minimize(propagateAndCheck,x0=vals)["x"] #,method='trust-constr',options={"finite_diff_rel_step":[.1]*len(vals),"xtol":1e-12})

	# scipy.optimize.brute: should be better at finding global minima. also convenient for plotting heatmaps of the parameter space
	ranges=[[v*.5,v*3] for v in vals ] # TODO WHAT SHOULD THESE BE? (user should probably be allowed to pass this, or we infer from the actual microscopy itself)
	# we will wrap scipy.optimize.minimize to use as brute's "polish" function
	def mini(*args,**kwargs):	
		kwargs["bounds"]=ranges
		return minimize(*args,**kwargs)
	# run fitting. full_output only required if you want the contour plot. same for Ns
	x0,r,vals,residuals=brute(propagateAndCheck,ranges=ranges,Ns=100,full_output=True,args=["dz"],finish=mini)
	# heatmap of parameter space
	residuals[residuals==10000]=np.nan
	plt.clf()
	plt.imshow(residuals[::-1,:],extent=(np.amin(vals[1]),np.amax(vals[1]),np.amin(vals[0]),np.amax(vals[0])))
	plt.xlabel(eleKeys[1]+" "+str(indices[1]))
	plt.ylabel(eleKeys[0]+" "+str(indices[0]))
	where=np.where(residuals==np.nanmin(residuals)) ; print(where)
	plt.plot(vals[1,0,where[1][0]],vals[0,where[0][0],0],c="r",marker="o")
	plt.title(str(targets)+"\n residuals, best at "+str(x0))
	plt.cbar=plt.colorbar()
	if len(filename)>0:
		plt.savefig(filename+"a.png")
		plot2D( propagateAndCheck(x0,"r1") , filename=filename+"b.png")
	else:
		plt.show()
		# plot the final rays
		plot2D( propagateAndCheck(x0,"r1") )



