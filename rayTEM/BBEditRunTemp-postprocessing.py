from .elements import columnByName
import numpy as np
from scipy.optimize import minimize,brute
import matplotlib.pyplot as plt
from matplotlib.cm import plasma as cmap

import sys
sys.path.insert(1,"../../../niceplot")
from nicecontour import *

# Basic 2D plotting (along z, and in whatever axis you have chosen)
def plot2D(r1,axis="x",filename=""):
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
			label=imdiff+" @ z="+str(np.round(z,3))+" / M="+str(np.round(m,3))
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
		Z=planes["x"][imdiff]["z"] ; print(len(Z),imdiff,"planes at",Z)
		M=planes["x"][imdiff]["M"]
		xlims = [ np.amin(r1[:,:,i]) , np.amax(r1[:,:,i]) ]
		ylims = [ np.amin(r1[:,:,j]) , np.amax(r1[:,:,j]) ]
		xsurf,ysurf=np.meshgrid(xlims,ylims)
		for m,z in zip(M,Z):
			z=zFromFractional(zs,z)
			zsurf=np.asarray([[z,z],[z,z]])
			ax.plot_surface(zsurf,xsurf,ysurf,color='r',alpha=.1)


	ax.view_init(elev, azi, roll)



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
	for r in range(len(rays[0])):
		if rays[0,r,xt]==0 and rays[0,r,yt]==0 and \
				rays[0,r,x]!=0 and rays[0,r,y]!=0: # take the first ray emitted at zero angle
			diffRayIndex=r
			break
	for r in range(len(rays[0])):
		if rays[0,r,x] != rays[0,r,y]:	# ignore rays with x!=y
			continue
		if len(imageRayIndices)==1:
			rr=imageRayIndices[0]		# ignore rays where previously-selected ray...
			if rays[0,r,xt] == rays[0,rr,xt] or rays[0,r,yt] == rays[0,rr,yt]: # ...has the same angles
				continue
		if np.sign(rays[0,r,xt])==np.sign(rays[0,r,x]):
			continue
		if np.sign(rays[0,r,yt])==np.sign(rays[0,r,y]):
			continue
		imageRayIndices.append(r)
		if len(imageRayIndices)==2:
			break
	print("using diffRayIndex",diffRayIndex,rays[0,diffRayIndex])
	print("using imageRayIndices",diffRayIndex,rays[0,imageRayIndices[0]])
	print("and",diffRayIndex,rays[0,imageRayIndices[1]])


	# loop through elements, get start/end points of each ray, and do some basic "crossing" math		
	for ij,axis in enumerate(["x","y"]):
		Zi=[] ; Zd=[] ; Mi=[] ; Md=[]
		c=columnByName(axis) ; ct=columnByName(axis+"t")
		for i in range(1,len(rays)):
			ya0,yb0=rays[i-1,imageRayIndices,c]
			ya1,yb1=rays[i,imageRayIndices,c]
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
				print(axis,"crosses center between",i-1,"and",i,"m",m,"dz",dz)
	
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
				print(axis,"rays cross between",i-1,"and",i,"m",m,"dz",dz)

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

# How does this work?
# We need a "update-parameters-and-propagate" function
# we need to inspect the output from that function, to find where the crossover IS
# and we need to plug these into a least-squares minimization algorithm 
# this will "iterate until the crossover is in the correct place"
# An example of targets: 
# targets=[{"plane":"image","z":6,"mag":3}] # "we want an image plane at z=6 with a magnification of 3
# TODO: would be nice if we could, for example, set a target of "mag=maximize" instead of passing a number. but how? could put the magnification in the "deltas" list (and record a "sign" so we can maximize or minimize it. tell brute to minimize deltas*sign so it minimizes all terms except the ones we want maximized. but there's a problem of weighting? dz is important (i absolutely cannot have my plane in the wrong place!)
def fitForCrossover(r0,section,targets=[],modifiable=[],axis="x",prefer={},ignoreSigns=True):

	# propoagateAndCheck below takes a list of values, so we need to "map" these to modifiable elements
	indices,eleKeys,vals=[],[],[]
	for i in modifiable.keys():
		k=modifiable[i] ; v=section.elements[i].kget(k)
		indices.append(i) ; eleKeys.append(k) ; vals.append(v)
	print("indices",indices,"eleKeys",eleKeys,"vals",vals)

	# a function which sets passed values, propagates, and finds the crossover location
	def propagateAndCheck(vals,passback="dz"):
		if isinstance(passback,list):
			passback=passback[0]
		#passback="".join(passback)
		#if len(passback)==0:
		#	passback="dz2"
		# "*passed" then parsing below is required for brute minimize with "args=(passback)" used to generate contour for anything other than dz2
		#vals=passed[0] ; passback="".join(passed[1:])
		#if len(passback)==0:
		#	passback="dz2"
		print("vals",vals,"passback",passback)
		for i,v in enumerate(vals):
			i,kk=indices[i],eleKeys[i]
			section.elements[i].kset(kk,v)
		# propagate the starting array through the (now-updated) section
		r1=section.propagate_ray(r0)
		# inspect the output: find all image and diffraction planes
		planes = findPlanes(r1) # IMPLICIT ASSUMPTION: RAY 0 IS THE DIFFRACTION ARRAY WHICH LEFT PARALLEL
		zs=r1[:,0,columnByName("z")]
		# our "error" defined by each metric in each target (e.g. checking if position z is off, or magnification is off)
		deltas=[] ; mags=[] #; signs=[]
		for target in targets:
			z_desired=target["z"]
			# find closest plane of the correct type
			Zi=planes[axis][ target["plane"] ]["z"]	# "fractional coordinates" of positions of all correct-type planes: target["plane"] is "image" or "diff"
			if len(Zi)==0:
				deltas.append(10000) ; mags.append(0) #; signs.append(1)
				continue
			Zf=[]
			for i in Zi:	# e.g. 1.2 for the point between indices 1 and 2, 20% of the way
				i,di=int(i),i-int(i)	# e.g. 1, and 0.2
				if i+1>=len(zs):
					continue
				z0=zs[i] ; z1=zs[i+1]
				z=z0+(z1-z0)*di
				Zf.append(z)
			n=np.argmin( np.absolute(np.asarray(Zf)-z_desired) )
			# Error in position
			dz=abs(Zf[n]-z_desired)
			deltas.append(dz) #; signs.append(1)
			# Error in magnification
			dM=None
			Ms=planes[axis][ target["plane"] ]["M"]
			mags.append( Ms[n] )
			if "mag" in target.keys() and not isinstance(target["mag"],str):
				if ignoreSigns:
					Ms[n]=abs(Ms[n]) ; target["mag"]=abs(target["mag"])
				dM=abs(Ms[n]-target["mag"])
				deltas.append(dM) #; signs.append(1)
			if "mag" in target.keys() and target["mag"]=="maximize" and deltas[-1]!=10000:
				# e.g. dz=-.05 M=-4. this is better than dz=.1 M=-4 and better than dz=.05 M=3
				# we don't care about sign. so both are absoluted
				# and we want to maximize abs(M) while minimizing dz
				# so we want to minimize -abs(M), minimize dz
				# or maximize 1-dz**2. possibly with a scaling on dz to weight it stronger
				#deltas[-1]=-1*abs(Ms[n])*(1-10*deltas[-1]**2)
				deltas.append(-1*abs(Ms[n])/10)
				#passback="dz"
				#	if deltas[-1]<.001:
				#		deltas.append(abs(Ms[n])) #; signs.append(-.0001) ; dM=Ms[n]
				#else:
			if "strength" in target.keys() and target["strength"]=="minimize" and deltas[-1]!=10000:
				strengths = [ v for v,k in zip(vals,eleKeys) if k=="strength" ]
				strength=max(strengths)
				#deltas[-1]=strength*deltas[-1]
				deltas.append(strength)
			if "strength" in target.keys() and target["strength"]=="maximize" and deltas[-1]!=10000:
				strengths = [ v for v,k in zip(vals,eleKeys) if k=="strength" ]
				strength=min(strengths)
				#deltas[-1]=-1*strength*(1-10*deltas[-1]**2)
				deltas.append(-1*strength)

			#if "strength" in target.keys():
			#	if target["strength"]=="maximize":
			#		strengths = [ v for v,k in zip(vals,eleKeys) if k=="strength" ]
			#		deltas.append(min(strengths)/3) ; signs.append(-1)
			#for k in prefer.keys():
			#	if k=="strength":
			#		strengths = [ v for v,k in zip(vals,eleKeys) if k=="strength" ]
			#		if prefer[k]=="high":
			#			deltas.append(2-min(strengths)) # we want to "raise the lowest"
			#		else:
			#			deltas.append(max(strengths)) # we want to "lower the highest"
			#if abs(dz)<.1:
			print("USING",vals,"CROSSOVER",n," IS AT",Zf[n],"DZ",dz,"DM",dM,"mags",mags) #,"deltas",deltas,"signs",signs)
		#plotRays(r1)
		
		deltas=np.asarray(deltas)
		
		if passback=="r1":
			return r1
		#if passback=="z":
		#	return closest
		if passback=="dz":
			return np.sum(deltas)
		if passback=="dz2":
			dzsq=np.sqrt(np.sum(deltas**2)) # *np.asarray(signs))
			#if abs(dz)<.1:
			#	print(dzsq)
			return dzsq
		if passback=="M":
			return mags[0]

	# A FEW STRATEGIES:
	# 1) "we want a specific z and M". given a z and M, we propagate rays, find the location/magnification of the nearest plane, calculate a residual from dz and dM, and try to minimize this residual
	# scipy.optimize.brute sweeps the entire parameter space ("all combinations of lens strengths") minimizing this residual
	# brute also includes a "polishing" step: brute force "try all combinations" then "run gradient descent" to find the "absolute best" answer
	# 2) "we want a specific z and the best M available". 
	# I can use brute to sweep the parameter space for both, generate two heatmaps, and overlay them. 
	# but if i want to "minimize", i need a smoothish residual function? 
	# if my residual is simply sum( abs(dz) + -M ) then this is not smooth (dz is sharp at dz=0), but -M will serve to maximize M
	# one may also wish to apply scaling on M. i should not tolerate dz in the totally wrong place to increase M
	# and this residual function is rather abnormal. normally it is MSE: sqrt(sum(deltas**2)) which gives a higher weighting to larger deltas


	#plotRays( propagateAndCheck(vals,passback="r1") )

	#x0=minimize(propagateAndCheck,x0=vals)#,method='trust-constr',options={"finite_diff_rel_step":[.1]*len(vals),"xtol":1e-12})
	#x0=x0["x"]
	#x0=brute(propagateAndCheck,ranges=[[v*.5,v*2] for v in vals ])
	#targets[0]["mag"]="maximize"
	ranges=[[v*.5,v*3] for v in vals ]
	def mini(*args,**kwargs):
		kwargs["bounds"]=ranges
		return minimize(*args,**kwargs)
	x0,r,vals,residuals=brute(propagateAndCheck,ranges=ranges,Ns=100,full_output=True,args=["dz"],finish=mini)
	residuals[residuals==10000]=np.nan
	contour(residuals,vals[1,0,:],vals[0,:,0],xlabel="focal length, L2",ylabel="focal length, L1",zlabel="residual",aspect=1,title="residuals",heatOrContour="pix")
	#print(x0,r)
	#print(x0)
	#for v in x0:
	#	print("\n",v,"\n")
	

	if False:
		targets[0]["mag"]="maximize"
		_,_,vals,residuals=brute(propagateAndCheck,ranges=[[v*.5,v*3] for v in vals ],Ns=300,full_output=True,finish=None)#,args=("dz2"))
		_,_,_,mags=brute(propagateAndCheck,ranges=[[v*.5,v*3] for v in vals ],Ns=300,full_output=True,args=("M"),finish=None)
		
		#x0,r,vals,residuals=brute(propagateAndCheck,ranges=[[v*.5,v*2.5] for v in vals ],Ns=100,full_output=True)
	
		residuals[residuals==10000]=np.nan
		
		#contour(residuals,vals[1,0,:],vals[0,:,0],xlabel="focal length, L2",ylabel="focal length, L1",zlabel="residual",aspect=1,title="residuals",heatOrContour="pix")#,filename="../../replicateFHFig211.svg")
		#contour(mags,vals[1,0,:],vals[0,:,0],xlabel="focal length, L2",ylabel="focal length, L1",zlabel="magnification",aspect=1,title="magnification",heatOrContour="pix")
	
		mags[residuals>.1]=0
		contour(mags,vals[1,0,:],vals[0,:,0],xlabel="focal length, L2",ylabel="focal length, L1",zlabel="magnification",aspect=1,title="magnification",heatOrContour="pix")


	#plt.imshow(residuals[::-1,:],extent=(np.amin(vals[1]),np.amax(vals[1]),np.amin(vals[0]),np.amax(vals[0])))
	#plt.cbar=plt.colorbar()

	#plt.show()

	plotRays( propagateAndCheck(x0,"r1") )


