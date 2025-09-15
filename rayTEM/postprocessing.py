from .elements import columnByName
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.cm import plasma as cmap

def plotRays(r1,axis="x"):
	Ys=list( r1[:,:,columnByName(axis)].T )
	zs=r1[:,0,columnByName("z")]
	Xs=[ zs ]*len(Ys)
	linestyles=["-"]*len(Ys)
	colors=list( cmap(np.linspace(0,1,len(Ys))) )

	Xs.append( zs ) ; Ys.append(zs*0) ; linestyles.append("-") ; colors.append("k")	

	Zdx,Zdy,Zix,Ziy=findPlanes(r1)
	if axis=="x":
		Zd,Zi=Zdx,Zix
	else:
		Zd,Zi=Zdy,Ziy

	for i in Zd:
		i,di=int(i),i-int(i)
		z0=zs[i] ; z1=zs[i+1]
		z=z0+(z1-z0)*di
		Xs.append([z,z]) ; Ys.append([-1,1]) ; linestyles.append(":") ; colors.append("k")
	
	for i in Zi:
		i,di=int(i),i-int(i)
		z0=zs[i] ; z1=zs[i+1]
		z=z0+(z1-z0)*di
		Xs.append([z,z]) ; Ys.append([-1,1]) ; linestyles.append("--") ; colors.append("k")
	
	for x,y,l,c in zip(Xs,Ys,linestyles,colors):
		plt.plot(x,y,linestyle=l,color=c,marker='')
	plt.show()
		
def findPlanes(rays,imageRayIndices=None,diffRayIndex=0):
	if imageRayIndices is not None:
		irays=rays[:,imageRayIndices,:] # zslice,whichRay,[x,xt,y,yt,...]
	else:
		irays=rays
	
	for ij,axis in enumerate(["x","y"]):
		Zi=[] ; Zd=[]
		c=columnByName(axis)
		for i in range(1,len(rays)):
			ya0,yb0=rays[i-1,:,c]
			ya1,yb1=rays[i,:,c]
			yd0=rays[i-1,diffRayIndex,c]
			yd1=rays[i,diffRayIndex,c]
	
			# if the originally-parallel ray crosses zero, this is a diffraction plane
			if yd1==0:								#'-.   ____m    (y-y0)=m*(x-x0)
				Zdx.append(i)						#    '-.   |    solve for x where y=0
			if yd0<0 and yd1>0 or yd0>0 and yd1<0:	#________'-.____x=-y0/m
				m=yd1-yd0 ; dz=-yd0/m				#   dz       '-.
				Zd.append(i-1+dz)					# I'm actually storing the fractional index of the crossover! 
				#print(axis,"crosses center between",i-1,"and",i,"m",m,"dz",dz)
	
			#print(axis,i,ya0,yb0,ya1,yb1)
			# If rays have crossed in x or y, there is an image plane between i-1 and i. See FultzHowe2013 Fig 2.9
			if ya1==yb1:							#'-.     /  (y-y0)=m*(x-x0)
				Zi.append(i)						#    '-./   solve for where
			if ( ( ya0>yb0 and ya1<yb1 ) or 		#      / '-. ya=yb
					( ya0<yb0 and ya1>yb1 ) ):		#     / ma*(x-xa0)+ya0=mb*(x-xb0)+yb0
				ma=ya1-ya0 ; mb=yb1-yb0				#    / 	https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_line_equations
				dz=(yb0-ya0)/(ma-mb)				# a=ma ; b=mb ; c=xa0 ; d=xb0
				Zi.append(i-1+dz)
				#print(axis,"rays cross between",i-1,"and",i,"m",m,"dz",dz)

		if axis=="x":
			Zix=Zi ; Zdx=Zd
		else:
			Ziy=Zi ; Zdy=Zd

	return Zdx,Zdy,Zix,Ziy

# How does this work?
# We need a "update-parameters-and-propagate" function
# we need to inspect the output from that function, to find where the crossover IS
# and we need to plug these into a least-squares minimization algorithm 
# this will "iterate until the crossover is in the correct place"
# guesses expects a dict such as {"0_strength":2} where the initial value is the index to the element, then the kwarg to set in the element, then the fitting process can loop through updatable elements, and update them (in-place)
def fitForCrossover(r0,section,target,guesses,plane="image",axis="x"):
	# establish an ordering for arguments in propagateAndcheck, by saving off element 
	indices,eleKeys,vals=[],[],[]
	for k in guesses.keys():
		i,kk = k.split("_") ; i=int(i)	# i = index of the element, kk = element's variable name 
		v=guesses[k]					# v = value to set it to
		indices.append(i) ; eleKeys.append(kk) ; vals.append(v)
	# a function which sets passed values, propagates, and finds the crossover location
	def propagateAndCheck(vals,passback="dz2"): 
		for i,v in enumerate(vals):
			i,kk=indices[i],eleKeys[i]
			section.elements[i].kset(kk,v)
		# propagate the starting array through the (now-updated) section
		r1=section.propagate_ray(r0)
		# inspect the output: 
		Zdx,Zdy,Zix,Ziy = findPlanes(r1) # IMPLICIT ASSUMPTION: RAY 0 IS THE DIFFRACTION ARRAY WHICH LEFT PARALLEL
		Zs={"imagex":Zix,"imagey":Ziy,"diffx":Zdx,"diffy":Zdy}[plane+axis] # indices
		zs=r1[:,0,columnByName("z")]
		crossovers=[]
		for i in Zs:	# e.g. 1.2 for the point between indices 1 and 2, 20% of the way
			i,di=int(i),i-int(i)	# e.g. 1, and 0.2
			z0=zs[i] ; z1=zs[i+1]
			z=z0+(z1-z0)*di
			crossovers.append(z)
		crossovers=np.asarray(crossovers)
		closest=np.argmin(np.absolute(crossovers-z))
		closest=crossovers[closest]
		dz=target-closest

		print("USING",vals,"CROSSOVER IS AT",closest,"DZ",dz)
		#plotRays(r1)

		if passback=="r1":
			return r1
		if passback=="z":
			return closest
		if passback=="dz":
			return dz
		if passback=="dz2":
			return dz**2

	x0=minimize(propagateAndCheck,*vals)
	print(x0)

	plotRays( propagateAndCheck(x0["x"],passback="r1") )


