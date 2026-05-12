from .elements import columnByName
import numpy as np
from scipy.optimize import minimize,brute
import matplotlib.pyplot as plt
from matplotlib.cm import plasma as cmap

#import sys
#sys.path.insert(1,"../../../niceplot")
#from nicecontour import *


# Basic 2D plotting (along z, and in whatever axis you have chosen)
def plot2D(r1,axis="x",filename=None,zpts="",sections=None,xlims=None,ylims=None,title=None,plt_ax=None):
	if plt_ax is None:
		fig,ax = plt.subplots()
	else:
		ax = plt_ax
	# add rays to plot, with a range of colors
	linecolors=list( cmap(np.linspace(0,1,len(r1[0]))) )

	# loop through rays
	i,j=columnByName(axis),columnByName("z")
	for ys,xs,c in zip( r1[:,:,i].T , r1[:,:,j].T , linecolors ):
		ax.plot(xs,ys,linestyle="-",color=c,marker='',linewidth=1)

	# add all image/diffraction planes
	planes=findPlanes(r1,axes=axis) ; ct=0 ; zs=r1[:,0,j]
	#print(planes)
	nplanes=len(planes[axis]["diff"]["z"])+len(planes[axis]["image"]["z"])+len(zpts)
	if ylims is None:
		ylims = [ np.amin(r1[:,:,i]) , np.amax(r1[:,:,i]) ]
	for imdiff in ["diff","image"]:
		Z=planes[axis][imdiff]["z"]
		M=planes[axis][imdiff]["M"]
		for m,z in zip(M,Z):
			ct+=1
			z=zFromFractional(zs,z)
			label=imdiff+" @ z="+str(np.round(z,3))+"\n M="+str(np.round(m,3))
			ls={"diff":"--","image":"-."}[imdiff]
			ax.plot([z,z],ylims,linestyle=ls,color="k",marker='',linewidth=1)
			ax.annotate(label,(z,ylims[1]*ct/nplanes))

	# add arbitrary passed z positions
	if len(zpts)>0:
		for label in zpts.keys():
			z=zpts[label] ; ct+=1
			ax.plot([z,z],ylims,linestyle=":",color="k",marker='')
			ax.annotate(label,(z,ylims[1]*ct/nplanes))

	# add shading for sections, if passed
	if sections is not None:
		colors = 'gbr'*10
		for i,k in enumerate(sections.keys()):
			z1,z2 = sections[k]
			#print("FILL",z1,z2)
			ax.fill_between([z1,z2],[ylims[0],ylims[0]],[ylims[1],ylims[1]],color=colors[i],alpha=.1)
			ax.annotate(k,(z1,ylims[0]))

	if xlims is not None:
		ax.set_xlim(xlims)
	ax.set_ylim(ylims)

	if title is not None:
		ax.set_title(title)

	#if returnObjectOnly:
	#	return ax
	if plt_ax is not None:
		return

	#ax = plt.gca() ; axs=[ax]
	#fig = plt.gcf()
	#axs[0].set_facecolor("black")  # inside area of plot --> black
	#fig.set_facecolor("black")  # outside area of plot --> black
	#for s in ["bottom", "top", "left", "right"]:  # border lines around plot --> white
	#	axs[0].spines[s].set_color("white")
	#axs[0].xaxis.label.set_color("white")  # x axis label text --> white
	#axs[0].tick_params(axis="x", colors="white")  # x axis tick marks --> white
	#axs[0].yaxis.label.set_color("white")
	#axs[0].tick_params(axis="y", colors="white")
	#axs[0].title.set_color("white")
	#leg = axs[0].legend()  # retreive legend from the axes
	#for text in leg.get_texts():
	#	text.set_color("white")  # each line of text on the legend --> white
	#frame = leg.get_frame()
	#frame.set_facecolor("black")  # inside area of legend --> black
	#frame.set_edgecolor("white")  # legend border lines --> white
	#fig.set_size_inches((32,8))


	# display or save
	if filename is not None:
		fig.savefig(filename,transparent=True)
	else:
		plt.show()


# Basic 3D plotting, rays in 3D
def plot3D(r1,filename="",elev=None,azi=None,roll=None):
	plt.clf()
	fig=plt.figure()
	# add rays to plot, with a range of colors
	linecolors=list( cmap(np.linspace(0,1,len(r1[0]))) )

	planes=findPlanes(r1)
	nPlanes=len(planes["x"]["image"]["z"])+len(planes["x"]["diff"]["z"])

	grid=int(np.ceil(np.sqrt(nPlanes+1)))
	# prepare for 3D line-plot
	ax = fig.add_subplot(grid,grid,1,projection='3d')

	# loop through rays
	i,j,k=columnByName("x"),columnByName("y"),columnByName("z")
	for r in range(len(r1[0])):
		xs,ys,zs=r1[:,r,i].T,r1[:,r,j].T,r1[:,r,k].T
		plt.plot(zs,xs,ys,linestyle="-",color=linecolors[r],marker='')

	# add all image/diffraction planes to 3D plot
	zs=r1[:,0,k]
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

	# add scatterplot showing where rays cross each plane, as subplots
	ct=1
	for imdiff in ["diff","image"]:
		P=planes["x"][imdiff]["p"]
		for n,p in enumerate(P):
			ct+=1
			xs,ys=np.asarray(p).T
			ax = fig.add_subplot(grid,grid,ct)
			ax.scatter(xs,ys,color=linecolors,alpha=.1)
			ax.set_title(imdiff+" plane "+str(n+1))
			#for x,y,z in zip(xs,ys,linecolors):
			#	ax.scatter(.plot(x,y,marker='.',color=c)


	# display or save off
	if len(filename)>0:
		plt.savefig(filename)
	else:
		plt.show()

def plotSliceSeries(rays,N,M,filename=""):
	# plotting setup
	plt.clf()
	fig=plt.figure(dpi=100,figsize=(3*N,3*M))
	# rays will be added with a range of colors
	linecolors=list( cmap(np.linspace(0,1,len(rays[0]))) )

	# z where rays are logged, and z of the slices we wish to plot
	zrays=rays[:,0,columnByName("z")]
	zslices=np.linspace(0,zrays[-1]*.99,N*M)

	def positionAtZ(y0,y1,dz):
		m=y1-y0
		return y0+m*dz

	for n,z in enumerate(zslices):
		# infer bounding element indices where rays were logged
		for i in range(len(zrays)):
			if zrays[i]>z:
				break
		# get fractional distance between ray logged points
		z1,z2=zrays[i-1:i+1]
		dz=(z-z1)/(z2-z1)
		# get x and y positions for ray segment on either side of this slice
		x1s,x2s=rays[i-1:i+1,:,columnByName("x")]
		y1s,y2s=rays[i-1:i+1,:,columnByName("y")]
		# get x,y, positions here at z, inferred from the positions on either side
		xplt=[ positionAtZ(x1,x2,dz) for x1,x2 in zip(x1s,x2s) ]
		yplt=[ positionAtZ(y1,y2,dz) for y1,y2 in zip(y1s,y2s) ]

		ax = fig.add_subplot(N,M,n+1)
		ax.scatter(xplt,yplt,c=linecolors,marker='.',s=1)
		ax.set_title(str(np.round(z,3))+", "+str(i-1)+"<i<"+str(i)+", dz="+str(np.round(dz,3)))
	
	# display or save off
	if len(filename)>0:
		plt.savefig(filename)
	else:
		plt.show()

# Returns a dict for each axis, image vs diffraction planes, and the magnification and z position (NOTE: Z IS IN FRACTIONAL COORDINATES: 4.2 = 20% of the way through the 4th element)
def findPlanes4(rays,axes="x"):
	# FIRST, collapse rays down to the axis of interest. x,y,m
	x=columnByName("x") ; y=columnByName("y") ; xt=columnByName("xt") ; yt=columnByName("yt")
	rays_x = rays[:,:,x] ; rays_y = rays[:,:,y] ; rays_xt = rays[:,:,xt] ; rays_yt = rays[:,:,yt] ;
	if axes == "x":
		rays = rays_x ; rays_t = rays_xt
	if axes == "y":
		rays = rays_y ; rays_t = rays_yt
	if axes=="m":
		rays = np.sqrt(rays_x**2+rays_y**2)*np.sign(rays_x)
		rays_t = np.sqrt(rays_xt**2+rays_yt**2)*np.sign(rays_xt)

	# Infer which rays we'll use for detecting the planes! we should not require the user to understand the above criteria (and pass them) nor should we make assumptions on how the user constructed their list of rays
	diffRays=[] ; imageRays=[]

	returnable={}
	for imdiff in ["diff","image"]:
		returnable[imdiff]={}
		for p in ["z","M","R","p"]:
			returnable[imdiff][p]=[]

	# diffraction ray is the first ray emitted at zero angle (nonzero position!)
	n_rays = len(rays[0])
	for r in range(n_rays):
		# diff X first ray is: zero angle, nonzero x
		if len(diffRays)==0 and rays[0,r]!=0 and rays_t[0,r]==0:
				diffRays.append(r)
				for rr in range(n_rays):
					if np.all(rays[0,r]==-rays[0,rr]):
						diffRaysX.append(rr)
						break
		# second is same, but opposite x position. TWP 20260317 edit: or, just a different x position?? changing all "==-" to "!="
		if len(diffRays)==1 and rays[0,r]!=0 and rays_t[0,r]==0 and rays[0,r]!=rays[0,diffRays[0]]: # TWP 20260317 edit: or, just a different x position?? changing all "==-" to "!="
			diffRays.append(r)
		# image X first ray is: nonzero angle x, zero angle y, zero x, zero y
		if len(imageRays)==0 and rays_t[0,r]!=0 and rays[0,r]==0:
			imageRays.append(r)
		# second is same, but opposite x angle
		if len(imageRays)==1 and rays_t[0,r]!=0 and rays[0,r]==0 and rays_t[0,r]!=rays_t[0,imageRaysX[0]]:
			imageRays.append(r)
		if len(diffRays)==2 and len(imageRays)==2:
				break
	if ( "x" in axes and ( len(diffRaysX)!=2 or len(imageRaysX)!=2 )) or ( "y" in axes and ( len(diffRaysY)!=2 or len(imageRaysY)!=2 )):
		print("WARNING: diffraction and/or image rays could not be inferred by findPlanes(). no planes found")
		if len(diffRaysX)<2 and "x" in axes:
			print("diffraction rays (x2) in X: must be finite x, zero y, zero xt and yt")
		if len(diffRaysY)<2 and "y" in axes:
			print("diffraction rays (x2) in Y: must be finite y, zero x, zero xt and yt")
		if len(imageRaysX)<2 and "x" in axes:
			print("image rays (x2) in X: must be finite xt, zero yt, zero x and y")
		if len(imageRaysY)<2 and "y" in axes:
			print("image rays (x2) in Y: must be finite yt, zero xt, zero x and y")
		return returnable

	#print(diffRaysX,rays[0,diffRaysX[0],[x,xt]],rays[0,diffRaysX[1],[x,xt]])

	x=columnByName("x") ; y=columnByName("y")
	xt=columnByName("xt") ; yt=columnByName("yt")

	def whereCrossesZero(y0,y1): # returns relative position (0-1) of crossover,
		# if the originally-parallel ray crosses zero, this is a diffraction plane
		if y1==0:								#'-.   ____m    (y-y0)=m*(x-x0)
			dz=1								#    '-.   |    solve for x where y=0
		elif y0<0 and y1>0 or y0>0 and y1<0:	#________'-.____x=-y0/m
			m=y1-y0 ; dz=-y0/m					#   dz       '-.
		else:
			return None
		return dz
			#Zd.append(i-1+dz)					# I'm actually storing the fractional index of the crossover!
			#ma=ya1-ya0 ; mb=yb1-yb0				# "magnification" of the diffraction plane
			#ya=ya0+ma*dz ; yb=yb0+mb*dz			# comes from the *difference in position*
			#Md.append((ya-yb)/(ta-tb))			# for two rays starting at *different angles*

	def whereRaysCross(ya0,ya1,yb0,yb1):
		# If rays have crossed in x or y, there is an image plane between i-1 and i. See FultzHowe2013 Fig 2.9
		if ya1==yb1:							#'-.     /  (y-y0)=m*(x-x0)
			dz=1								#    '-./   solve for where
		elif ( ( ya0>yb0 and ya1<yb1 ) or 		#      / '-. ya=yb
				( ya0<yb0 and ya1>yb1 ) ):		#     / ma*(x-xa0)+ya0=mb*(x-xb0)+yb0
			ma=ya1-ya0 ; mb=yb1-yb0				#    / 	https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_line_equations
			dz=(yb0-ya0)/(ma-mb)				# a=ma ; b=mb ; c=xa0 ; d=xb0
		else:
			return None
		return dz
			#Zi.append(i-1+dz)					# magnification of image plane is
			#ya=ya0+ma*dz ; Mi.append(ya/rays[0,0,c]) # ratio of original size to current
			#print(axis,"rays cross between",i-1,"and",i,"m",m,"dz",dz)

	def positionAtZ(y0,y1,dz):
		m=y1-y0
		return y0+m*dz

	# "magnification" of the diffraction plane comes from the *difference in position* for two rays starting at *different angles*. AKA, "camera length" if we assume positionondetector=scatteringangle*cameralength
	# "magnification" of the image plane comes from the *difference in position* for two rays starting at *different positions*
	def magnification(ya0,ya1,yb0,yb1,dz,ta,tb):
		ya=positionAtZ(ya0,ya1,dz)
		yb=positionAtZ(yb0,yb1,dz)
		#print("magnification where",ya0,"-",ya1,"and",yb0,"-",yb1,"cross",ya-yb,"/",ta-tb)
		return (ya-yb)/(ta-tb)

	# rays "a" and "b", original positions
	#xa0,xb0=rays[0,imageRayIndices,x]
	#ya0,yb0=rays[0,imageRayIndices,y]
	#xta0,xtb0=rays[0,imageRayIndices,xt]
	#yta0,ytb0=rays[0,imageRayIndices,yt]

	for axis,diffRays,imageRays,xy,yx,xyt in zip(["x","y"],[diffRaysX,diffRaysY],[imageRaysX,imageRaysY],[x,y],[y,x],[xt,yt]):
		if axis not in axes:
			continue
		for i in range(1,len(rays)):
			# CHECK DIFFRACTION: where originally-parallel rays cross
			(xa1,xb1),(xa2,xb2)=rays[i-1:i+1,diffRays,xy]
			dz=whereRaysCross(xa1,xa2,xb1,xb2)
			if dz is not None:
				xta0,xtb0 = rays[0,imageRays,xyt] # magnification comes from starting angles of two non-parallel rays
				# Magnification comes from conversion of angle to position
				(xa1,xb1),(xa2,xb2)=rays[i-1:i+1,imageRays,xy]
				(ya1,yb1),(ya2,yb2)=rays[i-1:i+1,imageRays,yx]
				M=magnification(xa1,xa2,xb1,xb2,dz,xta0,xtb0)
				returnable[axis]["diff"]["z"].append( i-1+dz )
				returnable[axis]["diff"]["M"].append( M )
				returnable[axis]["diff"]["p"].append([])
				for r in range(len(rays[0])):
					xr=positionAtZ(*rays[i-1:i+1,r,x],dz)
					yr=positionAtZ(*rays[i-1:i+1,r,y],dz)
					returnable[axis]["diff"]["p"][-1].append( [xr,yr] )

			# CHECK IMAGE PLANE: where rays leaving the same place re-cross
			(xa1,xb1),(xa2,xb2)=rays[i-1:i+1,imageRays,xy]
			dz=whereRaysCross(xa1,xa2,xb1,xb2)
			if dz is not None:
				xa0,xb0 = rays[0,diffRays,xy] # magnification comes from change in scaling (position) of originally-parallel rays
				(xa1,xb1),(xa2,xb2)=rays[i-1:i+1,diffRays,xy]
				(ya1,yb1),(ya2,yb2)=rays[i-1:i+1,diffRays,yx]
				M=magnification(xa1,xa2,xb1,xb2,dz,xa0,xb0)
				returnable[axis]["image"]["z"].append( i-1+dz )
				returnable[axis]["image"]["M"].append( M )
				returnable[axis]["image"]["p"].append([])
				for r in range(len(rays[0])):
					xr=positionAtZ(*rays[i-1:i+1,r,x],dz)
					yr=positionAtZ(*rays[i-1:i+1,r,y],dz)
					returnable[axis]["image"]["p"][-1].append( [xr,yr] )

	return returnable




# Returns a dict for each axis, image vs diffraction planes, and the magnification and z position (NOTE: Z IS IN FRACTIONAL COORDINATES: 4.2 = 20% of the way through the 4th element)
def findPlanes(rays,axes="xy"):
	# Infer which rays we'll use for detecting the planes! we should not require the user to understand the above criteria (and pass them) nor should we make assumptions on how the user constructed their list of rays
	diffRaysX=[] ; diffRaysY=[]
	imageRaysX=[] ; imageRaysY=[]
	x=columnByName("x") ; y=columnByName("y")
	xt=columnByName("xt") ; yt=columnByName("yt")

	returnable={}
	for xy in ["x","y"]:
		returnable[xy]={}
		for imdiff in ["diff","image"]:
			returnable[xy][imdiff]={}
			for p in ["z","M","R","p"]:
				returnable[xy][imdiff][p]=[]

	# diffraction ray is the first ray emitted at zero angle (nonzero position!)
	n_rays = len(rays[0])
	for r in range(n_rays):
		# diff X first ray is: zero angle x and y, nonzero x, zero y
		if len(diffRaysX)==0 and rays[0,r,x]!=0 and rays[0,r,y]==0 and rays[0,r,xt]==0 and rays[0,r,yt]==0:
				diffRaysX.append(r)
				for rr in range(n_rays):
					if np.all(rays[0,r]==-rays[0,rr]):
						diffRaysX.append(rr)
						break
		# second is same, but opposite x position. TWP 20260317 edit: or, just a different x position?? changing all "==-" to "!="
		if len(diffRaysX)==1 and rays[0,r,x]!=0 and rays[0,r,y]==0 and \
			rays[0,r,xt]==0 and rays[0,r,yt]==0 and \
				rays[0,r,x]!=rays[0,diffRaysX[0],x]: # TWP 20260317 edit: or, just a different x position?? changing all "==-" to "!="
					diffRaysX.append(r)
		# diff Y first ray is: zero angle x and y, nonzero y, zero x
		if len(diffRaysY)==0 and rays[0,r,y]!=0 and rays[0,r,x]==0 and \
			rays[0,r,xt]==0 and rays[0,r,yt]==0:
				diffRaysY.append(r)
		# second is same, but opposite x position
		if len(diffRaysY)==1 and rays[0,r,y]!=0 and rays[0,r,x]==0 and \
			rays[0,r,xt]==0 and rays[0,r,yt]==0 and \
				rays[0,r,y]!=rays[0,diffRaysY[0],y]:
					diffRaysY.append(r)
		# image X first ray is: nonzero angle x, zero angle y, zero x, zero y
		if len(imageRaysX)==0 and rays[0,r,xt]!=0 and rays[0,r,yt]==0 and \
			rays[0,r,x]==0 and rays[0,r,y]==0:
				imageRaysX.append(r)
		# second is same, but opposite x angle
		if len(imageRaysX)==1 and rays[0,r,xt]!=0 and rays[0,r,yt]==0 and \
			rays[0,r,x]==0 and rays[0,r,y]==0 and \
				rays[0,r,xt]!=rays[0,imageRaysX[0],xt]:
					imageRaysX.append(r)
		# image Y first ray is: nonzero angle y, zero angle x, zero x, zero y
		if len(imageRaysY)==0 and rays[0,r,yt]!=0 and rays[0,r,xt]==0 and \
			rays[0,r,x]==0 and rays[0,r,y]==0:
				imageRaysY.append(r)
		# second is same, but opposite x angle
		if len(imageRaysY)==1 and rays[0,r,yt]!=0 and rays[0,r,xt]==0 and \
			rays[0,r,x]==0 and rays[0,r,y]==0 and \
				rays[0,r,yt]!=rays[0,imageRaysY[0],yt]:
					imageRaysY.append(r)
		if len(diffRaysX)==2 and len(diffRaysY)==2 and \
			len(imageRaysX)==2 and len(imageRaysY)==2:
				break
	if ( "x" in axes and ( len(diffRaysX)!=2 or len(imageRaysX)!=2 )) or ( "y" in axes and ( len(diffRaysY)!=2 or len(imageRaysY)!=2 )):
		print("WARNING: diffraction and/or image rays could not be inferred by findPlanes(). no planes found")
		if len(diffRaysX)<2 and "x" in axes:
			print("diffraction rays (x2) in X: must be finite x, zero y, zero xt and yt")
		if len(diffRaysY)<2 and "y" in axes:
			print("diffraction rays (x2) in Y: must be finite y, zero x, zero xt and yt")
		if len(imageRaysX)<2 and "x" in axes:
			print("image rays (x2) in X: must be finite xt, zero yt, zero x and y")
		if len(imageRaysY)<2 and "y" in axes:
			print("image rays (x2) in Y: must be finite yt, zero xt, zero x and y")
		return returnable

	#print(diffRaysX,rays[0,diffRaysX[0],[x,xt]],rays[0,diffRaysX[1],[x,xt]])

	x=columnByName("x") ; y=columnByName("y")
	xt=columnByName("xt") ; yt=columnByName("yt")

	def whereCrossesZero(y0,y1): # returns relative position (0-1) of crossover, 
		# if the originally-parallel ray crosses zero, this is a diffraction plane
		if y1==0:								#'-.   ____m    (y-y0)=m*(x-x0)
			dz=1								#    '-.   |    solve for x where y=0
		elif y0<0 and y1>0 or y0>0 and y1<0:	#________'-.____x=-y0/m
			m=y1-y0 ; dz=-y0/m					#   dz       '-.
		else:
			return None
		return dz
			#Zd.append(i-1+dz)					# I'm actually storing the fractional index of the crossover! 
			#ma=ya1-ya0 ; mb=yb1-yb0				# "magnification" of the diffraction plane 
			#ya=ya0+ma*dz ; yb=yb0+mb*dz			# comes from the *difference in position* 
			#Md.append((ya-yb)/(ta-tb))			# for two rays starting at *different angles*

	def whereRaysCross(ya0,ya1,yb0,yb1):
		# If rays have crossed in x or y, there is an image plane between i-1 and i. See FultzHowe2013 Fig 2.9
		if ya1==yb1:							#'-.     /  (y-y0)=m*(x-x0)
			dz=1								#    '-./   solve for where
		elif ( ( ya0>yb0 and ya1<yb1 ) or 		#      / '-. ya=yb
				( ya0<yb0 and ya1>yb1 ) ):		#     / ma*(x-xa0)+ya0=mb*(x-xb0)+yb0
			ma=ya1-ya0 ; mb=yb1-yb0				#    / 	https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_line_equations
			dz=(yb0-ya0)/(ma-mb)				# a=ma ; b=mb ; c=xa0 ; d=xb0
		else:
			return None
		return dz
			#Zi.append(i-1+dz)					# magnification of image plane is
			#ya=ya0+ma*dz ; Mi.append(ya/rays[0,0,c]) # ratio of original size to current		
			#print(axis,"rays cross between",i-1,"and",i,"m",m,"dz",dz)

	def positionAtZ(y0,y1,dz):
		m=y1-y0
		return y0+m*dz

	# "magnification" of the diffraction plane comes from the *difference in position* for two rays starting at *different angles*. AKA, "camera length" if we assume positionondetector=scatteringangle*cameralength
	# "magnification" of the image plane comes from the *difference in position* for two rays starting at *different positions*
	def magnification(ya0,ya1,yb0,yb1,dz,ta,tb):
		ya=positionAtZ(ya0,ya1,dz)
		yb=positionAtZ(yb0,yb1,dz)
		#print("magnification where",ya0,"-",ya1,"and",yb0,"-",yb1,"cross",ya-yb,"/",ta-tb)
		return (ya-yb)/(ta-tb)

	# rays "a" and "b", original positions
	#xa0,xb0=rays[0,imageRayIndices,x]
	#ya0,yb0=rays[0,imageRayIndices,y]
	#xta0,xtb0=rays[0,imageRayIndices,xt]
	#yta0,ytb0=rays[0,imageRayIndices,yt]

	for axis,diffRays,imageRays,xy,yx,xyt in zip(["x","y"],[diffRaysX,diffRaysY],[imageRaysX,imageRaysY],[x,y],[y,x],[xt,yt]):
		if axis not in axes:
			continue
		for i in range(1,len(rays)):
			# CHECK DIFFRACTION: where originally-parallel rays cross
			(xa1,xb1),(xa2,xb2)=rays[i-1:i+1,diffRays,xy]
			dz=whereRaysCross(xa1,xa2,xb1,xb2)
			if dz is not None:
				xta0,xtb0 = rays[0,imageRays,xyt] # magnification comes from starting angles of two non-parallel rays
				# Magnification comes from conversion of angle to position
				(xa1,xb1),(xa2,xb2)=rays[i-1:i+1,imageRays,xy]
				(ya1,yb1),(ya2,yb2)=rays[i-1:i+1,imageRays,yx]
				M=magnification(xa1,xa2,xb1,xb2,dz,xta0,xtb0)
				returnable[axis]["diff"]["z"].append( i-1+dz )
				returnable[axis]["diff"]["M"].append( M )
				returnable[axis]["diff"]["p"].append([])
				for r in range(len(rays[0])):
					xr=positionAtZ(*rays[i-1:i+1,r,x],dz)
					yr=positionAtZ(*rays[i-1:i+1,r,y],dz)
					returnable[axis]["diff"]["p"][-1].append( [xr,yr] )

			# CHECK IMAGE PLANE: where rays leaving the same place re-cross
			(xa1,xb1),(xa2,xb2)=rays[i-1:i+1,imageRays,xy]
			dz=whereRaysCross(xa1,xa2,xb1,xb2)
			if dz is not None:
				xa0,xb0 = rays[0,diffRays,xy] # magnification comes from change in scaling (position) of originally-parallel rays
				(xa1,xb1),(xa2,xb2)=rays[i-1:i+1,diffRays,xy]
				(ya1,yb1),(ya2,yb2)=rays[i-1:i+1,diffRays,yx]
				M=magnification(xa1,xa2,xb1,xb2,dz,xa0,xb0)
				returnable[axis]["image"]["z"].append( i-1+dz )
				returnable[axis]["image"]["M"].append( M )
				returnable[axis]["image"]["p"].append([])
				for r in range(len(rays[0])):
					xr=positionAtZ(*rays[i-1:i+1,r,x],dz)
					yr=positionAtZ(*rays[i-1:i+1,r,y],dz)
					returnable[axis]["image"]["p"][-1].append( [xr,yr] )

	return returnable


def findPlanes1(rays):
	# Infer which rays we'll use for detecting the planes! we should not require the user to understand the above criteria (and pass them) nor should we make assumptions on how the user constructed their list of rays
	diffRayIndex=None ; imageRayIndices=[]
	x=columnByName("x") ; y=columnByName("y")
	xt=columnByName("xt") ; yt=columnByName("yt")

	returnable={}
	for xy in ["x","y"]:
		returnable[xy]={}
		for imdiff in ["diff","image"]:
			returnable[xy][imdiff]={}
			for p in ["z","M","R","p"]:
				returnable[xy][imdiff][p]=[]

	# diffraction ray is the first ray emitted at zero angle (nonzero position!)
	for r in range(len(rays[0])):
		if rays[0,r,xt]==0 and rays[0,r,yt]==0 and \
				rays[0,r,x]!=0 and rays[0,r,y]!=0:
			diffRayIndex=r
			break
	else:
		return returnable

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
		return returnable
	#print("using diffRayIndex",diffRayIndex,rays[0,diffRayIndex])
	#print("using imageRayIndices",diffRayIndex,rays[0,imageRayIndices[0]])
	#print("and",diffRayIndex,rays[0,imageRayIndices[1]])

	x=columnByName("x") ; y=columnByName("y")
	xt=columnByName("xt") ; yt=columnByName("yt")

	def whereCrossesZero(y0,y1): # returns relative position (0-1) of crossover, 
		# if the originally-parallel ray crosses zero, this is a diffraction plane
		if y1==0:								#'-.   ____m    (y-y0)=m*(x-x0)
			dz=1								#    '-.   |    solve for x where y=0
		elif y0<0 and y1>0 or y0>0 and y1<0:	#________'-.____x=-y0/m
			m=y1-y0 ; dz=-y0/m					#   dz       '-.
		else:
			return None
		return dz
			#Zd.append(i-1+dz)					# I'm actually storing the fractional index of the crossover! 
			#ma=ya1-ya0 ; mb=yb1-yb0				# "magnification" of the diffraction plane 
			#ya=ya0+ma*dz ; yb=yb0+mb*dz			# comes from the *difference in position* 
			#Md.append((ya-yb)/(ta-tb))			# for two rays starting at *different angles*

	def whereRaysCross(ya0,ya1,yb0,yb1):
		# If rays have crossed in x or y, there is an image plane between i-1 and i. See FultzHowe2013 Fig 2.9
		if ya1==yb1:							#'-.     /  (y-y0)=m*(x-x0)
			dz=1								#    '-./   solve for where
		elif ( ( ya0>yb0 and ya1<yb1 ) or 		#      / '-. ya=yb
				( ya0<yb0 and ya1>yb1 ) ):		#     / ma*(x-xa0)+ya0=mb*(x-xb0)+yb0
			ma=ya1-ya0 ; mb=yb1-yb0				#    / 	https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_line_equations
			dz=(yb0-ya0)/(ma-mb)				# a=ma ; b=mb ; c=xa0 ; d=xb0
		else:
			return None
		return dz
			#Zi.append(i-1+dz)					# magnification of image plane is
			#ya=ya0+ma*dz ; Mi.append(ya/rays[0,0,c]) # ratio of original size to current		
			#print(axis,"rays cross between",i-1,"and",i,"m",m,"dz",dz)

	def positionAtZ(y0,y1,dz):
		m=y1-y0
		return y0+m*dz

	# "magnification" of the diffraction plane comes from the *difference in position* for two rays starting at *different angles*. AKA, "camera length" if we assume positionondetector=scatteringangle*cameralength
	# "magnification" of the image plane comes from the *difference in position* for two rays starting at *different positions*
	def magnification(ya0,ya1,yb0,yb1,dz,ta,tb):
		ya=positionAtZ(ya0,ya1,dz)
		yb=positionAtZ(yb0,yb1,dz)
		return (ya-yb)/(ta-tb)

	# rays "a" and "b", original positions
	xa0,xb0=rays[0,imageRayIndices,x]
	ya0,yb0=rays[0,imageRayIndices,y]
	xta0,xtb0=rays[0,imageRayIndices,xt]
	yta0,ytb0=rays[0,imageRayIndices,yt]

	for i in range(1,len(rays)):
		# rays "a" and "b", start/end position (for this line segment)
		(xa1,xb1),(xa2,xb2)=rays[i-1:i+1,imageRayIndices,x]
		(ya1,yb1),(ya2,yb2)=rays[i-1:i+1,imageRayIndices,y]
		xd1,xd2=rays[i-1:i+1,diffRayIndex,x]	# ray "d", diffraction ray
		yd1,yd2=rays[i-1:i+1,diffRayIndex,y]
		
		dzx=whereCrossesZero(xd1,xd2)
		if dzx is not None:
			returnable["x"]["diff"]["z"].append(i-1+dzx)
			returnable["x"]["diff"]["M"].append( magnification(xa1,xa2,xb1,xb2,dzx,xta0,xtb0) )
			returnable["x"]["diff"]["p"].append([])
			for r in range(len(rays[0])):
				xr=positionAtZ(*rays[i-1:i+1,r,x],dzx)
				yr=positionAtZ(*rays[i-1:i+1,r,y],dzx)
				returnable["x"]["diff"]["p"][-1].append( [xr,yr] )

		dzy=whereCrossesZero(yd1,yd2)
		if dzy is not None:
			returnable["y"]["diff"]["z"].append(i-1+dzy)
			returnable["y"]["diff"]["M"].append( magnification(ya1,ya2,yb1,yb2,dzy,yta0,ytb0) )
			returnable["y"]["diff"]["p"].append([])
			for r in range(len(rays[0])):
				xr=positionAtZ(*rays[i-1:i+1,r,x],dzy)
				yr=positionAtZ(*rays[i-1:i+1,r,y],dzy)
				returnable["y"]["diff"]["p"][-1].append( [xr,yr] )

		dzx=whereRaysCross(xa1,xa2,xb1,xb2)
		if dzx is not None:
			returnable["x"]["image"]["z"].append(i-1+dzx)
			returnable["x"]["image"]["M"].append( magnification(xa1,xa2,xb1,xb2,dzx,xa0,xb0) )
			returnable["x"]["image"]["p"].append([])
			for r in range(len(rays[0])):
				xr=positionAtZ(*rays[i-1:i+1,r,x],dzx)
				yr=positionAtZ(*rays[i-1:i+1,r,y],dzx)
				returnable["x"]["image"]["p"][-1].append( [xr,yr] )


		dzy=whereRaysCross(ya1,ya2,yb1,yb2)
		if dzy is not None:
			returnable["y"]["image"]["z"].append(i-1+dzy)
			returnable["y"]["image"]["M"].append( magnification(ya1,ya2,yb1,yb2,dzy,ya0,yb0) )
			returnable["y"]["image"]["p"].append([])
			for r in range(len(rays[0])):
				xr=positionAtZ(*rays[i-1:i+1,r,x],dzy)
				yr=positionAtZ(*rays[i-1:i+1,r,y],dzy)
				returnable["y"]["image"]["p"][-1].append( [xr,yr] )


	return returnable



# Finds image and diffraction planes based on the crossing of rays: 
# ~ Any two rays originating from the same point form an image plane when they recross
# ~ Any singular ray which began at zero angle finds a diffraction plane when crossing x=0 or y=0
# returns nested dicts with keys: ["x" or "y"]["image" or "diff"]["M" or "z"]
# positions ("z") are stored in fractional coordinates, i.e., "1.2" is "20% of the way between index 1 and 2". use zFromFractional to convert to true z positions
# magnification ("M") in real space is defined as: the ratio of final/original positions of the ray
# magnification ("M") for a diffraction plane is defined as: the ratio of final position vs starting angle of the ray
# this diffraction mag is also known as the "camera length" based on the small angle approximation: dx=L*theta
def findPlanes2(rays):
	# Infer which rays we'll use for detecting the planes! we should not require the user to understand the above criteria (and pass them) nor should we make assumptions on how the user constructed their list of rays
	diffRayIndex=None ; imageRayIndices=[]
	x=columnByName("x") ; y=columnByName("y")
	xt=columnByName("xt") ; yt=columnByName("yt")

	returnable={}
	for xy in ["x","y"]:
		returnable[xy]={}
		for imdiff in ["diff","image"]:
			returnable[xy][imdiff]={}
			for p in ["z","M","R","p"]:
				returnable[xy][imdiff][p]=[]

	# diffraction ray is the first ray emitted at zero angle (nonzero position!)
	for r in range(len(rays[0])):
		if rays[0,r,xt]==0 and rays[0,r,yt]==0 and \
				rays[0,r,x]!=0 and rays[0,r,y]!=0:
			diffRayIndex=r
			break
	else:
		return returnable

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
		return returnable
	#print("using diffRayIndex",diffRayIndex,rays[0,diffRayIndex])
	#print("using imageRayIndices",diffRayIndex,rays[0,imageRayIndices[0]])
	#print("and",diffRayIndex,rays[0,imageRayIndices[1]])


	# loop through elements, get start/end points of each ray, and do some basic "crossing" math		
	for ij,axis in enumerate(["x","y"]):
		# we'll store each image (i) and diffraction (d) position (Z) magnification (M) and rotation (R) here
		Zi=returnable[axis]["image"]["z"] ; Zd=returnable[axis]["diff"]["z"] # references to the lists in the dict tree! 
		Mi=returnable[axis]["image"]["M"] ; Md=returnable[axis]["diff"]["M"] #; Ri=[] ; Rd=[]
		Pi=returnable[axis]["image"]["p"] ; Pd=returnable[axis]["diff"]["p"]
		# get indices for columns
		c=columnByName(axis) ; ct=columnByName(axis+"t")
		x=columnByName("x") ; y=columnByName("y")
		xt=columnByName("xt") ; yt=columnByName("yt")
		# loop through ray segments
		for i in range(1,len(rays)):
			ya0,yb0=rays[i-1,imageRayIndices,c] # rays "a" and "b", starting position (of this line segment)
			ya1,yb1=rays[i,imageRayIndices,c]
			ta,tb=rays[0,imageRayIndices,ct]	# initial angle (t) for rays "a" and "b"
			yd0=rays[i-1,diffRayIndex,c]		# ray "d", diffraction ray
			yd1=rays[i,diffRayIndex,c]


			# if the originally-parallel ray crosses zero, this is a diffraction plane
			if yd1==0:								#'-.   ____m    (y-y0)=m*(x-x0)
				Zd.append(i)						#    '-.   |    solve for x where y=0
			if yd0<0 and yd1>0 or yd0>0 and yd1<0:	#________'-.____x=-y0/m
				m=yd1-yd0 ; dz=-yd0/m				#   dz       '-.
				Zd.append(i-1+dz)					# I'm actually storing the fractional index of the crossover! 
				ma=ya1-ya0 ; mb=yb1-yb0				# "magnification" of the diffraction plane 
				ya=ya0+ma*dz ; yb=yb0+mb*dz			# comes from the *difference in position* 
				Md.append((ya-yb)/(ta-tb))			# for two rays starting at *different angles*
				#deltax0=rays[0,imageRayIndices,xt]
				#deltay0=rays[0,imageRayIndices,yt]
				#th0=np.atan2(deltay0[1]-deltay0[0],deltax0[1]-deltax0[0])
				#thf=
				#Rd.append( np.atan2(y
				# "rotation" of the diffraction plane comes from angle emitted vs angle of position?
				#th0=np.atan2(ta
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

		#if axis=="x":
		#	Zix=Zi ; Zdx=Zd ; Mix=Mi ; Mdx=Md
		#else:
		#	Ziy=Zi ; Zdy=Zd ; Miy=Mi ; Mdy=Md

	return returnable
	#return {"x":{"diff":{"z":Zdx,"M":Mdx},"image":{"z":Zix,"M":Mix}},
	#		"y":{"diff":{"z":Zdy,"M":Mdy},"image":{"z":Ziy,"M":Miy}}}

def zFromFractional(zs,z): # e.g. 1.2 is 20% of the distance through element index 1
	i,di=int(z),z-int(z) # 1.2 --> i=1, and di=0.2
	z0=zs[i] ; z1=zs[i+1]
	return z0+(z1-z0)*di

# TODO what's the "right" way to chain together a whole bunch of different criteria? sometimes i want to fit "multiple settings" (PL1=a1,PL2=b1,PL3=c1,PL4=d1, with a diffraction plane here, and a magnification of M1, and PL1=a2,PL2=b2,PL3=c2,PL4=d2 and magnification of M2, and so on, what are the calibrations for PL1 PL2 PL3 PL4?). sometimes i want to fit "multiple planes" (what values of PL1 and PL2 give me a diffraction plane here and an image plane there?) or "multiple criteria for a single plane" (what values of PL1 and PL2 give me a diffraction plane here with this magnification?).
# I Think the answer is: a bunch of sub-functions for each criteria, then custom error functions for the chaining.
# EXAMPLE 1: consider the case where I want "a crossover at the PL2 plane when PL1 is v2, a crossover at the PL3 plane when PL1 is v3, and a crossover at the PL4 plane when PL1 is v4, what is the calibration for PL1?", this can be done as follows, given the input dict PL1vals={"PL2":v2,"PL3":v3,"PL4":v4}
# def dz(vals):
#	microscope["PL1"].calibration = vals
#	deltas = []
#	for PL,v in PL1vals.items():
#		settings = {"PL1":{"stregth":v},"PL2":{"stregth":0},"PL3":{"stregth":0},"PL4":{"stregth":0}}
#		x = microscope["projector"].position+microscope[PL].position
#		targets = {"image":x}
#		deltas += error_dz( microscope, settings, targets )
#	return np.sqrt( np.sum(np.asarray(deltas)**2))
# then i simply call minimize(dz,guesses) etc
# EXAMPLE 2: TODO

def update_microscope_with_settings(microscope,settings):
	for element in settings.keys():
		for attribute,value in settings[element].items():
			if not hasattr(microscope[element],attribute):
				raise AttributeError("Attribute \""+attribute+"\" not found on "+str(type(microscope[element]))+" Element")
			setattr(microscope[element],attribute,value)

# given a Microscope object, a dict of lens parameters, and a dict of planes, detects nearest plane of the correct type, and return the delta in positions.
def error_dz(microscope,settings,targets): # settings is a dict of parameters to set {"PL1":{"strength":.475}}, targets is a dict of things to check {"diff":5,"image":7}
	# UPDATE ALL ELEMENTS SPECIFIED
	update_microscope_with_settings(microscope,settings)
	#microscope.show()
	# PROPAGATE, DETECT PLANES
	r1=microscope.propagate_ray()
	planes = findPlanes(r1,"x")
	# FOR EACH TARGET PLANE, FIND CLOSEST OF SAME TYPE, ERROR IS DELTA IN POSITION
	deltas = []
	for plane_type,z in targets.items():
		zs = r1[:,0,columnByName("z")] 								# all positions of 0th ray
		zps_fractional = planes["x"][ plane_type ]["z"]				# coordinates are nth-element, % distance between
		if len(zps_fractional)==0:
			deltas.append(1000) ; continue
		zps_real = [ zFromFractional(zs,z) for z in zps_fractional ]
		n=np.argmin( np.absolute(np.asarray(zps_real)-z) )	# find the index of the closest plane
		deltas.append( zps_real[n]-z )
	return deltas

# given a Microscope object, a dict of lens parameters, and a dict of planes, detects nearest plane of the correct type, and return the delta in magnifications
def error_magnification(microscope,settings,targets): # settings is a dict of parameters to set {"PL1":{"strength":.475}}, targets is a dict of things to check {"diff":{"z":5,"mag":10}}
	# UPDATE ALL ELEMENTS SPECIFIED
	update_microscope_with_settings(microscope,settings)
	#microscope.show()
	# PROPAGATE, DETECT PLANES
	r1=microscope.propagate_ray()
	planes = findPlanes(r1,"x")
	# FOR EACH TARGET PLANE, FIND CLOSEST OF SAME TYPE, ERROR IS DELTA IN POSITION
	deltas = []
	for plane_type,zm in targets.items():
		z=zm["z"] ; mag=zm["mag"]
		zs = r1[:,0,columnByName("z")] 								# all positions of 0th ray
		zps_fractional = planes["x"][ plane_type ]["z"]				# coordinates are nth-element, % distance between
		zps_real = [ zFromFractional(zs,z) for z in zps_fractional ]
		n=np.argmin( np.absolute(np.asarray(zps_real)-z) )	# find the index of the closest plane
		deltas.append( planes['x'][ plane_type ]['M']-mag )
	return deltas

# given a Microscope object, a dict of lens parameters, and a list of positions, simply returns the beam diameter at each position
def error_diameter(microscope,settings,targets,absolute=False): # settings is a dict of parameters to set {"PL1":{"strength":.475}}, targets is a list of positions [5,7]
	# UPDATE ALL ELEMENTS SPECIFIED
	update_microscope_with_settings(microscope,settings)
	# PROPAGATE, MEASURE BEAM
	r1=microscope.propagate_ray()
	diameters = []
	for z in targets:
		x,y,xt,yt = measureAtZ(z,rays=r1)
		if absolute:
			x=np.absolute(x)
		diameters.append(x)
	return diameters

# given a Microscope object, a dict of lens parameters, and a list of positions, simply returns the outermost ray's angles at each position???
def error_angles(microscope,settings,targets,absolute=False): # settings is a dict of parameters to set {"PL1":{"strength":.475}}, targets is a list of positions [5,7]
	# UPDATE ALL ELEMENTS SPECIFIED
	update_microscope_with_settings(microscope,settings)
	# PROPAGATE, MEASURE BEAM
	r1=microscope.propagate_ray()
	angles = []
	for z in targets:
		x,y,xt,yt = measureAtZ(z,rays=r1)
		if absolute:
			xt=np.absolute(xt)
		angles.append(xt)
	return angles


# error function (passable to scipy.minimize et al). modifiable is a dict of keywords, {"PL1":"calibration"}, settings is a list of dicts of settings: {"PL1":{"strength":.475}}
#def error_multisetting(microscope,modifiable=[],settings=[]):



# Given the ability to 1) generate a section 2) propagate rays and 3) measure attributes of the propagated rays (e.g. location of planes and magnifications), we should be able to fit for variables (like lens strength) to achieve a desired result
# Desired result may be: position of an image/diffraction plane, magnification at that plane, angles coming in, or unbounded desirables like "maximize the magnitude" or "minimize the lens currents"
# 1) create the function "propagateAndCheck" which updates section > element > properties, propagates, and finds the planes
# 2) define an "error" quantity which scipy.optimize minimize or brute can try to minimize by perturbing the properties
# pass:
# r0 : initial list of rays. at least one needs to be normal angle (and not at position 0), and another pair emitted from the same point
# section : a microscope section object
# targets : a list of dicts for what we want, e.g. [{"plane":"image","z":6,"mag":3}] would mean "we want an image plane at z=6 with a magnification of 3"
# modifiable : dict of element index/parameter pairs. e.g. {1:"strength",3:"strength"} if I wish to allow lens at index 1 and 3 to have their strength varied
def fitForCrossover(section,r0=None,targets=[],modifiable=[],axis="x",prefer={},ignoreSigns=True,filename=""):

	# propoagateAndCheck below takes a list of values, so we need to "map" these to modifiable elements and the parameters within that element
	indices,eleKeys,ivals=[],[],[]
	for i in modifiable.keys():
		k=modifiable[i] ; v=section[i].kget(k)
		if v is None:
			v=1
		indices.append(i) ; eleKeys.append(k) ; ivals.append(v)
	#print("indices",indices,"eleKeys",eleKeys,"ivals",ivals)

	# a function which sets passed values, propagates, finds the crossover location, and returns an "error" term to be minimized
	def propagateAndCheck(vals,passback="dz",ct_propagateAndCheck=[-1]):

		ct_propagateAndCheck[0]+=1

		if isinstance(passback,list):
			passback=passback[0]

		# set each element's parameter based on the list of values passed
		for i,v in enumerate(vals):
			i,kk=indices[i],eleKeys[i]
			section[i].kset(kk,v)

		# propagate the starting array through the (now-updated) section
		r1=section.propagate_ray(r0)

		#print("propagateAndCheck run",ct_propagateAndCheck[0],"trying vals",vals)
		#plot2D( r1 , filename = "tmp/"+str(ct_propagateAndCheck[0])+".png")

		# inspect the output: find all image and diffraction planes
		planes = findPlanes(r1,axes=axis)
		zs=r1[:,0,columnByName("z")] # all positions of 0th ray

		# our "error" defined by each metric in each target (e.g. checking if position z is off, or magnification is off)
		# NOTE: the default is to minimize the sum (rather than a more-conventional "mean squared error")
		deltas=[] ; mags=[]
		for target in targets:

			# SPECIFIED CROSSOVER LOCATION [ REQUIRED ]
			z_desired=target["z"]

			if "rays" in target.keys() and target["rays"]=="0x-parallel":
				_,nRays,_=r1.shape
				# "zero position rays which are now parallel": select the rays which started (z=0) with zero position (x or y) and non-zero angle (xt or yt). 
				zeroAngleRays=[ i for i in range(nRays) if r1[0,i,columnByName(axis)]==0 and r1[0,i,columnByName(axis+"t")]!=0 ]
				zi1=np.argwhere(zs<z_desired)[-1] ; zi2=np.argwhere(zs>=z_desired)[0]
				dz = zs[zi2]-zs[zi1]
				slopes = ( r1[zi2,zeroAngleRays,columnByName(axis)] - r1[zi1,zeroAngleRays,columnByName(axis)] ) / dz
				deltas.append(np.sqrt(np.sum(slopes**2)))

			if "scale-match" in target.keys():
				scales = np.asarray(vals)/np.asarray(ivals) # compare current vs initial value
				dz = scales-np.mean(scales)
				deltas.append(np.sqrt(np.sum(dz**2)))

			if "plane" not in target.keys():
				continue

			# find closest plane of the correct type
			Zi=planes[axis][ target["plane"] ]["z"]	# "fractional coordinates" of positions of all correct-type planes: target["plane"] is "image" or "diff"

			# plausible there isn't a plane of that type! (maybe the minimization algorith, pushed it out of range)
			if len(Zi)==0:
				deltas.append(10000.) ; mags.append(0) #; signs.append(1)
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
						#print("dM",dM,"target",target["mag"],"found",Ms[n])
						deltas.append(dM*100)				

				# LENS STRENGTH	
				if k=="strength":
					strengths = np.asarray( [ v for v,k in zip(vals,eleKeys) if k=="strength" ] )
					#strength=np.mean(strengths)/10
					if target["strength"]=="minimize":
						strength=max(strengths)**.5/10 #np.sum(strengths**2)
						deltas.append(strength)
					elif target["strength"]=="maximize":
						#strength=np.mean(strengths)/20
						strength=min(strengths)**5/10
						deltas.append(-1*strength)
		#print("found deltas",deltas)

			#print("USING",vals,"CROSSOVER",n," IS AT",Zf[n],"DZ",dz,"DM",dM,"mags",mags) #,"deltas",deltas,"signs",signs)
		#plotRays(r1)
		#print("FOUND",deltas,"(",vals,")")
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
	ranges=[[v*.5,v*1.5] for v in ivals ]
	#ranges=[[v*.1,v*3] for v in ivals ]

	x0=minimize(propagateAndCheck,x0=ivals,bounds=ranges,method='trust-constr')["x"] #,method='trust-constr',options={"finite_diff_rel_step":[.1]*len(vals),"xtol":1e-12})
	return

	# scipy.optimize.brute: should be better at finding global minima. also convenient for plotting heatmaps of the parameter space
	ranges=[[v/4,v*2] for v in ivals ] # TODO WHAT SHOULD THESE BE? (user should probably be allowed to pass this, or we infer from the actual microscopy itself)
	ranges=[[v*.1,v*5] for v in ivals ]
	print("ranges",ranges)
	# we will wrap scipy.optimize.minimize to use as brute's "polish" function
	def mini(*args,**kwargs):	
		kwargs["bounds"]=ranges
		return minimize(*args,**kwargs)
	# run fitting. full_output only required if you want the contour plot. same for Ns
	x0,r,vals,residuals=brute(propagateAndCheck,ranges=ranges,Ns=100,full_output=True,args=["dz"],finish=mini)
	# heatmap of parameter space
	residuals[residuals==10000]=np.nan

	return

	plt.clf()
	if len(residuals.shape)==2:
		plt.imshow(residuals[::-1,:],extent=(np.amin(vals[1]),np.amax(vals[1]),np.amin(vals[0]),np.amax(vals[0])))
		plt.xlabel(eleKeys[1]+" "+str(indices[1]))
		plt.ylabel(eleKeys[0]+" "+str(indices[0]))
		where=np.where(residuals==np.nanmin(residuals)) ; print("where",where)
		#if len(where[1])>0:
		plt.plot(vals[1,0,where[1][0]],vals[0,where[0][0],0],c="r",marker="o")
		plt.cbar=plt.colorbar()
	elif len(residuals.shape)==1:
		plt.plot(vals,residuals)
		plt.xlabel(eleKeys[0]+" "+str(indices[0]))
	plt.title(str(targets)+"\n residuals, best at "+str(x0))

	if len(filename)>0:
		plt.savefig(filename+"a.png")
		plot2D( propagateAndCheck(x0,"r1") , filename=filename+"b.png")
	else:
		plt.show()
		# plot the final rays
		plot2D( propagateAndCheck(x0,"r1") )

# PROPERTIES OF THE OUTERMOST RAYS
def measureAtZ(z,rays=None,section=None):
	if rays is None and section.rays is None:
		section.propagate_ray()
	if rays is None:
		rays = section.rays
	if isinstance(z,str):
		z=section[z].position
	zs = rays[:,0,columnByName('z')] # nthElement,nthRay,xythetaetc
	i=np.where(zs<=z)[0][-1] # closest elemnt before or at z
	#print(z,zs,i,zs[i])
	x,y,xt,yt = [ columnByName(v) for v in ["x","y","xt","yt"] ]
	def interp(z,z1,z2,y1,y2):
		return y1+(z-z1)/(z2-z1)*(y2-y1)
	xs = interp(z,zs[i],zs[i+1],rays[i,:,x],rays[i+1,:,x])	# lateral position of all rays between elements i and i+1
	ys = interp(z,zs[i],zs[i+1],rays[i,:,y],rays[i+1,:,y])
	selected = np.argmax(np.sqrt(xs**2+ys**2))				# index of outermost ray
	x = xs[selected]											# lateral position of outermost ray
	xt = rays[i,selected,xt]									# angle of outermost ray
	sel_y = np.argmax(ys)
	y = ys[selected]
	yt = rays[i,selected,yt]
	#print("x,y,xt,yt",x,y,xt,yt)
	return x,y,xt,yt

