# ELLIPSE FITTING BELOW STOLEN FROM SEA-ECO, WITH CACHING ADDED

import os,json
import numpy as np
from scipy.optimize import minimize,brute

def ellipse(t,xc,yc,a,b,theta):
	x,y=a*np.cos(t),b*np.sin(t)			# start with a scrunched circle
	c=np.cos(theta) ; s=np.sin(theta) ; R=np.asarray([[c,-s],[s,c]])
	x,y=np.matmul(R,[x,y]) 				# apply rotation matrix
	return x+xc,y+yc 					# shift by center position

def _ellipse_minimize(xs,ys):
	def dz(args):						# error function
		xc,yc,a,b,theta = args
		ts = np.linspace(0,2*np.pi,360*3,endpoint=False)
		x,y=ellipse(ts,xc,yc,a,b,theta)	# points for the ellipse for args passed
		# distance from all given points (xs,ys) to all ellipse points (x,y)
		distances=np.sqrt( (xs[:,None]-x[None,:])**2+(ys[:,None]-y[None,:])**2 )
		# collapse to find each xs,ys points' closest point on ellipse
		distances = np.amin(distances,axis=1)
		return np.sqrt(np.sum(distances**2))	# use MSE distance as our error metric
	# guesses: center in x,y, width and height, zero angle to start
	x0 = ( np.mean(xs) , np.mean(ys) , np.ptp(xs)/2 , np.ptp(ys)/2 , 0 )
	res = minimize(dz,x0)
	return res.x

def findEllipse(data,xs,ys,return_debugging=False,caching=False): # "caching" should be a filename where we save measured ellipse info
	if caching and os.path.exists(caching):						# if a filename is passed, and the json exists
		with open(caching, 'r') as fo:							# load it, parse it....
			dic = json.load(fo)
		print("reload ellipse from cache")
		x=np.asarray(dic['x']) ; y=np.asarray(dic['y'])
		cxe=dic['cxe'] ; cye=dic['cye'] ; ae=dic['ae'] ; be=dic['be'] ; thetae=dic['thetae']
		return (x,y),(cxe,cye,ae,be,thetae)						# and return results...

	# PREP: select pixels above a threshold
	mask = np.zeros(data.shape)
	mask[ data > np.mean(data) + np.std(data) ] = 1
	# denoising: rolling the mask in each direction and summing means we can filter to "only points who's neighbor was also above threshold"
	rolled = mask+np.roll(mask,1,axis=0)+np.roll(mask,-1,axis=0)+\
		np.roll(mask,1,axis=1)+np.roll(mask,-1,axis=1)
	bounds = np.where(rolled>2)
	# convert indices to datapoints on the plot
	ysf = ys[bounds[0]] ; xsf = xs[bounds[1]]

	# ELLIPSE FINDING

	# start with center of mass
	cx = np.sum(xs[None,:]*mask)/np.sum(mask)
	cy = np.sum(ys[:,None]*mask)/np.sum(mask)

	# try to detect the border
	border = np.where(rolled == 3) # 1 for mask-selected pixels, +2 neighbors
	#border = np.concatenate( [ np.where(rolled == b) for b in [3,4] ], axis=1 )
	# filter to only external borders (in case there is noise or signal inside the border)
	filtered = [[],[]]
	for jj,ii in zip(*border): # TODO i wish there was a better way than just looping...
		if sum(rolled[jj+1:,ii])<2 or sum(rolled[:jj-1,ii])<2 or\
			sum(rolled[jj,ii+1:])<2 or sum(rolled[jj,:ii-1])<2:
			filtered[0].append(jj) ; filtered[1].append(ii)
	border=np.asarray(filtered)
	# convert indices to datapoints on the plot
	ysb = ys[border[0]] ; xsb = xs[border[1]]

	# ellipse fitting?
	# https://stackoverflow.com/questions/77594526/fitting-an-ellipse-in-python
	#A = np.stack([qxb**2, qxb * qyb, qyb**2, qxb, qyb]).T
	#b = np.ones_like(qxb)
	#w = np.linalg.lstsq(A, b)[0].squeeze()
	#X, Y = np.meshgrid(qx, qy)
	#Z = w[0]*X**2 + w[1]*X*Y + w[2]*Y**2 + w[3]*X + w[4]*Y
	# https://stackoverflow.com/questions/47873759/how-to-fit-a-2d-ellipse-to-given-points
	#U, S, V = np.linalg.svd(np.stack((qxb-cx, qyb-cy)))
	#phi = np.linspace(0, 2*np.pi, 1000)
	#circle = np.stack((np.cos(phi), np.sin(phi)))    # unit circle
	#transform = np.sqrt(2/len(qxb)) * U.dot(np.diag(S))   # transformation matrix
	#fit = transform.dot(circle) + np.array([[cx], [cy]])
	# SVD approach seems to fail for unevenly-spaced border points
	#x,y=ellipse(np.linspace(0, 2*np.pi, 1000),cx,cy,np.ptp(qxf)/2,np.ptp(qyf)/2,0)
	cxe,cye,ae,be,thetae = _ellipse_minimize( xsb , ysb )
	x,y = ellipse( np.linspace(0, 2*np.pi, 1000) , cxe,cye,ae,be,thetae )

	if return_debugging:
		return (x,y),(cxe,cye,ae,be,thetae),(xsf,ysf),(xsb,ysb)

	if caching:							# if a filename is passed, and the json does *not* exist, save it off
		with open(caching,'w') as fo:
			json.dump({ "x":list(x), "y":list(y), "cxe":cxe, "cye":cye, "ae":ae, "be":be, "thetae":thetae },fo)

	return (x,y),(cxe,cye,ae,be,thetae)
