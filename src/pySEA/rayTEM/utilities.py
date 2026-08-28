import os,json
import numpy as np
from scipy.optimize import minimize,brute

trapezoid = getattr(np, "trapezoid", None)
if trapezoid is None:
	trapezoid = np.trapz

# Physical constants (CODATA 2018), SI units
_PLANCK = 6.62607015e-34        # J s
_ELECTRON_MASS = 9.1093837015e-31   # kg
_ELECTRON_CHARGE = 1.602176634e-19  # C
_SPEED_OF_LIGHT = 299792458.0       # m / s


def relativistic_wavelength(voltage_kV: float) -> float:
	r"""Relativistic de Broglie wavelength of an electron at a given accelerating voltage.

	Computes the wavelength of an electron accelerated through a potential
	difference ``voltage_kV`` (in kilovolts), including the relativistic
	correction that matters at TEM energies. Used to seed the wave-optics
	initial field and to convert accelerating voltage to wavelength for
	envelope/wave propagation.

	Parameters
	----------
	voltage_kV : float
		Accelerating voltage in kilovolts (e.g. ``200`` for a 200 kV instrument).

	Returns
	-------
	float
		Electron wavelength in metres.

	Raises
	------
	ValueError
		If ``voltage_kV`` is not strictly positive.

	Notes
	-----
	The wavelength follows

	.. math::

		\lambda = \frac{h}{\sqrt{2 m_0 e V \left(1 + \dfrac{e V}{2 m_0 c^2}\right)}}

	with :math:`V` the accelerating voltage in volts and :math:`h, m_0, e, c`
	the Planck constant, electron rest mass, elementary charge, and speed of
	light. The parenthetical term is the relativistic correction.

	Examples
	--------
	>>> round(relativistic_wavelength(200) * 1e12, 3)  # picometres
	2.508

	References
	----------
	Williams, D. B. and Carter, C. B., *Transmission Electron Microscopy*,
	2nd ed., Springer (2009), Eq. 1.6-1.7.
	"""
	if voltage_kV <= 0:
		raise ValueError(f"voltage_kV must be positive, got {voltage_kV}.")
	V = voltage_kV * 1e3        # kV -> V
	eV = _ELECTRON_CHARGE * V
	denominator = np.sqrt(2 * _ELECTRON_MASS * eV *
						  (1 + eV / (2 * _ELECTRON_MASS * _SPEED_OF_LIGHT**2)))
	return _PLANCK / denominator


# ELLIPSE FITTING BELOW STOLEN FROM SEA-ECO, WITH CACHING ADDED



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
