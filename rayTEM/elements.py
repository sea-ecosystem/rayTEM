# try:
#     import cupy as xp
#     flag_gpu = True
#     from cupy.typing import ArrayLike
# except:
#     import numpy as xp
#     flag_gpu = False
#     from numpy.typing import ArrayLike
import numpy as xp
flag_gpu = False
from numpy.typing import ArrayLike

from pandas import DataFrame
from warnings import warn


class Element:
    def __init__(self, name:str='Unnamed',
                 kind:None|str=None, poles:None|int=None,
                 position:float=0, length:float=0,
                 strength:float=0, calibration:None|float=None,
                 ndim:int=2, chroma_dim:bool=False,
                 label:bool=False, print_fancy:bool=True
                 ) -> object:
        """General microscope element class.

        Parameters
        ----------
        name : str, optional
            Name given to the lens, by default ''
        kind : str, optional
            Type of element, by default None
        poles : None, int, optional
            Number of poles in the element.
            Drift = 0
            Dipole = 2
            Quadropole = 4
        position : float, optional
            The position of the element along the z-axis, by default 0
        length : int, optional
            Length of the element, by default 0
        strength : float, optional
            Defined as the The focusing strength (K) of a thin lens, by default 0
        calibration : float, optional
            Currnet calibration of the lens in units of ???/A, by default None
        ndim : int, optional
            The spatial dimensionality of the ray system perpendicular to propogation.
            The first-order lens matrix will have axes with size 2*ndim, which acounts for the derivatives.
            A 1D element without chromatic contributions will have `ndim=1`.
            A 2D element without chromatic contributions will have `ndim=2`.
        chroma_dim: bool, optional
            Is there a chromatic dimension, by default False
        label : bool, optional
            If the element should be labeled when plotted, by default False
        print_fancy : bool, optional
            If a fancy table should be used when printed, by default True

        To Do
        -----
        TODO: Change ndim to take a list or str with dimension names.
            e.g. 'X', 'XY', 'XYE', 'XE'.
        TODO: Remove pring_fancy.
            Revert back to __repr__ returning a str and add a print_fancy function.

        """
        self.name = name
        self.kind = kind
        self.poles = poles
        self.position = position
        self.length = length
        self.strength = strength
        self.calibration = calibration
        self.ndim = ndim
        self.label = label
        self.print_fancy = print_fancy

        self.z = None

    def __repr__(self) -> str:
        rep = {'name':self.name,
               'kind':self.kind,
               'length':self.length,
               'strength':self.strength,
               'calibration':self.calibration,
               }
        if  self.print_fancy:
            display(DataFrame({key:[value] for key, value in rep.items()}))
            return ''
        else:
            return '\t'.join([f"{key}: {value}, " for key, value in rep.items()])
    def __copy__(self):
        return type(self)(self.name, self.strength,self.calibration, self.label)
    
    def get_s(self,
              z:None|int|float|ArrayLike=None, z0:None|int|float=None,
              store_z=True):
        #check if z is provided to thin lens
        if self.length == 0 and z is not None:
            warn('z was provided for a zero length element and will not be used.') 
            return None #! This may result in no output for transfer matrices. If so need to think about how to handle zero length.
        
        #initialize the initial position
        if z0 is None: z0 = self.position

        #initialize the propogation distance(s)
        if z is None: z = xp.array([self.length]) #length
        elif isinstance(z, int): z = self.length * xp.linspace(0,1,z) #steps
        elif isinstance(z, float): z = xp.array([z])
        elif isinstance(z, ArrayLike): pass #distance or array of distances #TODO: typeerror: Subscripted generics cannot be used with class and instance checks
        else: raise ValueError('Please eneter a vlaid z value.')

        s = z-z0 #propogation distance

        return s

    def transfer_matrix(self,
                         s:int|float|ArrayLike,
                         #type='Hills' TODO: Add `type` in paramaters to describe the type of transfer matrix. Hill's, Twiss, etc.
                         ) -> ArrayLike:
        """Transfer matrix for ray propogation.
        
        The homogenous equaiton of motion approximation leads to a linear solution of $u"+k(s)u=0$ given as $u(s)=C(s)u_0+S(s)u_0', where s is the distance traveled (~z for small u').
        For K>0 $C=cos(\sqrt{Ks})$ and $S=\frac{1}{\sqrt{K}} sin(\sqrt{Ks})$ and for K<0 $C=cosh(\sqrt{|K|s})$ and $S=\frac{1}{\sqrt{|K|}} sinh(\sqrt{|K|s})$.
        The transfer matrix representation is then,
        $$ 
        T = \begin{matrix}
            C & S\\
            C' & S'
            \end{matrix}
        $$

        To Do
        -----
        TODO: Add `type` in paramaters to describe the type of transfer matrix. Hill's, Twiss, etc.
            Might need to move the bulk of the current function to a hidden function (e.g. __transfer_matrix_hills(...)) then call the hidden transfer matrix options.
        TODO: make the z initialization in propogate_ray or leave in here?
        """
        poles = self.poles
        if poles is None:   raise ValueError('The number of poles is not set.')
        elif poles%2 != 0:  raise ValueError(f'Only even number poles are allowed. The current element has {poles:d} poles.')
        elif poles > 4:     raise ValueError('Only multipoles with N<=4 are implemented (i.e. Quadropoles and lower).  The current element has {poles:d} poles.')
        else:               pass
        
        sK = xp.sqrt(xp.abs(self.strength))

        #Calculate transfer matrix.
        m = xp.eye(self.ndim*2)
        return m

    def propogate_ray(self, r0:ArrayLike,
                      z:None|int|float|ArrayLike=None, z0:None|float=0,
                      spectral_included:bool=False):
        if z0 is None: z0 = self.position
        s = self.get_s(z=z, z0=z0)
        m = self.transfer_matrix(s=s)

        if not spectral_included: r0 = xp.pad(r0, ((0,0), (0,2)), constant_values=0)

        rf = xp.einsum('mnz,in->izm', m, r0)
        rf[...,-2] = s

        return rf
        
class Element1D(Element):
    def __init__(self, name:str='Unnamed',
                 kind:None|str=None, poles:None|int=None,
                 position:float=0, length:float=0,
                 strength:float=0, calibration:None|float=None,
                 label:bool=False, print_fancy:bool=True
                 ) -> object:
        """
        TODO: Docstring
        """
        super().__init__(name=name,
                         kind=kind, poles=poles,
                         position=position, length=length,
                         strength=strength, calibration=calibration,
                         ndim=1,
                         label=label, print_fancy=print_fancy)


class Quadripole1D(Element1D):
    def __init__(self, name:str='', 
                 position:float=0, length:float=0,
                 strength:float=0, calibration:float=None,
                 label:float=False, print_fancy:float=True) -> object:
        """1D Quadripole. This effectively acts as a lens in 1D.

        Parameters
        ----------
        name : str, optional
            Name given to the lens, by default ''
        position : float, optional
            The position of the element along the z-axis, by default 0
        length : int, optional
            Length of the element, by default 0
        strength : float, optional
            Defined as the focal length, by default 0
            Note this in not the focusing strength (K) and is simply f.
            A thin lens is defind as KL=-1/fas L goes to zero.
        calibration : float, optional
            Currnet calibration of the lens in units of ???/A, by default None
        label : bool, optional
            If the element should be labeled when plotted, by default False
        print_fancy : bool, optional
            If a fancy table should be used when printed, by default True
        """
        
        if length == 0: kind = 'Thin quad'
        else:           kind = 'Quad'
        super().__init__(name=name,
                         kind=kind, poles=4,
                         position=position, length=length, 
                         strength=strength, calibration=calibration,
                         label=label, print_fancy=print_fancy)
    def transfer_matrix(self,
                         s:int|float|ArrayLike
                         #type='Hills' TODO: Add `type` in paramaters to describe the type of transfer matrix. Hill's, Twiss, etc.
                         ) -> ArrayLike:
        """Transfer matrix for ray propogation.
        
        The homogenous equaiton of motion approximation leads to a linear solution of $u"+k(s)u=0$ given as $u(s)=C(s)u_0+S(s)u_0', where s is the distance traveled (~z for small u').
        For K>0 $C=cos(\sqrt{Ks})$ and $S=\frac{1}{\sqrt{K}} sin(\sqrt{Ks})$ and for K<0 $C=cosh(\sqrt{|K|s})$ and $S=\frac{1}{\sqrt{|K|}} sinh(\sqrt{|K|s})$.
        The transfer matrix representation is then,
        $$ 
        T = \begin{matrix}
            C & S\\
            C' & S'
            \end{matrix}
        $$

        To Do
        -----
        """
        
        if self.length != 0:
            sK = xp.sqrt(xp.abs(self.strength))
            #get trig functions for transfer matrix
            if self.strength>0: #focusing, trig funcitons
                C = xp.cos(sK*s)
                S = 1/sK * xp.sin(sK*s)
                dC = -sK * xp.sin(sK*s)
                dS = C
            elif self.strength<0: #defocusing, hyperbolic trig functions
                C = xp.cosh(sK*s)
                S = 1/sK * xp.sinh(sK*s)
                dC = sK * xp.sinh(sK*s)
                dS = C
            else: #drift
                C = 1
                S = s
                dC = 0
                dS = 1
        elif self.length == 0:
            f = self.strength
            if self.strength != 0: #(de)focusing, trig funcitons
                C = 1
                S = 0
                dC = -1/f
                dS = 1
            else: #off, identity matrix
                C = 1
                S = 0
                dC = 0
                dS = 1

        #Calculate transfer matrix.
        m = xp.array([[C ,  S, 0, 0],
                      [dC, dS, 0, 0],
                      [ 0,  0, 1, 0],
                      [ 0,  0, 0, 1]])
        return m

class Lens1D(Element1D):
    def __init__(self, name:str='', 
                 position:float=0, length:float=0,
                 strength:float=0, calibration:float=None,
                 label:float=False, print_fancy:float=True) -> object:
        """1D round lens.

        Parameters
        ----------
        name : str, optional
            Name given to the lens, by default ''
        position : float, optional
            The position of the element along the z-axis, by default 0
        length : int, optional
            Length of the element, by default 0
        strength : float, optional
            Defined as the focal length, by default 0
            Note this in not the focusing strength (K) and is simply f.
            A thin lens is defind as KL=-1/fas L goes to zero.
        calibration : float, optional
            Currnet calibration of the lens in units of ???/A, by default None
        label : bool, optional
            If the element should be labeled when plotted, by default False
        print_fancy : bool, optional
            If a fancy table should be used when printed, by default True
        """
        
        if length == 0: kind = 'Thin Lens'
        else:           kind = 'Lens'
        super().__init__(name=name,
                         kind=kind, poles=0,
                         position=position, length=length, 
                         strength=strength, calibration=calibration,
                         label=label, print_fancy=print_fancy)
        
    def transfer_matrix(self,
                         s:None|int|float|ArrayLike,
                         #type='Hills' TODO: Add `type` in paramaters to describe the type of transfer matrix. Hill's, Twiss, etc.
                         ) -> ArrayLike:
        """Transfer matrix for ray propogation.
        
        $$ 
        T = \begin{matrix}
            1 & 0\\
            -1/f & 0
            \end{matrix}
        $$
        """
        
        sK = xp.sqrt(xp.abs(self.strength))

        #get trig functions for transfer matrix
        if self.length == 0:            
            f = self.strength
            if self.strength != 0: #(de)focusing, trig funcitons
                C = 1
                S = 0
                dC = -1/f
                dS = 1
            else: #off, identity matrix
                C = 1
                S = 0
                dC = 0
                dS = 1
        else: 
            if self.strength>0: #focusing, trig funcitons
                C = xp.cos(sK*s)
                S = 1/sK * xp.sin(sK*s)
                dC = -sK * xp.sin(sK*s)
                dS = C
            elif self.strength<0: #defocusing, hyperbolic trig functions
                C = xp.cosh(sK*s)
                S = 1/sK * xp.sinh(sK*s)
                dC = sK * xp.sinh(sK*s)
                dS = C
            else: #drift
                C = 1
                S = s
                dC = 0
                dS = 1

        #Calculate transfer matrix.
        m = xp.array([[C ,  S, 0, 0],
                      [dC, dS, 0, 0],
                      [ 0,  0, 1, 0],
                      [ 0,  0, 0, 1]])

        if self.length == 0: m = m[...,None]
        return m


class Drift1D(Element1D):
    def __init__(self, name='',
                 position=0, length=0,
                 label=False, print_fancy=True):
        """General microscope element class.

        Parameters
        ----------
        name : str, optional
            Name given to the lens, by default ''
        length : int, optional
            Length of the element, by default=0
        calibration : float, optional
            Currnet calibration of the lens in units of ???/A, by default None
        label : bool, optional
            If the element should be labeled when plotted, by default False
        print_fancy : bool, optional
            If a fancy table should be used when printed, by default True

        To Do
        -----
        TODO: generalize the class to any dimensions.
            The transfer matrix can probably be generalized by expanding dim and making the eye as the n-dim of the ray.
        """
        super().__init__(name=name,
                         kind='drift', poles=0,
                         position=position, length=length,
                         strength=0, calibration=1,
                         label=label, print_fancy=print_fancy)

    def transfer_matrix(self,
                         s:int|float|ArrayLike=None,
                         #type='Hills' TODO: Add `type` in paramaters to describe the type of transfer matrix. Hill's, Twiss, etc.
                         ) -> ArrayLike:
        """Transfer matrix for ray propogation.
        
        $$ 
        T = \begin{matrix}
            1 & s\\
            0 & 0
            \end{matrix}
        $$
        """

        m = xp.eye(2+2)[...,None]*xp.ones_like(s)[None, None, :]
        m[0,1] = s

        return m
