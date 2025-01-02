try:
    import cupy as xp
    flag_gpu = True
    from cupy.typing import ArrayLike
except:
    import numpy as xp
    flag_gpu = False
    from numpy.typing import ArrayLike

from pandas import DataFrame


class Element:
    def __init__(self, kind:None|str=None, name:str='Unnamed',
                 position:float=0, length:float=0,
                 strength:float=0, calibration:None|float=None,
                 ndim:int=3,
                 label:bool=False, print_fancy:bool=True) -> object:
        """General microscope element class.
        $$ 
        T = \begin{matrix}
            C & S\\
            C' & S'
            \end{matrix}
        $$
        where
        $$C=cos(\sqrt{Kl}) \therefore C'=-\sqrt{K}sin({\sqrt{Kl}})$$
        $$S=\frac{1}{\sqrt{K}}sin(\sqrt{Kl}) \therefore S'=sin({\sqrt{Kl}})$$

        Parameters
        ----------
        kind : stry, optional
            Type of element, by default None
        name : str, optional
            Name given to the lens, by default ''
        position : float, optional
            The position of the element along the z-axis, by default 0
        length : int, optional
            Length of the element, by default 0
        strength : float, optional
            Defined as the The focusing strength (K) of a thin lens, by default 0
        calibration : float, optional
            Currnet calibration of the lens in units of ???/A, by default None
        ndim : int, optional
            The dimensionality of the ray system. The first-order lens matrix will have axes with size 2*ndim, which acounts for the derivatives.
            A 1D element without chromatic contributions will have `ndim=1`.
            A 2D element without chromatic contributions will have `ndim=2`.
            A 2D element with chromatic contributions will have `ndim=3`.
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
        self.kind = kind
        self.name = name
        self.position = position
        self.length = length
        self.strength = strength
        self.calibration = calibration
        self.ndim = ndim
        self.label = label
        self.print_fancy = print_fancy


    # Could perhaps look at if is not None...:
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
    
    def get_scaled_z(self, zs, allow_array=False):
        if zs is None: lzs = self.length
        elif isinstance(zs, float): lzs = self.length * zs
        elif allow_array:
            if isinstance(zs, int): lzs = self.length * xp.linspace(0,1,zs)
            elif isinstance(zs, ArrayLike): lzs = self.length * zs
        else: ValueError(f'Transform recieved an incorrect type. Recieved type {type(zs)}.')
        return lzs
   
    def transform(self, input:ArrayLike, zs:None|float) -> ArrayLike:
        """Transform the input through the microscope section.

        Parameters
        ----------
        input : ArrayLike
            Initial array to transform.
        zs : None | int , optional
            Scaled propogation positions, by default None
            The positions (or created ones) are scaled from 0-1, with 0 being the start of the lens and 1 the total length.
            If None,      a signle tranformation at the length of the element is performed.
            If float,     a scaled position.

        Returns
        -------
        ArrayLike
            Matrix after transformation.
        """
        lzs = self.get_scaled_z(zs)
        
        T = xp.array([[1, lzs],
                      [0, 1]
                      ])
        
        return T@input

    def propogate(self, input:ArrayLike, zs:None|float|int|ArrayLike=None) -> ArrayLike:
        """Propogate the input through the element.

        Parameters
        ----------
        input : ArrayLike
            Initial array to transform.
        zs : None | float | int | ArrayLike, optional
            Scaled propogation positions, by default None
            The positions (or created ones) are scaled from 0-1, with 0 being the start of the lens and 1 the total length.
            If None,      a signle tranformation at the length of the element is performed.
            If float,     a scaled position.
            If int,       an array of size z from 0-1 is created.
            If ArrayLike, the input array is used as is.

        Returns
        -------
        ArrayLike
            Matricies during propogation.
        """
        lzs = self.get_scaled_z(zs, allow_array=True)

        output = xp.asarray([self.transform(input, zs=s) for s in xp.asarray([lzs]).squeeze()])
        return output

class Lens1D(Element):
    def __init__(self, name:str='', 
                 strength:float=0, calibration:float=None,
                 label:float=False, print_fancy:float=True) -> object:
        """Infinitly thin lens.
        $$ 
        T = \begin{matrix}
            1 & 0\\
            -1/f & 0
            \end{matrix}
        $$

        Parameters
        ----------
        name : str, optional
            Name given to the lens, by default ''
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
        super().__init__(kind='lens',name=name,length=0, strength=strength, calibration=calibration,label=label, print_fancy=print_fancy)
    
    def transform(self, input:ArrayLike, zs:None|float) -> ArrayLike:
        """Transform the input through the element.

        Parameters
        ----------
        input : ArrayLike
            Initial array to transform.
        zs : None | int , optional
            Scaled propogation positions, by default None
            Meaningless for this zero lengthed element and will not be used, but the input is retained for consistency.

        Returns
        -------
        ArrayLike
            Matrix after transformation.
        """
        
        T = xp.array([[1, 0],
                      [-1/self.strength, 1]
                      ])
        
        return T@input

class Drift1D(Element):
    def __init__(self, name='', 
                 length=0,
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
        """
        super().__init__(kind='drift', name=name, length=length, strength=0, ndim=1, label=label, print_fancy=print_fancy)

    def transform(self, input:ArrayLike, zs:None|float) -> ArrayLike:
        """Transform the input.

        Parameters
        ----------
        input : ArrayLike
            Initial array to transform.
        z : None | int , optional
            Scaled propogation positions, by default None
            The positions (or created ones) are scaled from 0-1, with 0 being the start of the lens and 1 the total length.
            If None,      a signle tranformation at the length of the element is performed.
            If float,     a scaled position.

        Returns
        -------
        ArrayLike
            Matrix after transformation.
        """
        lzs = self.get_scaled_z(zs)
        
        T = xp.array([[1, lzs],
                      [0, 1]
                      ])
        
        return T@input
