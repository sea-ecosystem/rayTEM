try:
    import cupy as xp
    flag_gpu = True
    from cupy.typing import ArrayLike
except:
    import numpy as xp
    flag_gpu = False
    from numpy.typing import ArrayLike

from pandas import DataFrame

class MicroscopeSection:
    """ 
    TODO: Document

    To Do
    -----
    TODO: Remove pring_fancy.
        Revert back to __repr__ returning a str and add a print_fancy function.
    """
    def __init__(self, name:str='',
                 elements:ArrayLike=None, 
                 position:float=0.,
                 print_fancy:bool=True) -> object:
        self.name = name
        self.elements = elements
        self.position = position
        self.print_fancy = print_fancy

        self.ndim = None

        self.length = 0#xp.sum([e.length for e in self.elements])
        
        for ele in elements:
            ele.position = self.position + self.length
            self.length += ele.length
    
    def __repr__(self) -> str:
        if self.elements is None:
            return ''
        else:
            columns=['name', 'kind', 'length', 'strength', 'calibration']
            reps = [[e.name, e.kind, e.length, e.strength, e.calibration] for e in self.elements]
            
            if  self.print_fancy:
                display(DataFrame(reps, columns=columns))
                return ''
            else:
                return '\n'.join(['\t'.join([f"{key}: {value}, " for key,value in zip(columns,e)])for e in reps])

    def propogate_ray(self, r0:ArrayLike,
                       z:None|int|float=None):
        """
        To do
        -----
        #TODO: Allow for an array to be passed to z.
        """
        ri = self.elements[0].propogate_ray(r0, z=z)
        for ele in self.elements[1:]:
            ele_ri = ele.propogate_ray(ri[:,-1], z=z)
            if ele.length == 0:
                ri[:,-1] = ele_ri[:,-1]
            else:
                ri = xp.append(ri, ele_ri, axis=1)
        
        ri = xp.append(r0[:,None,:], ri, axis=1)
        return ri

    def propogate(self, input:ArrayLike, zs:None|float|int|ArrayLike=None,
                   output_structure:str='per layer') -> ArrayLike:
        """Propogate the input through the microscope section.

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
        output_structure : str
            How to return the output, by default 'per layer'
            'per layer', list with propogation in each element.
            'collapsed', single array.
            'last',      the last transformation during propocation.

        Returns
        -------
        ArrayLike
            Matricies during propogation.
        """
        if input is None:
            input = xp.zeros((self.ndim*2,1))
            input[0] = 1
        output = [xp.asarray([input])]
        
        #lzs = self.get_scaled_z(zs, allow_array=True)
        
        for e in self.elements:
            output.append(e.propogate(output[-1][-1], zs=zs))

        if output_structure == 'per layer': return output
        elif output_structure == 'collapsed': return xp.vstack(output)
        elif output_structure == 'last': return output[-1]
        else: ValueError('An improper `output_structure` was requested.')

        return output

class MicroscopeSection1D(MicroscopeSection):
    """ 
    TODO: Document
    """
    def __init__(self, name:str='', 
                 elements:ArrayLike=None, 
                 position:float=0., 
                 print_fancy:bool=True) -> object:
        super().__init__(name=name, elements=elements, position=position, print_fancy=print_fancy)

        self.ndim = 1