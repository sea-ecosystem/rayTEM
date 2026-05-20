import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.optimize import minimize
from copy import copy
#from numba import jit

#### Might be elegant but might create other problems
class multipole:
    def __init__(self,kind='multi',name='Unnamed',length=0,strength=0,label=False):
        self.kind = kind
        self.name = name
        self.length = length
        self.strength = strength
        self.label = label

    # Could perhaps look at if is not None...:
    def __repr__(self):
        return(str(self.kind)+'\t'+str(self.name)+'\tL='+str(self.length)+'\tS='+str(self.strength) )
    def __copy__(self):
        return type(self)(self.name, self.length,self.strength, self.label)
        
class quad(multipole):
    def __init__(self,name='',length=0,strength=0,label=False):
        super().__init__(kind='quad',name=name,length=length,strength=strength,label=label)
    def propogate(self,Z,X,Y,A,B,E,step):
        L = self.length
        K = self.strength
        C = np.cos(K*L)
        S = np.sin(K*L)
        Ch = np.cosh(K*L)
        Sh = np.sinh(K*L)
        E[step+1] = E[step]
        Z[step+1] = Z[step] + L
        if K == 0: # Basically a drift
            X[step+1] = X[step] + L*A[step]  
            Y[step+1] = Y[step] + L*B[step]  
            A[step+1] = A[step]
            B[step+1] = B[step]
        elif K>0:
            X[step+1] = X[step]*C    + S/K*A[step]  
            A[step+1] = -X[step]*K*S + C*A[step]  
            Y[step+1] = Y[step]*Ch   + 1/K*Sh*B[step]
            B[step+1] = Y[step]*K*Sh + Ch*B[step]
        elif K<0:
            X[step+1] = X[step]*Ch   + Sh/K*A[step]  
            A[step+1] = X[step]*K*Sh + Ch*A[step]  
            Y[step+1] = Y[step]*C    + 1/K*S*B[step]
            B[step+1] = -Y[step]*K*S + C*B[step]                
    
class drift(multipole):
    def __init__(self,name='',length=0,strength=0,label=False):
        super().__init__(kind='drift',name=name,length=length,strength=strength,label=label)
    def propogate(self,Z,X,Y,A,B,E,step):
        L = self.length
        Z[step+1] = Z[step] + L
        X[step+1] = X[step] + L*A[step]  
        Y[step+1] = Y[step] + L*B[step]  
        A[step+1] = A[step]
        B[step+1] = B[step]
        E[step+1] = E[step]
        
class virt(multipole):
    # Like a drift but don't change the Z position
    def __init__(self,name='',length=0,strength=0,label=False):
        super().__init__(kind='virt',name=name,length=length,strength=strength,label=label)
    def propogate(self,Z,X,Y,A,B,E,step):
        L = self.length
        Z[step+1] = Z[step] 
        X[step+1] = X[step] + L*A[step]  
        Y[step+1] = Y[step] + L*B[step]  
        A[step+1] = A[step]
        B[step+1] = B[step]
        E[step+1] = E[step]

class lens(multipole):
    def __init__(self,name='',length=0,strength=0,label=False):
        super().__init__(kind='lens',name=name,length=length,strength=strength,label=label)
    def propogate(self,Z,X,Y,A,B,E,step):
        strength = np.abs(self.strength)   # Hmmm... Maybe ABS?
        Z[step+1] = Z[step] 
        X[step+1] = X[step]
        Y[step+1] = Y[step]
        E[step+1] = E[step]
        if strength != 0:
            A[step+1] = A[step] - X[step]/strength
            B[step+1] = B[step] - Y[step]/strength
        else:
            A[step+1] = A[step]
            B[step+1] = B[step]
        
class rotate(multipole):
    def __init__(self,name='',length=0,strength=0,label=False):
        super().__init__(kind='sole',name=name,length=length,strength=strength,label=label)
    def propogate(self,Z,X,Y,A,B,E,step):
        L = self.length
        K = self.strength
        # Rotation = K * L      # How to track this?
        C = np.cos(K*L)
        S = np.sin(K*L)
        E[step+1] = E[step]
        Z[step+1] = Z[step] + L
        if K == 0: # Basically a drift
            X[step+1] = X[step] + L*A[step]  
            Y[step+1] = Y[step] + L*B[step]  
            A[step+1] = A[step]
            B[step+1] = B[step]
        else:                                # Note: Fix typo from TRANSPORT also KS = 1/F
            X[step+1] =  X[step]*C*C    + A[step]*S*C/K  + Y[step]*C*S   + B[step]*S*S/K
            A[step+1] =  X[step]*-K*S*C + A[step]*C*C    + Y[step]*-K*S*S + B[step]*S*C
            Y[step+1] =  X[step]*-C*S   + A[step]*-S*S/K + Y[step]*C*C    + B[step]*S*C/K
            B[step+1] =  X[step]*K*S*S  + A[step]*-S*C   + Y[step]*-K*S*C + B[step]*C*C

class sole(multipole):
    def __init__(self,name='',length=0,strength=0,label=False):
        super().__init__(kind='sole',name=name,length=length,strength=strength,label=label)
    def propogate(self,Z,X,Y,A,B,E,step):
        L = self.length
        K = self.strength
        # Rotation = K * L      # How to track this?
        C = np.cos(K*L)
        S = np.sin(K*L)
        Ch = np.cosh(K*L)
        Sh = np.sinh(K*L)
        E[step+1] = E[step]
        Z[step+1] = Z[step] + L
        if K == 0: # Basically a drift
            X[step+1] = X[step] + L*A[step]  
            Y[step+1] = Y[step] + L*B[step]  
            A[step+1] = A[step]
            B[step+1] = B[step]
        else:                                         # Note KS = 1/F
            X[step+1] = X[step]*C    + S/K*A[step] 
            A[step+1] = -X[step]*K*S + C*A[step]  
            Y[step+1] = Y[step]*C   + S/K*B[step]
            B[step+1] = -Y[step]*K*S + C*B[step]

class prism(multipole):
    def __init__(self,name='',length=0,strength=0,label=False):
        super().__init__(kind='prism',name=name,length=length,strength=strength,label=label)
    def propogate(self,Z,X,Y,A,B,E,step):
        L = self.length
        n = self.strength   # Should this be K????
        #n = 0.25           # Hardwire for Nion
        h = np.pi/2/L       # Force to have a 90 degree bend! --- Should be easier to change now
        #k = np.abs(K)
        Kx = np.sqrt((1-n)*h*h)
        Ky = np.sqrt(n*h*h)
        Cx = np.cos(Kx*L)
        Sx = np.sin(Kx*L)
        Cy = np.cos(Ky*L)
        Sy = np.sin(Ky*L)
        Z[step+1] = Z[step] + L
        E[step+1] = E[step]
        if Kx == 0: # Basically a drift
            X[step+1] = X[step] + L*A[step]  
            Y[step+1] = Y[step] + L*B[step]  
            A[step+1] = A[step]
            B[step+1] = B[step]
        else:
            X[step+1] = X[step]*Cx    + Sx/Kx*A[step]  +h/(Kx*Kx)*(1-Cx)*E[step]
            A[step+1] = -X[step]*Kx*Sx + Cx*A[step]    + h/Kx*Sx*E[step]
            Y[step+1] = Y[step]*Cy   + 1/Ky*Sy*B[step]
            B[step+1] = -Y[step]*Ky*Sy + Cy*B[step]

class name(multipole):
    # Essentially a spacer
    def __init__(self,name='',length=0,strength=0,label=False):
        super().__init__(kind='name',name=name,length=length,strength=strength,label=label)
    def propogate(self,Z,X,Y,A,B,E,step):
            # Just use this one in the analysis or plotting loops
            Z[step+1] = Z[step]
            X[step+1] = X[step]  
            Y[step+1] = Y[step]  
            A[step+1] = A[step]
            B[step+1] = B[step]
            E[step+1] = E[step]    

class start(multipole):
    # A scaler such that we can make the beam match an aperture later
    def __init__(self,name='',length=0,strength=0,label=False):
        super().__init__(kind='START',name=name,length=length,strength=strength,label=label)
    def propogate(self,Z,X,Y,A,B,E,step):
        L = self.length           # Probably should be zero
        strength = self.strength
        Z[step+1] = Z[step] + L
        X[step+1] = X[step] + L*A[step] 
        Y[step+1] = Y[step] + L*B[step]
        A[step+1] = A[step] * strength
        B[step+1] = B[step] * strength
        E[step+1] = E[step]

class shiftx(multipole):
    # A shifter in X
    def __init__(self,name='',length=0,strength=0,label=False):
        super().__init__(kind='shiftx',name=name,length=length,strength=strength,label=label)
    def propogate(self,Z,X,Y,A,B,E,step):
        L = self.length           
        strength = self.strength
        Z[step+1] = Z[step] + L
        X[step+1] = X[step] + L*A[step]+strength 
        Y[step+1] = Y[step] + L*B[step]
        A[step+1] = A[step] 
        B[step+1] = B[step] 
        E[step+1] = E[step]

class tiltx(multipole):
    # A tilter in X
    def __init__(self,name='',length=0,strength=0,label=False):
        super().__init__(kind='tiltx',name=name,length=length,strength=strength,label=label)
    def propogate(self,Z,X,Y,A,B,E,step):
        L = self.length           # Probably should be zero
        strength = self.strength
        Z[step+1] = Z[step] + L
        X[step+1] = X[step] + L*A[step] 
        Y[step+1] = Y[step] + L*B[step]
        A[step+1] = A[step] + strength
        B[step+1] = B[step] 
        E[step+1] = E[step]


def plotit(Z,X,Y,A,B,elems):
    fig,ax=plt.subplots(1,dpi=400)
    for iii,init in enumerate(X[0]):
        if False:
            var = iii//3
            if var ==0:  c='r'
            elif var==1:  c='b'
            else: c='g'
            foo = iii%3
            if foo ==0:        ls='dotted'
            elif foo==1:       ls='solid'
            else:              ls='dashed'
            lw=0.5
            ax.plot(Z[:,iii],X[:,iii],c=c,ls=ls,lw=lw, marker='.', markersize=2)
            lw=1
            ax.plot(Z[:,iii],Y[:,iii],c=c,ls=ls,lw=lw, marker='.', markersize=2)
        else:
            if iii in [0,1,2,5,6,7]:  # X-type including E and zero-ray
                c,ls,lw = 'black','solid',1
                if iii == 1:
                    c,ls,lw = 'red','solid',1
                elif iii == 2:
                    c,ls,lw = 'red','solid',0.5            
                if iii == 6:
                    c,ls,lw = 'green','solid',1
                elif iii == 7:
                    c,ls,lw = 'green','solid',0.5
                elif iii == 5:    # This is the energy - oddball
                    c,ls,lw = 'darkgrey','dashed',0.5
                ax.plot(Z[:,iii],X[:,iii],c=c,ls=ls,lw=lw, marker='.', markersize=2)
            else:
                # iii in [3,4,8,9]            # These are really the Y-types
                if iii == 3:
                    c,ls,lw = 'blue','dotted',1.25  
                if iii == 4:
                    c,ls,lw = 'blue','dotted',0.75  
                if iii == 8:
                    c,ls,lw = 'orange','dotted',1.25  
                if iii == 9:
                    c,ls,lw = 'orange','dotted',0.75  
                ax.plot(Z[:,iii],Y[:,iii],c=c,ls=ls,lw=lw, marker='.', markersize=2)
                
    # Now label the poles
    maxx = np.max(X)
    for step,elem in enumerate(elems): # np.arange(1,steps):
        if elem.kind == 'prism':
            ZZZ = Z[step+1][0]-Z[step][0]
            ax.add_patch(Rectangle((Z[step][0], -maxx),ZZZ,maxx*2,facecolor='yellow',edgecolor='orange',alpha=0.5))
        if elem.kind ==  'sole':
            ZZZ = Z[step+1][0]-Z[step][0]
            ax.add_patch(Rectangle((Z[step][0], -maxx),ZZZ,maxx*2,facecolor='lightgrey',edgecolor='red',alpha=0.5))
        if elem.kind == 'lens':
            xx,yy = Z[step][0], 0.95*maxx #np.max(X[step])
            # Think about how to plot
            ax.add_patch(Rectangle((xx,-yy),0,yy*2,edgecolor ='grey',facecolor='lightgrey',lw=2,alpha=0.5))
        if elem.kind == 'quad':
            ZZZ = Z[step+1][0]-Z[step][0]
            L = elem.length
            K = elem.strength
            if K>0: ecol,fcol = 'blue','lightblue'
            if K<0: ecol,fcol = 'red', 'pink'
            if K==0: ecol,fcol = 'green','lightgreen'
            ax.add_patch(Rectangle((Z[step][0], -maxx),ZZZ,maxx*2,facecolor=fcol,edgecolor=ecol,alpha=0.5)) # same; alpha?
        if elem.kind == 'name':
            xx,yy = Z[step][0], 0.95*maxx #np.max(X[step])
            ax.annotate(
                str(elem.name),xy=(xx,yy+maxx*0.05), xycoords='data',
            xytext= (xx,yy+maxx*0.2),
            arrowprops=dict(alpha=0.5,facecolor='grey',edgecolor='grey',headwidth=4,width=1,headlength=4),
                horizontalalignment='center', verticalalignment='top',size=5)
        if elem.label==True:
            xx,yy = Z[step][0], 0.95*maxx #np.max(X[step])
            ax.annotate(
                str(elem.name),xy=(xx,yy+maxx*0.05), xycoords='data',
            xytext= (xx,yy+maxx*0.13),
                horizontalalignment='center', verticalalignment='top',size=8)

    plt.show()


def propogate2D(elems,X0,Y0,A0,B0,E0):    
    steps = (len(elems))+1
    X = np.zeros((steps,len(X0)))
    #print(steps, X0.shape, X.shape)
    Y = np.zeros_like(X)
    A = np.zeros_like(X)
    B = np.zeros_like(X)
    Z = np.zeros_like(X)
    E = np.zeros_like(X)
    X[0] = X0
    Y[0] = Y0
    A[0] = A0
    B[0] = B0
    E[0] = E0
    try:        
        for step,what in enumerate(elems): # np.arange(1,steps):
            what.propogate(Z,X,Y,A,B,E,step)
        return(Z,X,Y,A,B,E)
    except AttributeError as AE:
        print(AE)
        print(step,what)
    except TypeError as TE:
        print(TE)
        print(step,what)

def update_elems(elems_ext,variables,namesof):
    # Change some of the elements
    changeables = {}
    for i,variable in enumerate(variables):
        varname = namesof[i]
        changeables[varname]=variable
        changeables['-'+varname]=-1*variable  # Hacky!
    #print(changeables)
    elems_int = []
    for elem in elems_ext:     # Copy list
        length = elem.length
        strength = elem.strength
        name = elem.name
        new_elem = copy(elem)
        if isinstance(strength,str):
            try:
                strength = changeables[strength]
            except KeyError:
                pass
        if isinstance(length,str):
            try:
                length = changeables[length]
            except KeyError:
                pass
        new_elem.length = length
        new_elem.strength = strength
        elems_int.append(new_elem) # Simple copy
    return(elems_int)

def obj_fun(variables,elems_ext,namesof,X0,Y0,A0,B0,E0):
    elems_fun = update_elems(elems_ext,variables,namesof)
    Z,X,Y,A,B,E = propogate2D(elems_fun,X0,Y0,A0,B0,E0)
    return( 
           10*(X[2,1]-30e-6)**2
        #+(X[18,1]-Y[18,3] )**2
        +(X[18,6]**2 + Y[18,8]**2)
        #+(X[18,1]-Y[18,3])**2
        +(X[33,6]**2 + Y[33,8]**2)
        #+(A[33,1])**2
        +(X[48,6]**2+Y[48,8]**2)
          )