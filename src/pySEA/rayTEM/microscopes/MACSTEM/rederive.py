import sys
sys.path.insert(1,"../../../../../../../niceplot/")
from niceplot import *

rays = sum([[[x,xt] for x in [-1.5,0,1.5]] for xt in [-1,0,1]],[])
rays = np.asarray(rays) ; z=0
print(rays)

zs = [ [] for i in range(9) ] ; xs = [ [] for i in range(9) ]

def log():
    for i in range(9):
        zs[i].append(z) ; xs[i].append(rays[i,0])
log()

#elements = [["drift",1],["lens",2],["drift",3],["lens",.3],["drift",4]]
#elements = [["drift",1],["lens",.7],["drift",3],["lens",.3],["drift",.7],["lens",.34],["drift",10]]
elements = [["drift",1],["lens",1.5],["drift",3],["lens",1],["drift",4]] ; zd=[2.5,7.0] ; zi=[5.2]

def propagate():
    global z,rays
    for e,v in elements:
        if e=="drift":
            M=[[1,v],[0,1]] ; z+=v
        else:
            M=[[1,0],[-1/v,1]]
        rays = np.einsum('xy,ry->rx',M,rays)
        log()

propagate()
rays = sum([[[x,xt] for x in [-1.5,0,1.5]] for xt in [-1,0,1]],[])
rays = np.asarray(rays)
z=zd[0]
elements = elements[2:] ; elements[0][1]-=(zd[0]-1)
zs = [ [] for i in range(9) ] + zs
xs = [ [] for i in range(9) ] + xs
log()
propagate()
for i in range(9):
    for l in range(len(zs[i])):
        xs[i][l]-=10

mkrs = rainbow(9)+rainbow(9)

for z in zd:
    zs.append([z,z]) ; xs.append([-5,5]) ; mkrs.append("k:")
for z in zi:
    zs.append([z,z]) ; xs.append([-5,5]) ; mkrs.append("k,--")


plot(zs,xs,markers=mkrs,labels=[""]*len(zs))

