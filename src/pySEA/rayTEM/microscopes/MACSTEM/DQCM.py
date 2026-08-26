import sys
sys.path.insert(1,"../../../../")
from pySEA.rayTEM import load_microscope as mic_load,Source,Lens
from pySEA.rayTEM.xmlNion import lookupPositions

microscope = mic_load("macstem")
print("CLs")
mic  = mic_load("macstem_calibratedCL") ; print("MIC\n",repr(mic))
print("OLs")
ros  = mic_load("macstem_calibratedOL") ; print("ROS\n",repr(ros))
print("PLs")
cope = mic_load("macstem_calibratedPL") ; print("COPE\n",repr(cope))

#sys.exit()

i = microscope.index("condenser")
microscope.sections[i]=mic["condenser"]
i = microscope.index("projector")
microscope.sections[i]=cope["projector"]

do_PL1OL2 = cope.get_element_position("PL1") - cope.get_element_position("OL2")
df_PL1OL2 = ros.get_element_position("PL1") - ros.get_element_position("OL2")
dz = do_PL1OL2-df_PL1OL2

microscope["objective"].move("OL2",dz=dz)
microscope["OL2"].calibration = ros["OL2"].calibration

#microscope = microscope["sample":]
#microscope.insert( 0., Source(size=(2e-4/100,0),np_xy=(3,1),angle=(.0001,0),na_xy=(3,0),name="gun") )

positions_file = "lens_positions.txt"

z1 = lookupPositions("DQCM_1O",positions_file)
z2 = lookupPositions("DQCM_4O",positions_file)
z3 = lookupPositions("CCD",positions_file)

z = microscope["projector"].position+microscope["CCD"].position-abs(z3-z1)
l = abs(z2-z1)

microscope.insert(z,Lens(name="DQCM",length=0,strength=0.12345))

#microscope = microscope["sample":]
#microscope.insert( 0., Source(size=(2e-4/100,0),np_xy=(3,1),angle=(.0001,0),na_xy=(3,0),name="gun") )
#microscope.show()

microscope.save("macstem_calibratedFull")
