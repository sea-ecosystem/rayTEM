import xml.etree.ElementTree as ET

# Written to parse Nion AS2 state-save xml files. other microscopes will need another function, and live-reading from AS2 will need another function
# see TWP-settingsAnalyzer.py for another example of how this works
# XML STRUCTURE:
# root
# > Control (e.g. name="S_Condensers" or "S_Projectors". "local" attribute contains index to active setting)
# > > Settings (e.g. name="30mrad15iRef", "value" attribute contains the same index)
# > Vector (e.g. dependency="S_Condensers" and control="S_Condensers")
# > > CalPoint (e.g. value=[index from "local" above])
# > > > Drive (e.g. driven="C1 ConstW", contains child Rpn with drive value)
# > Vector (e.g. control="C1 ConstW") [this is the "child" Drive pointing to the elments]
# > > CalPoint( no value )
# > > > Drive (e.g. driven="BP1_C2r", contains a child Rpn with equation for drive)

log=open("xmlNion.log",'w')
def nionSettingsDict(filename="",settings={},active={},reload=False):
	# ON FIRST RUN, WE WILL READ THE XML. ON SUBSEQUENT QUERIES, WE WILL USE THE PRE-LOADED DICT
	if len(settings)==0 or reload:
		tree = ET.parse(filename)
		root = tree.getroot()

		# LOOP CONTROLS, WHICH MERELY LIST NESTED SETTINGS
		for control in root.findall('.//Control'): # https://stackoverflow.com/questions/50077734/python-find-tags-in-deep-nodes-xml
			control_name = control.get('name') ; active_setting_id = control.get('local')
			if control_name[:2] != "S_":
				continue
			settings[control_name]={}
			active[control_name]=None
			log.write("found Control "+control_name+", with active setting id "+str(active_setting_id)+"\n")
			# LOOP NESTED SETTINGS TO IDENTIFY NAME OF ACTIVE SETTING
			for setting in control.findall('.//Setting[@value]'): # https://stackoverflow.com/questions/55906438/python-elementtree-how-to-find-all-elements-in-xml-with-certain-attribute
				setting_name=setting.get("name")
				setting_id = setting.get('value')
				if setting_id == active_setting_id: # Setting>value must match Control>local (this is a setting index)
					active[control_name]=setting_name
				#print("[Setting] > > name="+setting_name+" & ID="+setting_id)
				settings[control_name][setting_name]={}
				log.write("Setting "+setting_name+" id "+str(setting_id)+" is under Control "+control_name+{True:" [active]",False:""}[setting_id == active_setting_id]+"\n")

				# LOOP THROUGH OUTER Vector>CalPoint>Drives TO FIND VALUES FOR ACTIVE SETTING
				for vector in root.findall('.//Vector[@dependancy]'):
					#print(vector,vector.get('dependancy'))
					if vector.get('dependancy')!=control_name: # vector>dependency must match Control>name
						continue
					#print("[Vector] > ",vector)
					for calpoint in vector.findall(".//CalPoint[@value]"):
						if calpoint.get('value')!=setting_id: # CalPoint>value must match Control>local (this is a setting index)
							continue
						#print("[Vec>CaP] for Control "+control_name+" & settingID "+setting_id)
						for drive in calpoint.findall(".//Drive[@driven]"):
							driven = calpoint.get("driven")
							value = drive.find("Rpn").text
							drive_name=drive.get("driven")
							#print("[Vec>CaP] > drive="+drive_name+" > value="+value)
							settings[control_name][setting_name][drive_name]=float(value)
							log.write("Driven "+drive_name+" = "+str(value)+" (under Control "+control_name+" > Setting "+str(setting_name)+"\n")

		# I ASSUME GLOBALS ARE THE MAIN-LEVEL CONTROL ENTRY?????
		settings["global"]={"global":{}}
		active["global"]="global"
		for control in root.findall('.//Control'):
			name=control.get('name')
			value=control.get('target')
			settings["global"]["global"][name]=float(value)

	return settings,active

# level="R" for "root", we will return a list of controls
# level="C" for "control" and a control name (e.g. "S_Projectors") and we will return a list of settings
# level="S" for "setting" and a path (e.g. "S_Projectors/_Diffn 20mm (ref)") and we will return a list of drives
# level="D" for "drive" and a path (e.g. "S_Projectors/_Diffn 20mm  (ref)/PV2_1Da" and we will return a value
# setting key "active" is also allowed: level="D" and path="S_Projectors/active/PV2_1Da" should automatically select "_Diffn 20mm  (ref)"
# this function is used by lookupCurrentStrengthsXML with the "active" setting key to find the *current value* for the named drive
def rootControlSettingValue(level="R",path="",activeOnly=False,filename="",settings={},active={},reload=False):

	settings,active = nionSettingsDict(filename,reload=reload)

	# PARSE THE PATH TO RETURN EITHER THE KEYS FOR A GIVEN LEVEL, OR THE FINAL END VALUE
	path=path.split("/")
	if level=="R":
		return settings.keys() # list of control_names
	if level=="C":
		return settings[path[0]].keys(),active[path[0]] # list of setting_names, and the name of the active setting
	if level=="S":
		if path[1]=="active":		# caller has requested the active setting, so look it up
			path[1]=active[path[0]]
		return settings[path[0]][path[1]].keys() # list of drive names
	if level=="D":
		if path[1]=="active":		# caller has requested the active setting, so look it up
			path[1]=active[path[0]]
		return settings[path[0]][path[1]][path[2]] # list of drive value

# WHEREVER THE DRIVE IS, RETURN IT'S TOTAL VALUE (whichever control, from the active setting, plus the global value)
def lookupCurrentStrengthsXML(requested_drive,filename,settings={},reload=False): # settings will hold: section (control) > setting > drive = value, and be populated on the first run
	tree = ET.parse(filename)
	root = tree.getroot()

	val=0
	
	for c in rootControlSettingValue(level="R",filename=filename,reload=reload): # for each control
		settings,s = rootControlSettingValue(level="C",path=c,filename=filename) # get active control
		if s is None:
			continue
		for d in rootControlSettingValue(level="S",path=c+"/active",filename=filename): # automatically jump to active setting within this control, and loop through drives
			if d==requested_drive:
				v=rootControlSettingValue(level="D",path=c+"/active/"+d,filename=filename)
				print(c,">",s,">",d,"=",v)
				val+=v
	val+=rootControlSettingValue(level="D",path="global/global/"+d,filename=filename)

	return val

def lookupPositions(requested_drive,filepath,settings={}):
	if len(settings)==0:
		lines=open(filepath).readlines()
		for l in lines:
			if len(l)<3 or l[0]=="#":
				continue
			d,v = l.split()
			settings[d]=float(v)
	return settings[requested_drive]
