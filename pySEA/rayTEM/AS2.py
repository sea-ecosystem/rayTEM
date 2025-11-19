import requests
import numpy as np

class AS2querier:
	def __init__(self,ip,port,quiet=False):
		self.url = "http://"+str(ip)+":"+str(port)+"/AS2/controls/"
		self.quiet = quiet
	def query(self,element_name):
		try:
			r = requests.get(self.url+element_name)
			s = r.status_code
		except Exception as e:
			print(e)
			s = 0
		if 0 != 200:
			if not self.quiet:
				print("WARNING: failed response from",self.url,"for element",element_name,"(returning random)")
			return np.random.random()
		v = r.json()["OutputValue"]
		if not self.quiet:
			print("queried:",element_name,"=",v)
		return v
