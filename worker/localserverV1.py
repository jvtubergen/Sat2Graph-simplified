
# python 3
try:
	from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
except:
	from http.server import BaseHTTPRequestHandler, HTTPServer
	
import logging
import json 
import os 
import os.path  
import scipy.ndimage 
import math 
import cv2
import numpy as np 
import tensorflow.compat.v1 as tf 
from time import time 
from subprocess import Popen
import sys 
from decoder import DecodeAndVis, findClearKeypoints
from douglasPeucker import simpilfyGraph, colorGraph
SAT_SCALE = 1
import requests
import pickle 
batchsize = 1
     
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

if sys.argv[1] == "test":
	pass
	# Popen("mkdir -p serverdebug", shell=True).wait()
	# img = cv2.imread("/data/songtao/Sat2GraphLib/download/global_dataset_gt_with_road_types/region_100_sat.jpg")
	# img = img[0:2112,0:2112,:]
	# cv2.imwrite("serverdebug/sat.png", img)
	# toBackend = {}
	# toBackend["img_in"] = "serverdebug/sat.png"
	# toBackend["output_json"] = "serverdebug/raw"
	# toBackend["v_thr"] = 0.05
	# toBackend["e_thr"] = 0.15
	# toBackend["snap_dist"] = 15
	# toBackend["snap_w"] = 100
	# toBackend["size"] = 1000
	# toBackend["padding"] = 28
	# toBackend["stride"] = 176


	# #url = "http://localhost:%d" % global_lock_var_backend_ports[bk_id]
	# url = "http://localhost:8007" 

	# x = requests.post(url, data = json.dumps(toBackend))

	# graph = json.loads(x.text) 

	# print(graph)

	# exit()

else:
	gpu_options = tf.GPUOptions(allow_growth=True)
	tfcfg = tf.ConfigProto(gpu_options=gpu_options)
	tfcfg.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1 # enable xla 

	sess = tf.Session(config=tfcfg)
	
	tf_output = None
	tf_inputsat = None
	tf_istraining = None

def infer(sat):
	global tf_output
	global tf_inputsat
	global tf_istraining
	global sess 

	if tf_output is None:
		if int(sys.argv[1]) == 8006:
			pbfile = "../models/usv1.pb"
		if int(sys.argv[1]) == 8005:
			pbfile = "../models/globalv1.pb"

		with tf.gfile.GFile(pbfile, 'rb') as f:
			graph_def_optimized = tf.GraphDef()
			graph_def_optimized.ParseFromString(f.read())

		for node in graph_def_optimized.node:            
			if node.op == 'RefSwitch':
				node.op = 'Switch'
				for index in range(len(node.input)):
					if 'moving_' in node.input[index]:
						node.input[index] = node.input[index] + '/read'
			elif node.op == 'AssignSub':
				node.op = 'Sub'
				if 'use_locking' in node.attr: del node.attr['use_locking']
			elif node.op == 'AssignAdd':
				node.op = 'Add'
				if 'use_locking' in node.attr: del node.attr['use_locking']

		tf_output = tf.import_graph_def(graph_def_optimized, return_elements=['output:0'])
		tf_inputsat = tf.get_default_graph().get_tensor_by_name('import/input:0')
		tf_istraining = tf.get_default_graph().get_tensor_by_name('import/istraining:0')


	out = sess.run(tf_output, feed_dict={tf_inputsat: sat, tf_istraining: False})
	return out[0]



gt_prob_placeholder = np.zeros((1,352,352,14))
gt_vector_placeholder = np.zeros((1,352,352,12))
gt_seg_placeholder = np.zeros((1,352,352,1))
gt_class_placeholder = np.zeros((1,352,352,3))


class S(BaseHTTPRequestHandler):
	def _set_response(self):
		self.send_response(200)
		self.send_header('Content-type', 'text/html')
		self.end_headers()

	def do_GET(self):
		logging.info("GET request,\nPath: %s\nHeaders:\n%s\n", str(self.path), str(self.headers))
		self._set_response()
		self.wfile.write("GET request for {}".format(self.path).encode('utf-8'))

	def do_POST(self):
		global model 
		global gt_prob_placeholder
		global gt_vector_placeholder
		global gt_seg_placeholder
		global gt_class_placeholder
		global SAT_SCALE

		def progress(x):
			n = int(x * 40)
			sys.stdout.write("\rProgress (%3.1f%%) "%(x*100.0) + ">" * n  + "-" * (40-n)  )
			sys.stdout.flush()

		content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
		post_data = self.rfile.read(content_length) # <--- Gets the data itself
		logging.info("POST request,\nPath: %s\nHeaders:\n%s\n\n\n",
				str(self.path), str(self.headers))

		return_str = ""

		#try:
		data = json.loads(post_data.decode('utf-8'))

		input_file = data["img_in"]
		output_file = data["output_json"]

		v_thr = data["v_thr"]
		e_thr = data["e_thr"]
		snap_dist = data["snap_dist"]
		snap_w = data["snap_w"]

		if "size" not in data:
			data["size"] = 1000
			data["padding"] = 72
			data["stride"] = 88

		tile_size = data["padding"] * 2 + data["size"] 
		padding = data["padding"]
		stride = data["stride"]


		# run the model 

		sat_img = scipy.ndimage.imread(input_file).astype(np.float)
		max_v = 255
		sat_img = (sat_img.astype(np.float)/ max_v - 0.5) * 0.9 
		sat_img = sat_img.reshape((1,tile_size * SAT_SCALE,tile_size * SAT_SCALE,3))

		image_size = 352 

		weights = np.ones((image_size,image_size, 2+4*6 + 2)) * 0.001 
		weights[32:image_size-32,32:image_size-32, :] = 0.5 
		weights[56:image_size-56,56:image_size-56, :] = 1.0 
		weights[88:image_size-88,88:image_size-88, :] = 1.5 

		mask = np.zeros((tile_size, tile_size, 2+4*6 + 2)) + 0.00001
		output = np.zeros((tile_size, tile_size, 2+4*6 + 2))
		input_keypoints = np.zeros((tile_size, tile_size, 1))
		sat_batch = np.zeros((batchsize, 352*SAT_SCALE, 352*SAT_SCALE, 3))
		road_batch = np.zeros((batchsize, 352, 352, 1))
		
		t0 = time()
		coordinates = []

		for x in range(0,tile_size,stride):  # 10
			if x + 352 > tile_size :
				continue

			for y in range(0,tile_size,stride): # 10
				if y + 352 > tile_size :
					continue
				coordinates.append((x,y))
		
		pc = 0
		nPhase = 0
		tc = (len(coordinates) // batchsize) * (nPhase + 1)

		for i in range(0,len(coordinates),batchsize):
			
			N = min(batchsize, len(coordinates)-i)

			for bt in range(N):
				x,y = coordinates[i+bt][0], coordinates[i+bt][1]
				sat_batch[bt,:,:,:] = sat_img[0,x*SAT_SCALE:x*SAT_SCALE+image_size*SAT_SCALE, y*SAT_SCALE:y*SAT_SCALE+image_size*SAT_SCALE,:] 
			
			outputs = infer(sat_batch)

			for bt in range(N):
				x,y = coordinates[i+bt][0], coordinates[i+bt][1]
				mask[x:x+image_size, y:y+image_size, :] += weights
				output[x:x+image_size, y:y+image_size,:] += np.multiply(outputs[bt,:,:,:], weights)

			pc += 1
			progress(min(tc, float(pc)) / tc)


		print("GPU time (pass 1):", time() - t0)
		t0 = time()

		output = np.divide(output, mask)

		# alloutputs  = model.Evaluate(sat_img, gt_prob_placeholder, gt_vector_placeholder, gt_seg_placeholder)
		# output = alloutputs[1][0,:,:,:]

		#graph = DecodeAndVis(output, output_file, thr=0.01, edge_thr = 0.1, angledistance_weight=50, snap=True, imagesize = 704)
		
		fast = True
		graph = DecodeAndVis(output, output_file, thr=v_thr, edge_thr = e_thr, angledistance_weight=snap_w, snap_dist = snap_dist, snap=True, imagesize = tile_size, fast=fast)

		print("Decode time (pass 1):", time() - t0)
		t0 = time()

		graph = simpilfyGraph(graph)
		print("Graph simpilfy time:", time() - t0)
		t0 = time()

		lines = []
		points = []

		biasx = -padding
		biasy = -padding

		def addbias(loc):
			return (loc[0]+biasx, loc[1]+biasy)

		def inrange(loc):
			if loc[0] > padding and loc[0] < tile_size + padding and loc[1] > padding and loc[1] < tile_size + padding:
				return True 
			else:
				return False 

		for nid, nei in graph.items():
			for nn in nei:
				if inrange(nn) or inrange(nid):
					edge = (addbias(nid), addbias(nn))
					edge_ = (addbias(nn), addbias(nid))

					if edge not in lines and edge_ not in lines:
						lines.append(edge)  

			if inrange(nid) and len(nei)!=2:
				points.append(addbias(nid))


		# graph to json 
		def convert(o):
			if isinstance(o, np.int64): return int(o)  
			raise TypeError
		return_str = json.dumps({"graph":[lines, points], "success":"true"}, default=convert)


		# create some visualization here



		# except:
		# 	return_str = json.dumps({"success":"false"})
		# 	print("parse json data failed")


		self._set_response()
		self.wfile.write(return_str.encode('utf-8'))

def run(server_class=HTTPServer, handler_class=S, port=8080):
	logging.basicConfig(level=logging.INFO)
	server_address = ('', port)
	httpd = server_class(server_address, handler_class)
	logging.info('Starting httpd...\n')
	try:
		httpd.serve_forever()
	except KeyboardInterrupt:
		pass
	httpd.server_close()
	logging.info('Stopping httpd...\n')

if __name__ == '__main__':
	from sys import argv
	# 8008 global model
	# 8007 us model
	if len(argv) == 2:
		run(port=int(argv[1]))
	else:
		run()