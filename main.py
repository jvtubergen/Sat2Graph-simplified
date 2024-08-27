import logging
import json 
import os 
import os.path  
import math 
import numpy as np 
import tensorflow.compat.v1 as tf 
from time import time, strftime
from subprocess import Popen
import sys 
from decoder import DecodeAndVis, findClearKeypoints
from douglasPeucker import simpilfyGraph, colorGraph
# from mrf import mrf
import requests
import pickle 
from PIL import Image
from gmaps.lib import *

# Constants.
SAT_SCALE = 2
batchsize = 1
fast = True


tf_state = {
    "is_initiated": False
}


# Infer satellite/road data.
def infer(sat, road):
    global tf_state
    if not tf_state["is_initiated"]:
        
        print("Loading TF:")
        print("* GPU properties.")
        # GPU properties.
        gpu_options = tf.GPUOptions(allow_growth=True)
        tfcfg = tf.ConfigProto(gpu_options=gpu_options)
        tfcfg.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1 # enable xla 

        print("* Session.")
        # Model (tensorflow) setup.
        sess = tf.Session(config=tfcfg)
        pbfile = "./models/globalv2.pb"

        print("* Optimized state.")
        with tf.gfile.GFile(pbfile, 'rb') as f:
            graph_def_optimized = tf.GraphDef()
            graph_def_optimized.ParseFromString(f.read())

        print("* Loading nodes.")
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
            
        
        print("* Loading Tensors.")
        tf_state["sess"]       = sess
        tf_state["output"]     = tf.import_graph_def(graph_def_optimized, return_elements=['output:0'])

        print("* Listing Tensors:")
        graph = tf.get_default_graph()
        for op in graph.get_operations():
            print(f"  Operation: {op.name}")
            for tensor in op.outputs:
                print(f"   - Tensor: {tensor.name}")

        tf_state["inputsat"]   = tf.get_default_graph().get_tensor_by_name('inputsat:0')
        tf_state["inputroad"]  = tf.get_default_graph().get_tensor_by_name('inputroad:0')
        tf_state["istraining"] = tf.get_default_graph().get_tensor_by_name('istraining:0')

        tf_state["is_initiated"] = True
    
    tf_output     = tf_state["output"]    
    tf_inputsat   = tf_state["inputsat"]  
    tf_inputroad  = tf_state["inputroad"] 
    tf_istraining = tf_state["istraining"]
    sess          = tf_state["sess"]

    out = sess.run(tf_output, feed_dict={tf_inputsat: sat, tf_inputroad: road, tf_istraining: False})
    return out[0]


# Printing progres..
def progress(x):
	n = int(x * 40)
	sys.stdout.write("\rProgress (%3.1f%%) "%(x*100.0) + ">" * n  + "-" * (40-n)  )
	sys.stdout.flush()


# graph to json 
def convert(o):
    if isinstance(o, np.int64): return int(o)  
    raise TypeError


def addbias(loc):
    global biasx
    global biasy
    return (loc[0]+biasx, loc[1]+biasy)


def inrange(loc):
    global padding
    global tile_size_w
    global tile_size_h
    if loc[0] > padding and loc[0] < tile_size_h + padding and loc[1] > padding and loc[1] < tile_size_w + padding:
        return True 
    else:
        return False 


# Infer road network from satellite image.
def infer_network(sat_img, output_file):
    global padding
    global tile_size_w
    global tile_size_h


    # Decode and visualization parameters.
    v_thr     = 0.05
    e_thr     = 0.01
    snap_dist = 15
    snap_w    = 100


    # Image (Assume image in ~1 GSD.)
    # axis 0 = rows = height, axis 1 = columns = width.
    im_height = sat_img.shape[0]
    im_width  = sat_img.shape[1]

    print("h:", im_height)
    print("w:", im_width)

    # Ensure width and height is a multiple of SAT_SCALE
    assert im_height % SAT_SCALE == 0
    assert im_width % SAT_SCALE == 0

    # Image cutting/processing parameters.
    tile_size_w= int(im_width / SAT_SCALE)
    tile_size_h= int(im_height / SAT_SCALE)
    stride     = 88        # Step after each inferrence (half the inferrence window size).
    padding    = 176       # Seems more logical to relate padding to stride.
    image_size = 352       # 4 * stride
    feat_size  = 2+4*6 + 5 # Feature size? (6 edge probabilities)


    # Normalize and reformat image (for inferrence code).  
    max_v = 255
    sat_img = (sat_img / max_v - 0.5) * 0.9 
    sat_img = sat_img.reshape((1,tile_size_h * SAT_SCALE,tile_size_w * SAT_SCALE,3))


    # Would this be the (six directions) edge weights?
    weights = np.ones((image_size,image_size, feat_size)) * 0.001 
    weights[32:image_size-32,32:image_size-32, :] = 0.5 
    weights[56:image_size-56,56:image_size-56, :] = 1.0 
    weights[88:image_size-88,88:image_size-88, :] = 1.5 


    # Other 
    mask            = np.zeros((tile_size_h, tile_size_w, feat_size)) + 0.00001
    output          = np.zeros((tile_size_h, tile_size_w, feat_size))
    input_keypoints = np.zeros((tile_size_h, tile_size_w, 1))
    sat_batch       = np.zeros((batchsize, image_size*SAT_SCALE, image_size*SAT_SCALE, 3))
    road_batch      = np.zeros((batchsize, image_size, image_size, 1))
        
    rsx = list(range(0, tile_size_w - image_size, stride))
    rsy = list(range(0, tile_size_h - image_size, stride))
    coordinates = [(y,x) for x in rsx for y in rsy]
    # print("nr of coordinates: ", len(coordinates))
        
    nPhase = 2
    pc = 0 # Current iteration counter.
    tc = (len(coordinates) // batchsize) * nPhase # Total iterations.

    for phase in range(nPhase): # One additional iteration for updating the graph.

        # Infer
        mask *= 0.0 
        mask += 0.00001
        output *= 0.0

        t0 = time()

        for i in range(0,len(coordinates),batchsize):

            N = min(batchsize, len(coordinates)-i)

            for bt in range(N):
                y,x = coordinates[i+bt][0], coordinates[i+bt][1]
                sat_batch[bt,:,:,:] = sat_img[0,x*SAT_SCALE:x*SAT_SCALE+image_size*SAT_SCALE, y*SAT_SCALE:y*SAT_SCALE+image_size*SAT_SCALE,:] 
                if phase > 0: # Skip setting road info on first iteration.
                    road_batch[bt,:,:,:] = input_keypoints[y:y+image_size, x:x+image_size,:] 
                
            outputs = infer(sat_batch, road_batch)
            for bt in range(N):
                y,x = coordinates[i+bt][0], coordinates[i+bt][1]
                mask[y:y+image_size, x:x+image_size, :] += weights
                output[y:y+image_size, x:x+image_size,:] += np.multiply(outputs[bt,:,:,:], weights)

            pc += 1
            progress(min(tc, float(pc)) / tc)

        # print("GPU time (pass %d):"% (phase+1), time() - t0)
        t0 = time()

        output = np.divide(output, mask)
        # print("Decode time (pass %d):" % (phase+1), time() - t0)

        # Extracting keypoints.
        # Convert inferred into road network
        # Resampling keypoints, updating road network.
        graph = DecodeAndVis(output, output_file+"pass%d" % phase, thr=v_thr, edge_thr = e_thr, angledistance_weight=snap_w, snap_dist = snap_dist, snap=True, fast=fast, w=tile_size_w, h=tile_size_h)
        input_keypoints = np.zeros((tile_size_h, tile_size_w, 1))
        dim = (tile_size_h, tile_size_w)
        for x,y in graph.keys():
            if x > 3 and x < dim[1]-3:
                for _x in range(x-1,x+2):
                    for _y in range(y-1,y+2):
                        input_keypoints[_x,_y,0] = 1.0
        # We only need graph to derive input_keypoints and resultingly road_batch use in subsequent inferrences.

    return graph, output


# Post-processing inferred road network.
def post_process(graph, image):
    global padding
    global biasx
    global biasy

    t0 = time()
    graph = simpilfyGraph(graph)
    graphcolor = colorGraph(graph, image[:,:,2+4*6:2+4*6+3])
    # try:
    #     mrf_result = mrf(graph, graphcolor) 
    #     graphcolor = mrf_result
    # except Exception as e:
    #     print("mrf failed:")
    #     print(e)
    #     print("")

    print("Simplification 1:", time() - t0)

    t0 = time()

    lines = []
    points = []

    biasx = -padding
    biasy = -padding

    for nid, nei in graph.items():
        for nn in nei:
            if inrange(nn) or inrange(nid):

                edgecolor = np.argmax(graphcolor[(nid, nn)])
                edge = (addbias(nid), addbias(nn), edgecolor)
                edge_ = (addbias(nn), addbias(nid), edgecolor)

                if edge not in lines and edge_ not in lines:
                    lines.append(edge)  

        if inrange(nid) and len(nei)!=2:
            points.append(addbias(nid))

    print("Simplification 2:", time() - t0)

    return lines, points


def read_image(filename):
    image = Image.open(filename)
    image = image.convert("RGB")
    image = np.array(image)
    return image.astype(float)


def run():

    input_file    = "./sample.png"
    result_folder = "./results/" + strftime("%Y-%m-%d-%H:%M")
    # result_folder = "./results/" + "2024-07-08-13:45"
    output_file   = result_folder  + "/" + "infer"
    Popen("mkdir -p "+ result_folder, shell=True).wait() 

    # Infer
    sat_img = read_image(input_file)
    graph, image = infer_network(sat_img, output_file)
    # with open(name, 'rb') as file:
    #     graph = pickle.load(graph, open(output_file+"_graph.p","w"))

    # Post-process
    lines, points = post_process(graph, image)

    # Store edges and vertices as JSON.
    # Python 3 does not support numpy objects out of the box, requires to manually adapt JSON encoder [source](https://stackoverflow.com/a/65151218).
    def np_encoder(object):
        if isinstance(object, np.generic):
                return object.item()

    json.dump({"graph":{"edges": lines, "vertices": points}}, open(result_folder + "/graph.json", "w"), indent=2, default=np_encoder)





# run()


# Example (Retrieve/Construct image with GSD ~0.88 between two coordinates):
def run_example():
    upperleft  = (41.799575, -87.606117)
    lowerright = (41.787669, -87.585498)
    scale = 1
    zoom = 17 # For given latitude and scale results in gsd of ~ 0.88
    api_key = read_api_key()
    # superimage = construct_image(upperleft, lowerright, zoom, scale, api_key)   # Same result as below.
    superimage, coordinates = construct_image(upperleft, lowerright, zoom-1, scale+1, api_key) # Same result as above.
    write_image(superimage, "superimage.png")

run_example()