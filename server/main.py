import os 
import sys 
sys.path.append(os.path.dirname(sys.path[0]))
print(os.path.dirname(sys.path[0]))
import satellite.mapbox as md
from osm.osm import getOSMGraph

from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from SimpleHTTPServer import SimpleHTTPRequestHandler
from SocketServer import ThreadingMixIn
import BaseHTTPServer

class CORSRequestHandler (SimpleHTTPRequestHandler):
    def end_headers (self):
        self.send_header('Access-Control-Allow-Origin', '*')
        SimpleHTTPRequestHandler.end_headers(self)

import logging
import json 
import os 
import os.path  
import scipy.ndimage 
import scipy.misc 
from PIL import Image 
import math 
import cv2
import numpy as np 
from time import time, sleep 
import requests
import threading 
from subprocess import Popen 
import base64 
import math 



global_lock = threading.Lock()
global_lock_var_backend_id = 0 
global_lock_var_backend_ports = [8005, 8006, 8007, 8008, 8009, 8010, 8011, 8012, 8013, 8014, 8015]
global_lock_var_total_num_of_worker = 1
global_lock_var_session = {}
global_lock_var_transaction_num = int(sys.argv[2])

mapbox_cache_folder = "../satellite/mapbox_cache/"
result_folder = "../results/"

Popen("mkdir -p "+ result_folder, shell=True).wait() 
Popen("mkdir -p "+ mapbox_cache_folder, shell=True).wait() 

class S(BaseHTTPRequestHandler):
    # def send_response(self, *args, **kwargs):
    # 	BaseHTTPRequestHandler.send_response(self, *args, **kwargs)
    # 	self.send_header('Access-Control-Allow-Origin', '*')

    def _set_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')                
        #self.send_header('Access-Control-Allow-Origin','http://localhost:8080/')
        self.end_headers()

    def do_OPTIONS(self):           
        self.send_response(200, "ok")       
        self.send_header('Access-Control-Allow-Origin', '*')                
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type")


    def do_GET(self):
        logging.info("GET request,\nPath: %s\nHeaders:\n%s\n", str(self.path), str(self.headers))
        self._set_response()
        self.wfile.write("GET request for {}".format(self.path).encode('utf-8'))

    def do_POST(self):
        global global_lock 
        global global_lock_var_backend_id
        global global_lock_var_backend_ports
        global global_lock_var_total_num_of_worker
        global global_lock_var_transaction_num
        global global_lock_var_session

        content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
        post_data = self.rfile.read(content_length) # <--- Gets the data itself
        logging.info("POST request,\nPath: %s\nHeaders:\n%s\n\n\n",
                str(self.path), str(self.headers))

        return_str = ""
        local_sid = "abc123"

        # try to decode the json request
        #data = json.loads(post_data.decode('utf-8'))
        try:
            #print(post_data)
            data = json.loads(post_data.decode('utf-8'))

        except:
            return_str = json.dumps({"success":"false"})
            print("parse json data failed")
            self._set_response()
            self.wfile.write(return_str.encode('utf-8'))
            return 

        # check if this request is just a 'getstatus' command
        if "cmd" in data and data["cmd"] == "getstatus" :
            status = "Ready"

            if "sessionID" in data:
                sid = data["sessionID"]
                
                global_lock.acquire()
                if sid in global_lock_var_session:
                    status = global_lock_var_session[sid]

                global_lock.release()


            return_str = json.dumps({"success":"true", "status": status})
            self._set_response()
            self.wfile.write(return_str.encode('utf-8'))
            return


        

        # create task and log request
        # get transaction id 
        global_lock.acquire()
        bk_id = global_lock_var_backend_id
        trsc_id = global_lock_var_transaction_num
        global_lock_var_backend_id += 1
        global_lock_var_backend_id %= global_lock_var_total_num_of_worker
        global_lock_var_transaction_num += 1
        
        global_lock_var_session[local_sid] = "Prepareing Images"

        global_lock.release()

        log_folder = result_folder + "/t%d/" % trsc_id
        Popen("mkdir -p "+log_folder, shell=True).wait() 

        data["time"] = time() 
        json.dump(data, open(log_folder+"msg.json", "w"), indent=2)

        with open(log_folder+"content.txt", "w") as fout:
            fout.write(post_data)


        # check input type
        if "inputtype" in data:
            inputtype = data["inputtype"]
        else:
            inputtype = "download"
    
        if inputtype == "download":
            try:
                lat, lon = data["lat"], data["lon"]
            except:
                return_str = json.dumps({"success":"false"})
                print("no lat,lon")
                self._set_response()
                self.wfile.write(return_str.encode('utf-8'))
                return 

        if "sessionID" in data:
            local_sid = data["sessionID"]

        try:
            modelid = data["model_id"]
        except:
            modelid = 0
            
        
        try:
            if "size" not in data:
                tile_size = 704
                data["size"] = 500
                padding = 102
                data["padding"] = 102
                stride = 88
                data["stride"] = 88
            else:
                tile_size = data["padding"] * 2 + data["size"] 
                padding = data["padding"]
                stride = data["stride"]


            if inputtype == "download":

                lat_end = lat + (tile_size - padding) / 111111.0 
                lon_end = lon + (tile_size - padding) / 111111.0 / math.cos(math.radians(lat_end))

                lat_st = lat - padding / 111111.0 
                lon_st = lon - padding / 111111.0 / math.cos(math.radians(lat_end))



                if abs(lat) < 33:
                    zoom = 17
                else:
                    zoom = 16

                if modelid  == 2 or modelid == 3:
                    if abs(lat) < 33:
                        zoom = 18
                    else:
                        zoom = 17

                if "osm" not in data:

                    img, _ = md.GetMapInRect(lat_st, lon_st, lat_end, lon_end, folder = mapbox_cache_folder,  start_lat = lat, start_lon = lon, zoom=zoom)
                    
                    # Popen("rm -r " + cache_folder, shell=True).wait()
                    if modelid  == 2 or modelid == 3:
                        img = scipy.misc.imresize(img.astype(np.uint8), (tile_size*2,tile_size*2))
                    else:
                        img = scipy.misc.imresize(img.astype(np.uint8), (tile_size,tile_size))
                    
                    Image.fromarray(img).save(log_folder+"sat.png")



            else:
                img_base64 = data["imagebase64"]
                data["imagebase64"] = ""
                img_decode = base64.decodestring(img_base64) 
                image_result = open(log_folder+"sattmp." + data["imagetype"], 'wb') # create a writable image and write the decoding result
                image_result.write(img_decode)
                image_result.close()

                img = scipy.ndimage.imread(log_folder+"sattmp." + data["imagetype"])
                dim = np.shape(img) 
                print(dim)

                size = int(max(dim[0], dim[1]) * data["imagegsd"])

                data["size"] = size 
                n = int(math.ceil(size / 176.0))
                data["stride"] = 176
                data["padding"] = (n * 176 - size)//2;


                tile_size = data["padding"] * 2 + data["size"] 
                padding = data["padding"]
                stride = data["stride"]


                if modelid  == 2 or modelid == 3:
                    scale = 2
                else:
                    scale = 1

                img = scipy.misc.imresize(img, (size*scale, size*scale))
                newimg = np.ones((tile_size*scale,tile_size*scale,3), dtype=np.uint8) * 127
                newimg[padding*scale:padding*scale + size*scale,padding*scale:padding*scale + size*scale] = img 

                Image.fromarray(newimg).save(log_folder+"sat.png")

            toBackend = {}
            toBackend["img_in"] = log_folder+"sat.png"
            toBackend["output_json"] = log_folder+"raw"
            toBackend.update(data)

            #url = "http://localhost:%d" % global_lock_var_backend_ports[bk_id]
            url = "http://localhost:%d" % global_lock_var_backend_ports[modelid]

            global_lock.acquire()
            global_lock_var_session[local_sid] = "Running Sat2Graph"
            global_lock.release()

            if "osm" not in data:
                x = requests.post(url, data = json.dumps(toBackend))
                graph = json.loads(x.text) 
            else:
                graph = {}
            
            retdata = {"graph":graph, "success":"true", "taskid":trsc_id}

            if "osm" in data and inputtype == "download":
                Popen("mkdir -p tmp", shell=True).wait()
                osmgraph = getOSMGraph([lat_st, lon_st, lat_end, lon_end], tile_size = tile_size-padding*2, padding = padding)
                retdata["osmgraph"] = osmgraph

            return_str = json.dumps(retdata)

        except:
            return_str = json.dumps({"success":"false"})
            print("load image failed")

        global_lock.acquire()
        global_lock_var_session[local_sid] = "Ready"
        global_lock.release()

        self._set_response()
        self.wfile.write(return_str.encode('utf-8'))

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    pass


#def run(server_class=HTTPServer, handler_class=S, port=8080):
def run(server_class=ThreadedHTTPServer, handler_class=S, port=8080):
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
    # python server.py 8123 1500
    # port 8123 
    if len(argv) >= 2:
        run(port=int(argv[1]))
    else:
        run()


