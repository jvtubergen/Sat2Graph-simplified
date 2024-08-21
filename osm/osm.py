import cv2 as cv 
import sys
import numpy as np
from subprocess import Popen
import math
import time
import os.path
import scipy.ndimage
import scipy.misc
import pickle, socket
from PIL import Image
import xml.etree.ElementTree
from time import sleep, time 
import graph_ops as graphlib 

class OSMLoader:
    def __init__(self, region, noUnderground = False, osmfile=None, includeServiceRoad = False, useblacklist = False, allinfo = False):
        self.allinfo = allinfo 
        sub_range = str(region[1])+","+str(region[0])+","+str(region[3])+","+str(region[2])

        #Popen("mkdir -p tmp").wait()
        if osmfile  is None:
            while not os.path.exists("tmp/map?bbox="+sub_range):
                Popen("wget http://overpass-api.de/api/map?bbox="+sub_range, shell = True).wait()
                Popen("mv \"map?bbox="+sub_range+"\" tmp/", shell = True).wait()
                if not os.path.exists("tmp/map?bbox="+sub_range):
                    print("Error. Wait for one minitue")
                    sleep(60)   

            filename = "tmp/map?bbox="+sub_range

        else:
            filename = osmfile


        #roadForMotorDict = {'motorway','trunk','primary','secondary','tertiary','residential'}
        roadForMotorDict = {'motorway','trunk'}
        roadForMotorBlackList = {'None', 'pedestrian','footway','bridleway','steps','path','sidewalk','cycleway','proposed','construction','bus_stop','crossing','elevator','emergency_access_point','escape','give_way'}


        mapxml = xml.etree.ElementTree.parse(filename).getroot()

        nodes = mapxml.findall('node')
        ways = mapxml.findall('way')
        relations = mapxml.findall('relation')

        self.nodedict = {}
        self.waydict = {}
        self.roadlist = []
        self.roaddict = {}
        self.edge2edgeid = {}
        self.edgeid2edge = {}
        self.edgeProperty = {}
        self.edgeId = 0
        way_c = 0


        self.minlat = float(mapxml.find('bounds').get('minlat'))
        self.maxlat = float(mapxml.find('bounds').get('maxlat'))    
        self.minlon = float(mapxml.find('bounds').get('minlon'))
        self.maxlon = float(mapxml.find('bounds').get('maxlon'))

        print("[OSM] Parsing nodes ... (%d)" % len(nodes))
        for anode in nodes:
            tmp = {}
            tmp['node'] = anode
            tmp['lat'] = float(anode.get('lat'))
            tmp['lon'] = float(anode.get('lon'))
            tmp['to'] = {}
            tmp['from'] = {}

            self.nodedict.update({anode.get('id'):tmp})


        self.buildings = []
        
        print("[OSM] Parsing ways ... (%d)" % len(ways))
        #print("")
        waycount = 0

        t0 = time() 
        for away in ways:
            waycount += 1
            if waycount % 100 == 0:
                sys.stdout.write("\r[OSM] Parsing ways %d/%d time elasped %.2f seconds" % (waycount, len(ways), time() - t0 ))
                sys.stdout.flush()

            # if waycount % (len(ways) // 20) == 0:
            #     print("[OSM] %d/%d" % (waycount, len(ways)))

            #nds = away.findall('nd')
            highway = 'None'
            lanes = -1
            width = -1
            layer = 0

            #hasLane = False
            #hasWidth = False
            #fromMassGIS = False


            parking = False

            oneway = 0

            isBuilding = False

            building_height = 6

            cycleway = "none"


            info_dict = {}

            for atag in away.findall('tag'):
                info_dict[atag.get('k')] = atag.get('v')

                if atag.get('k').startswith("cycleway"):
                    cycleway = atag.get('v')

                if atag.get('k') == 'building':
                    #if atag.get('v') == "yes":
                        #print("find buildings")
                    isBuilding = True


                if atag.get('k') == 'highway':
                    highway = atag.get('v')
                if atag.get('k') == 'lanes':
                    try:
                        lanes = float(atag.get('v').split(';')[0])
                    except ValueError:
                        lanes = -1 

                    hasLane = True
                if atag.get('k') == 'width':
                    #print(atag.get('v'))
                    try:
                        width = float(atag.get('v').split(';')[0].split()[0])
                    except ValueError:

                        width == -1

                    hasWidth = True
                if atag.get('k') == 'layer':
                    try:
                        layer = int(atag.get('v'))
                    except ValueError:
                        print("ValueError for layer", atag.get('v'))
                        layer = -1
                        
                if atag.get('k') == 'source':
                    if 'massgis' in atag.get('v') :
                        fromMassGIS = True

                if atag.get('k') == 'amenity':
                    if atag.get('v') == 'parking':
                        parking = True

                if atag.get('k') == 'service':
                    if atag.get('v') == 'parking_aisle':
                        parking = True

                if atag.get('k') == 'service':
                    if atag.get('v') == 'driveway':
                        parking = True

                if atag.get('k') == 'oneway':
                    if atag.get('v') == 'yes':
                        oneway = 1
                    if atag.get('v') == '1':
                        oneway = 1
                    if atag.get('v') == '-1':
                        oneway = -1

                if atag.get('k') == 'height':
                    try:
                        building_height = float(atag.get('v').split(' ')[0])
                    except ValueError:
                        print(atag.get('v'))


                if atag.get('k') == 'ele':
                    try:
                        building_height = float(atag.get('v').split(' ')[0]) * 3
                    except ValueError:
                        print(atag.get('v'))

            if width == -1 :
                if lanes == -1 :
                    width = 6.6
                else :
                    if lanes == 1:
                        width = 6.6
                    else:
                        width = 3.7 * lanes

            if lanes != -1:
                if width > lanes * 3.7 * 2:
                    width = width / 2
                if lanes == 1:
                    width = 6.6
                else:
                    width = lanes * 3.7

            if noUnderground:
                if layer < 0 :
                    continue 



            if isBuilding :
                idlink = []
                for anode in away.findall('nd'):
                    refid = anode.get('ref')
                    idlink.append(refid)

                    self.buildings.append([[(self.nodedict[x]['lat'],self.nodedict[x]['lon']) for x in idlink],building_height, info_dict])



            #if highway in roadForMotorDict: #and hasLane and hasWidth and fromMassGIS: 
            #if highway not in roadForMotorBlackList:
            #if highway in roadForMotorDict:

            #if highway not in roadForMotorBlackList and parking == False:
            # if highway not in roadForMotorBlackList and (includeServiceRoad == True or parking == False): # include parking roads!
            if ((not useblacklist) and highway in roadForMotorDict) or ((useblacklist) and highway not in roadForMotorBlackList and parking == False) or allinfo:
                idlink = []
                for anode in away.findall('nd'):
                    refid = anode.get('ref')
                    idlink.append(refid)

                for i in range(len(idlink)-1):
                    link1 = (idlink[i], idlink[i+1])
                    link2 = (idlink[i+1], idlink[i])

                    #if link1 not in self.edge2edgeid.keys():
                    if link1 not in self.edge2edgeid:
                        self.edge2edgeid[link1] = self.edgeId
                        self.edgeid2edge[self.edgeId] = link1
                        self.edgeProperty[self.edgeId] = {"width":width, "lane":lanes, "layer":layer, "roadtype": highway, "cycleway":cycleway, "info":dict(info_dict)}
                        self.edgeId += 1

                    #if link2 not in self.edge2edgeid.keys():
                    if link2 not in self.edge2edgeid:
                        self.edge2edgeid[link2] = self.edgeId
                        self.edgeid2edge[self.edgeId] = link2
                        self.edgeProperty[self.edgeId] = {"width":width, "lane":lanes, "layer":layer, "roadtype": highway, "cycleway":cycleway, "info":dict(info_dict)}
                        self.edgeId += 1


                if oneway >= 0 :
                    for i in range(len(idlink)-1):
                        self.nodedict[idlink[i]]['to'][idlink[i+1]] = 1
                        self.nodedict[idlink[i+1]]['from'][idlink[i]] = 1

                    self.waydict[way_c] = idlink
                    way_c += 1
                    
                idlink.reverse()

                if oneway == -1:
                    for i in range(len(idlink)-1):
                        self.nodedict[idlink[i]]['to'][idlink[i+1]] = 1
                        self.nodedict[idlink[i+1]]['from'][idlink[i]] = 1

                    self.waydict[way_c] = idlink
                    way_c += 1

                if oneway == 0:
                    for i in range(len(idlink)-1):
                        self.nodedict[idlink[i]]['to'][idlink[i+1]] = 1
                        self.nodedict[idlink[i+1]]['from'][idlink[i]] = 1

def getOSMGraph(region, tile_size = 1000, padding = 0):
    OSMMap = OSMLoader(region, False, allinfo=True)
    node_neighbor = {}
    nodeClass = {}
    def getEdgeClass(info):
        # class-1 roads
        if info["roadtype"] in {'motorway','trunk', 'motorway_link', 'trunk_link'}:
            return 1
        
        # class-2 roads
        if info["roadtype"] in {'primary','secondary','tertiary','residential','primary_link','secondary_link','tertiary_link','residential_link'}:
            return 2
        
        # blacklist 
        if info["roadtype"] in {'None','sidewalk','proposed','construction','bus_stop','crossing','elevator','emergency_access_point','escape','give_way'}:
            return -1

        #if info["roadtype"] == "footway":
        for k,v in info["info"].iteritems():
            if "sidewalk" in v or "crossing" in v or "sidewalk" in k or "crossing" in k:
                return -1 

        if info["roadtype"] == "cycleway":
            if "foot" in info["info"] and info["info"]["foot"] == "no":
                return -1
        # class-3 roads
        return 3
    
    for node_id, node_info in OSMMap.nodedict.iteritems():
        lat = node_info["lat"]
        lon = node_info["lon"]

        n1key = (lat,lon)

        neighbors = []
        for nid in node_info["to"].keys() + node_info["from"].keys() :
            link = (node_id, nid)
            eid = OSMMap.edge2edgeid[link]
            edgeProperty = OSMMap.edgeProperty[eid]
            cid = getEdgeClass(edgeProperty)
            if cid < 0:
                continue

            if nid not in neighbors:
                neighbors.append(nid)

            n2key = (OSMMap.nodedict[nid]["lat"],OSMMap.nodedict[nid]["lon"])

            # update nodeClass

            if n1key not in nodeClass:
                nodeClass[n1key] = cid 
            
            if nodeClass[n1key] > cid:
                nodeClass[n1key] = cid

            if n2key not in nodeClass:
                nodeClass[n2key] = cid 
            
            if nodeClass[n2key] > cid:
                nodeClass[n2key] = cid

        for nid in neighbors:
            n2key = (OSMMap.nodedict[nid]["lat"],OSMMap.nodedict[nid]["lon"])

            node_neighbor = graphlib.graphInsert(node_neighbor, n1key, n2key)
    
    node_neighbor, node_class = graphlib.graphDensifyWithClass(node_neighbor, nodeClass)
    node_neighbor_region, node_class_region = graphlib.graph2RegionCoordinateWithClass(node_neighbor, node_class, region, tile_size = tile_size + padding * 2)
    
    lines = []
    vertices = []

    def addbias(loc):
        return (loc[0]-padding, loc[1]-padding)

    def inrange(loc):
        if loc[0] > padding and loc[0] < tile_size + padding and loc[1] > padding and loc[1] < tile_size + padding:
            return True 
        else:
            return False 

    for nid, nei in node_neighbor_region.items():
        for nn in nei:
            if inrange(nn) or inrange(nid):
                edgecolor = max(node_class_region[nid], node_class_region[nn])
                edge = (addbias(nid), addbias(nn), edgecolor)
                edge_ = (addbias(nn), addbias(nid), edgecolor)

                if edge not in lines and edge_ not in lines:
                    lines.append(edge) 

    return lines
            

if __name__ == "__main__":
    pass 