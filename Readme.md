# Sat2Graph simplified infer

This is not a direct fork of the [Sat2Graph github repo](https://github.com/songtaohe/Sat2Graph), but practically it is a fork.
This code repo iterates upon the docker image (songtaohe/sat2graph_inference_server_cpu) published by the repo owner (songtaohe).


## Why change the docker image content rather than the sat2graph github repo

Note: The Sat2Graph repo provides the logic, but due to the following points I prefer to iterate upon the Docker image content:
1. Availability of pretrained models.
2. The pretrained models do not work with the Github code base.
3. Having an working starting point (this includes dependencies and Python 2 environment).


## What changes are made

The following changes are made (or are planned to make):
* [x] Logic runs with Python 3 and up-to-date dependencies.
* [x] Run the inference code without the server logic.
* [x] Act on a pretrained (by the original author songtaohe) globalv2 model.
* [x] Infer images with arbitrary widht and height.
* [ ] Decoupling GTE from satellite image inference with transforming GTE into a graph. 


## How to run the code
[Hatch](https://hatch.pypa.io/latest/) is used for package management.
To run this code, the following should work: 
``` bash
git clone https://github.com/jvtubergen/sat2graph-simplified
cd sat2graph-simplified # Move to root of project folder.
hatch shell # Install package dependencies.
# Place image to infer against at "./sample.png"
python main.py # Infer graph from sample.png
```
