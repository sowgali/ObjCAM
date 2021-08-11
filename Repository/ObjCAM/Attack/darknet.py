#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import division
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt
from utils import *


# In[2]:


def parse_cfg(cfgfile):
    """
    Takes a configuration file .
    Returns a list of blocks. Each of the blocks describes a block in the neural
    network to be built.
    """
    #Some basic preprocessing of input file
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')
    #Removes empty strings
    lines = [x for x in lines if len(x) > 0]
    #Removes commented input
    lines = [x for x in lines if x[0] is not '#']
    lines = [x.rstrip().lstrip() for x in lines]
    
    block = {}
    blocks = []
    
    for line in lines:
        if line[0] is "[":
            if len(block) is not 0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            key,value = line.split('=')
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)
    
    return blocks


# In[3]:


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors
        

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


# In[4]:


def create_modules(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []
    
    for index,x in enumerate(blocks[1:]):
        module = nn.Sequential()
        bias = False
        if(x["type"] == "convolutional"):
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
            except:
                batch_normalize = 0
                bias = True
            
            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])
            
            if padding is 1:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0
            
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module("conv_{0}".format(index), conv)
            
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)
            
            if activation is "Leaky":
                ac = nn.LeakyReLU(0.1, inplace = True)
                module.add_module("leaky_{0}".format(index), ac)
                
                
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor = 2, mode = "bilinear")
            module.add_module("upsample_{}".format(index), upsample)
            
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')
            start = int(x["layers"][0])
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)
            filters = output_filters[index + start]
            if end < 0:
                filters += output_filters[index + end]
        
        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)
            
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]
            
            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]
            
            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)
        
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
    
    return (net_info, module_list)


# In[5]:


blocks = parse_cfg("yolov3.cfg")


# In[15]:


class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)
        
    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {}
        
        write = 0
        for i,module in enumerate(modules):
            module_type = (module["type"])
            
            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)
            
            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]
                
                if layers[0] > 0:
                    layers[0] -= 1
                if len(layers) == 1:
                    x = outputs[i + layers[0]]
                else:
                    if (layers[1] > 0):
                        layers[1] -= i
                    
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    
                    x = torch.cat((map1, map2), 1)
            
            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i + from_]
            
            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                inp_dim = int(self.net_info["height"])
                num_classes = int(module["classes"])
                
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)
            outputs[i] = x
        return detections
    
    
    def load_weights(self, weightfile):
        fp = open(weightfile, "rb")
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        weights = np.fromfile(fp, dtype = np.float32)

        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i+1]["type"]

            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]
                if(batch_normalize):
                    bn = model[1]
                    num_bn_biases = bn.bias.numel()
                    bn_biases = torch.from_numpy(weights[ptr : ptr + num_bn_biases])
                    ptr += num_bn_biases
                    bn_weights = torch.from_numpy(weights[ptr : ptr + num_bn_biases])
                    ptr += num_bn_biases
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                else:
                    num_biases = conv.bias.numel()
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases
                    conv_biases = conv_biases.view_as(conv.bias.data)
                    conv.bias.data.copy_(conv_biases)
                num_weights = conv.weight.numel()
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)


# In[9]:


def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416,416))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W 
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_


# In[10]:


model = Darknet("yolov3.cfg")
inp = get_test_input()
pred = model(inp, False)#torch.cuda.is_available())
print (pred)


# In[16]:


model = Darknet("yolov3.cfg")
model.load_weights("yolov3.weights")

