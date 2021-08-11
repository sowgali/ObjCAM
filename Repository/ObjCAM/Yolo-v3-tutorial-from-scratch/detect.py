from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import cv2 
from util import *
import argparse
import os 
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random
from skimage import img_as_ubyte
from skimage.transform import resize
from imageio import imwrite
import matplotlib.pyplot as plt
from PIL import Image

def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
   
    parser.add_argument("--images", dest = 'images', help = 
                        "Image / Directory containing images to perform detection upon",
                        default = "imgs", type = str)
    parser.add_argument("--det", dest = 'det', help = 
                        "Image / Directory to store detections to",
                        default = "det", type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    
    return parser.parse_args()
    
args = arg_parse()
images = args.images
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()



num_classes = 80
classes = load_classes("data/coco.names")



#Set up the neural network
print("Loading network.....")
model = Darknet(args.cfgfile)
derivatives = model.load_weights(args.weightsfile)
print("Network successfully loaded")
#for der in derivatives:
    #print(der.shape)

model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0 
assert inp_dim > 32

#If there's a GPU availible, put the model on GPU
if CUDA:
    model.cuda()


#Set the model in evaluation mode
model.eval()

read_dir = time.time()
#Detection phase
try:
    imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
except NotADirectoryError:
    imlist = []
    imlist.append(osp.join(osp.realpath('.'), images))
except FileNotFoundError:
    print ("No file or directory with the name {}".format(images))
    exit()
    
if not os.path.exists(args.det):
    os.makedirs(args.det)

load_batch = time.time()
loaded_ims = [cv2.imread(x) for x in imlist]

im_batches = list(map(prep_image, loaded_ims, [inp_dim for x in range(len(imlist))]))
im_dim_list1 = [(x.shape[1], x.shape[0]) for x in loaded_ims]
im_dim_list = torch.FloatTensor(im_dim_list1).repeat(1,2)


leftover = 0
if (len(im_dim_list) % batch_size):
    leftover = 1

if batch_size != 1:
    num_batches = len(imlist) // batch_size + leftover            
    im_batches = [torch.cat((im_batches[i*batch_size : min((i +  1)*batch_size,
                        len(im_batches))]))  for i in range(num_batches)]  

write = 0

'''
    for q in derivatives:
        print("Shape of derivatives:")
        print(q.shape)
    for q in myAs:
        print("Shape of As:")
        print(q.shape)
        temp = torch.sum(q,(3,2))
        sumAs.append(temp)
        print("Shape of sum:")
        print(temp.shape)
    for q in mySs:
        print("Shape of Ss:")
        print(q.shape)
    for q in mySigs:
        print("Shape of Sigs:")
        print(q.shape)
    '''
if CUDA:
    im_dim_list = im_dim_list.cuda()
j = 0
start_det_loop = time.time()
heat_ims = []
for i, batch in enumerate(im_batches):
#load the image 
    start = time.time()
    if CUDA:
        batch = batch.cuda()
    prediction, myAs, mySs, mySigs = model(Variable(batch), CUDA)
    for a in myAs:
        print(a.shape)
    for a in derivatives:
        print(a.shape)
    for k,w in enumerate(derivatives):
        a = myAs[k//3][0]
        a_ = a.to('cpu')
        maps = w*a_
        maps_ = maps.sum(axis = 0)
        maps_ = maps_.detach().numpy()
        maps_ /= np.max(maps_)
        cm = resize(maps_, (im_dim_list1[i][1],im_dim_list1[i][0]))
        cm = 1-cm
        cm_heatmap = cv2.applyColorMap(np.uint8(255*cm), cv2.COLORMAP_JET)
        cm_heatmap = cm_heatmap/255.0
        myimg = resize(cv2.cvtColor(loaded_ims[j], cv2.COLOR_BGR2RGB), (im_dim_list1[i][1],im_dim_list1[i][0]))
        fin = (myimg*0.7) + (cm_heatmap*0.3)
        out = "Test/oheat_" + str(k) + '.png'
        imwrite(out, fin)
        print(k)
        '''
    print(len(myAs))
    for k,a in enumerate(myAs):
        w = derivatives[k]
        maps = w*a[0,:,:,:]
        maps_ = maps.sum(axis = 0)
        maps_ = maps_.to('cpu')
        maps_ = maps_.detach().numpy()
        maps_ /= np.max(maps_)
        cm = resize(maps_, (im_dim_list1[i][1],im_dim_list1[i][0]))
        cm = 1-cm
        cm_heatmap = cv2.applyColorMap(np.uint8(255*cm), cv2.COLORMAP_JET)
        cm_heatmap = cm_heatmap/255.0
        myimg = resize(cv2.cvtColor(loaded_ims[j], cv2.COLOR_BGR2RGB), (im_dim_list1[i][1],im_dim_list1[i][0]))
        #myimg = resize(loaded_ims[j], (im_dim_list1[i][1],im_dim_list1[i][0])) 
        fin = (myimg*0.7) + (cm_heatmap*0.3)
        out = "Test/oheatS_" + str(k) + '.png'
        #out = "Test/oheat_1.png"
        imwrite(out, fin)
        #print(3*k+l)
        '''
    x = input()
    sumAs = []
    l = 0
    f,ax = plt.subplots(3,2, figsize=(10,15))
    f1,ax1 = plt.subplots(1, figsize=(10,5))
    for q in mySigs:
        means = torch.mean(q, axis = 2)
        means_ = means.to('cpu')
        means_ = means_.numpy()
        means_ /= np.max(means_)
        cam = resize(means_, (im_dim_list1[i][1],im_dim_list1[i][0]))
        cam = 1-cam
        cam_heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        #plt.axis("off")
        #imgplot = ax[l][0].imshow(cam_heatmap)
        cam_heatmap = cam_heatmap/255.0
        myimg = resize(cv2.cvtColor(loaded_ims[j], cv2.COLOR_BGR2RGB), (im_dim_list1[i][1],im_dim_list1[i][0]))
        
        fin = (myimg*0.7) + (cam_heatmap*0.3)
        #print(type(fin))
        #plt.axis("off")
        #imgplot = ax[l][1].imshow(fin)
        if l == 0:
            #fin.save(out)
            out = "Heat/oheat_" + str(j) + '.png'
            imwrite(out, fin)
            #rescaled = (255.0 / fin.max() * (fin - fin.min())).astype(np.uint8)
            #scipy.misc.toimage(fin).save(out)
            #im_ = Image.fromarray(rescaled)
            #im_.save(out)
            #plt.axis("off")
            #plt.imshow(fin)
            #plt.savefig(out,bbox_inches='tight')
            #plt.close(f1)
            heat_ims.append(fin)
        l += 1
    #plt.savefig("Test_out/heat_" + str(j), dpi=600)
    #plt.close(f)
    '''
    print("Shape of As:")
    print(myAs.shape)
    print("Shape of Sigs:")
    print(mySigs.shape)
    ders = []
    fincam = np.zeros((im_dim_list1[i][1],im_dim_list1[i][0]))
    f,ax = plt.subplots(1,2, figsize=(20,5))
    for k in range(0,3):
        myA = (myAs[3*k+0] + myAs[3*k+1] + myAs[3*k+2])/3
        A = torch.sum(torch.sum(myA))
        for tens in derivatives:
            ders.append(float(torch.sum(tens)))
        mySigs[k] = torch.mean(mySigs[k], 0)
        coeff_1 = 1 - 2 * mySigs[k]
        coeff_2 = 6 * mySigs[k]**2 - 6 * mySigs[k] + 1
        numerator = coeff_1 * (ders[k]**2)
        denominator = 2*numerator + (coeff_2 * A * (ders[k]**3))
        alpha = numerator/denominator

        alpha_ = alpha.to('cpu')
        cam = alpha_.numpy()
        cam = np.maximum(cam, 0)
        cam = cam / np.max(cam)
        cam = resize(cam, (im_dim_list1[i][1],im_dim_list1[i][0]))
        cam = (cam*-1.0) + 1.0
        fincam += cam
    cam_heatmap = cv2.applyColorMap(np.uint8(255*fincam), cv2.COLORMAP_JET)
    cam_heatmap = np.array(cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    imgplot = ax[0].imshow(cam_heatmap)
    cam_heatmap = cam_heatmap/255.0
    myimg = resize(cv2.cvtColor(loaded_ims[j], cv2.COLOR_BGR2RGB), (im_dim_list1[i][1],im_dim_list1[i][0]))
    fin = (myimg*0.7) + (cam_heatmap*0.3)
    plt.axis("off")
    imgplot = ax[1].imshow(fin)
    plt.savefig("Test_out/att_test" + str(j), dpi=600)
    plt.close(f)
    fin = img_as_ubyte(fin)
    heat_ims.append(fin)
    '''
    j += 1
    prediction, p1, p2 = write_results(prediction, derivatives, confidence, num_classes, nms_conf = nms_thesh)

    end = time.time()
    
    if type(prediction) == int:

        for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
            im_id = i*batch_size + im_num
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", ""))
            print("----------------------------------------------------------")
        continue

    prediction[:,0] += i*batch_size    #transform the atribute from index in batch to index in imlist 

    if not write:                      #If we have't initialised output
        output = prediction  
        write = 1
    else:
        output = torch.cat((output,prediction))

    for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
        im_id = i*batch_size + im_num
        objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
        print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
        print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
        print("----------------------------------------------------------")

    if CUDA:
        torch.cuda.synchronize()       
try:
    output
except NameError:
    print ("No detections were made")
    exit()

im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())

scaling_factor = torch.min(416/im_dim_list,1)[0].view(-1,1)


output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2



output[:,1:5] /= scaling_factor

for i in range(output.shape[0]):
    output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
    output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])
    
    
output_recast = time.time()
class_load = time.time()
colors = pkl.load(open("pallete", "rb"))

draw = time.time()


def write(x, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results[int(x[0])]
    cls = int(x[-1])
    color = random.choice(colors)
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
    return img


list(map(lambda x: write(x, loaded_ims), output))
list(map(lambda x: write(x, heat_ims), output))

det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(args.det,x.split("/")[-1]))
det_names2 = pd.Series(imlist).apply(lambda x: "{}/detheat_{}".format(args.det,x.split("/")[-1]))

list(map(cv2.imwrite, det_names, loaded_ims))
list(map(cv2.imwrite, det_names2, heat_ims))

end = time.time()

print("SUMMARY")
print("----------------------------------------------------------")
print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
print()
print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) +  " images)", output_recast - start_det_loop))
print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch)/len(imlist)))
print("----------------------------------------------------------")

torch.cuda.empty_cache()
    
