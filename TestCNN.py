#!/usr/bin/env python

#################################################################################################
# Import libraries
# These are libraries that you will want to use for this homework
# cv2 is a libary that is useful for image manipulation
# os is a libary that does system type things (like walking directories)
# xml... is a library that lets you parse XML files easily (the annotations are stored in xml files)
# the next three are necessary to import the edge_boxes library that we'll be using
# random helps you sample things in a random fashion
# caffe lets you use caffe
#################################################################################################
import cv2
import os
from os import walk
import sys
sys.path.append('/home/min/a/ee570/edge-boxes-with-python/')
import edge_boxes
import random
import caffe
import numpy

#################################################################################################
# This is how you load an image and then grab a "patch" of that image
#################################################################################################
model = "/home/min/a/chaod/ee570/hw3/hw3-deploy.prototxt"
weights6 = "/home/min/a/chaod/ee570/hw3/hw3-weights.caffemodel"   
# specify caffe mode of operation
caffe.set_mode_cpu()
# create net object
net = caffe.Net(model, weights6, caffe.TEST)



datasetDir   = '/home/min/a/ee570/hw3-files/hw3-dataset'
#classid = 'n02119789'  # fox
#classid = 'n03792782'  # bike
#classid = 'n02504458'  # ele
classid = 'n04037443'  # car
classid_dict = {'n02119789':"fox", 'n03792782':"bike", 'n02504458':"elephant", 'n04037443':"car"}
#image_path = 
# load an image
imageDir = datasetDir + "/" + classid
imageFiles = []
imageFileNames = []

for (dirpath, dirnames, filenames) in walk(imageDir):
	imageFiles.extend(filenames)
	break
for idx, name in enumerate(imageFiles):
        imname = imageDir + "/" + name
        imageFileNames.append(imname)
print("Making windows..........")
windows = edge_boxes.get_windows(imageFileNames)
class_name = {0:"fox",1:"bike",2:"elephant",3:"car"}
total_round = 0
correct = 0
for i in range(len(imageFileNames)-423):
    total_round = total_round + 1
    score_his = []
    class_his = []
    window_need = []
    orig_image = cv2.imread(imageFileNames[i])
    round_num = 0
    boxnum = 0
    while round_num < len(windows[i]):
        #boxnum = boxnum + 1
        round_num = round_num + 1
        crop_img = orig_image[int(windows[i][boxnum][0]):int(windows[i][boxnum][2]) , int(windows[i][boxnum][1]):int(windows[i][boxnum][3])]
        image = crop_img
        inputResImg = cv2.resize(image, (32, 32) , interpolation=cv2.INTER_CUBIC)
        transposedInputImg = inputResImg.transpose((2,0,1))
        net.blobs['data'].data[...]=transposedInputImg
        out = net.forward()
        scores = out['prob']
        j = numpy.argmax(scores[0])  # predicted class
        #### we only save the class other than background
        if j < 4:
            window_need.append(windows[i][boxnum])
            score = max(scores)   # score
            score_his.append(score)
            class_his.append(j)
        boxnum = boxnum + 1
    highest_score_idx = numpy.argmax(score_his)
    x1 = int(window_need[highest_score_idx][1])
    y1 = int(window_need[highest_score_idx][0])
    x2 = int(window_need[highest_score_idx][3])
    y2 = int(window_need[highest_score_idx][2])
    print("predict class: ", class_name[class_his[highest_score_idx]])
    print("classid: ", classid_dict[classid])
    print(class_name[class_his[highest_score_idx]] == classid_dict[classid])
    if (str(class_name[class_his[highest_score_idx]]) == str(classid_dict[classid])):
        correct = correct + 1
        print("correct= ", correct)
        print("YA")
	
	# draw bounding boxes on the image
    img = cv2.rectangle(orig_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv2.imshow(str(class_name[class_his[highest_score_idx]]), img)
    print "Press the space bar to move on"
    cv2.waitKey(0)
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("correct= ", correct)
print("Accuracy : ", float(correct)/total_round)




###########        
#image = cv2.imread('/home/min/a/ee570/hw3-files/hw3-dataset/n02119789/n02119789_14086.JPEG')
############
# get the patch
#x1 = 50
#y1 = 50
#x2 = 100
#y2 = 150
#proposalImage = image[y1:y2, x1:x2]
#cv2.imshow('origImage', image)
#cv2.imshow('patch', proposalImage)
#print "Press the space bar to exit"
#cv2.waitKey(0)

#################################################################################################
# Everything else you need to do in this file has been provided in previous guides.
#
# All the instructions on how to load caffe, resize an image, transpose the image to match how
# caffe likes the input, put the data in caffe's "data blob", perform a forward pass, extract 
# the predictions, etc. were done in hw2.
#
# All the instructions on how to use edge-boxes is found in hw3-guide-phase1.py
# 
#################################################################################################


