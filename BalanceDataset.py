#!/usr/bin/env python

#################################################################################################
# Import libraries
# These are libraries that you will want to use for this homework
# os is a libary that does system type things (like walking directories)
# random helps you sample things in a random fashion
#################################################################################################
import os
from os import walk
import random

###############################################################################################
classid_all = []
#imageFiles = []
f = file("outputFileName.txt", "w")
datasetDir   = '/home/min/a/chaod/ee570/hw3/images/'
classid0 = 'n02119789'  # fox
classid1 = 'n03792782'  # bike
classid2 = 'n02504458'  # ele
classid3 = 'n04037443'  # car
classid_all.append(classid0)
classid_all.append(classid1)
classid_all.append(classid2)
classid_all.append(classid3)
pic_num = 0
for idx, filename in enumerate(classid_all):
    imageFiles = []
    imageDir = datasetDir + filename + "/pos"
    for (dirpath, dirnames, filenames) in walk(imageDir):
	imageFiles.extend(filenames)
	break 
    for idxx, name in enumerate(imageFiles):
        imname = imageDir + "/" + name
        f.write(imname + " " + str(idx) + "\n")
        pic_num = pic_num + 1
        #imageFileNames.append(imname)
#f.close()  
neg_num = int(pic_num/4)
neg_file = [] 
for idx, filename in enumerate(classid_all):
    imageFiles = []
    imageDir = datasetDir + filename + "/neg"
    for (dirpath, dirnames, filenames) in walk(imageDir):
	imageFiles.extend(filenames)
	break 
    for idxx, name in enumerate(imageFiles):
        imname = imageDir + "/" + name
        neg_file.append(imname)

    #for idxx, name in enumerate(imageFiles):
        #imname = imageDir + "/" + name
        #f.write(imname + " " + str(4) + "\n")
        #imageFileNames.append(imname)
reducedRandomList = random.sample(neg_file, 326)  # can use neg_num here
for t, name in enumerate(reducedRandomList):
    f.write(name + " " + str(4) + "\n")
f.close()   




#################################################################################################
# This is how you would open a file, write to the file, and close the file
#################################################################################################
#f = file("outputFileName.txt", "w")
#pathToImage = "<absolutePathToAnImage>"
#label = 5
#f.write(pathToImage + " " + str(label) + "\n")
#f.close()

#################################################################################################
# This is how you would shuffle a list and keep N samples
#################################################################################################
#myList = []
#myList.append("test1")
#myList.append("test2")
#myList.append("test3")
#myList.append("test4")
#myList.append("test5")
#myList.append("test6")
#numSamplesKeep = 3
#reducedRandomList = random.sample(myList, numSamplesKeep)

#print myList
#print reducedRandomList



