# Imports

import os as os        # for iteration throug directories
import pandas as pd # for Series and DataFrames
import cv2          # for OpenCV 
import datetime     # for TimeStamp in CSVFile
from scipy.cluster.vq import * # for Clustering http://docs.scipy.org/doc/scipy/reference/cluster.vq.html
import numpy as np  # for arrays
import time       # for time calculations
from feature_extraction_try import imgCrawl, getClassLabels
from skimage.feature import hog
from sklearn.cluster import MiniBatchKMeans

#in : npImages, color

# In order to calculate HOG, we will use a bag of word approach : cf SURF function, well documented. 

def imageSequencing(npImages, CELL_DIMENSION):

  blocks=[]
  for k in range(len(npImages)):
    image = cv2.imread(npImages[k][1])
    resizedImage = reSize(image, CELL_DIMENSION)
    height, width, channels = resizedImage.shape
    blocks.append(\
      np.array([\
        resizedImage[\
          j*CELL_DIMENSION:j*CELL_DIMENSION+CELL_DIMENSION,\
          i*CELL_DIMENSION:i*CELL_DIMENSION+CELL_DIMENSION] \
        for i in range(width/CELL_DIMENSION) \
        for j in range(height/CELL_DIMENSION)\
      ])\
    )
  return np.array(blocks)  

def reSize(image, CELL_DIMENSION):
  height, width, channels = image.shape
  if height%CELL_DIMENSION==0 and width%CELL_DIMENSION==0:
    resizedImage = image
  elif width%CELL_DIMENSION==0:
    missingPixels = CELL_DIMENSION-height%CELL_DIMENSION
    resizedImage = cv2.copyMakeBorder(image,0,missingPixels,0,0,cv2.BORDER_REPLICATE)
  elif height%CELL_DIMENSION==0:
    missingPixels = CELL_DIMENSION-width%CELL_DIMENSION
    resizedImage = cv2.copyMakeBorder(image,0,0,0,missingPixels,cv2.BORDER_REPLICATE)
  else:
    missingWidthPixels = CELL_DIMENSION-width%CELL_DIMENSION
    missingHeightPixels = CELL_DIMENSION-height%CELL_DIMENSION
    resizedImage = cv2.copyMakeBorder(image,0,missingHeightPixels,0,missingWidthPixels,cv2.BORDER_REPLICATE)
  return resizedImage

def hogAllBlocks(blocks):
  gradients = np.array([np.array([hog(cv2.cvtColor(block, cv2.COLOR_BGR2GRAY), orientations=8, pixels_per_cell=(5,5), cells_per_block=(1,1)) for block in image]) for image in blocks])
  return gradients

def clusterGradients(gradients, NB_CLUSTERS, MAXITER):
  sizes = np.array([len(gradient) for gradient in gradients])
  nbImages =  len(gradients)
  flattenedHogs = np.array([block for image in gradients for block in image])
  miniBatchKMeans = MiniBatchKMeans(n_clusters=NB_CLUSTERS, max_iter=MAXITER, compute_labels=True)
  hogsLabels = miniBatchKMeans.fit_predict(flattenedHogs)
  return hogsLabels, sizes

def makeHistograms(labels, NB_CLUSTERS, sizes):
  indiceInLabels = 0
  hogs = []
  for image in sizes:
    histogram = np.zeros(NB_CLUSTERS)
    for i in range(image):
      histogram[labels[indiceInLabels+i]] += 1
    hogs.append(histogram)
    indiceInLabels+=i 
  return np.array(hogs)

# Main for testing
if __name__ == '__main__':


  start = time.time()
  path ='../../03-jeux-de-donnees/101_ObjectCategories'
  testNpImages = [ [1,'testImage.jpg'] ]
  NB_CLUSTERS = 12
  MAXITER = 100
  print testNpImages[0][1]
  print "Fetching Images in " + path

  # get dictionary to link classLabels Text to Integers
  # sClassLabels = getClassLabels(path)

  # Get all path from all images inclusive classLabel as Integer
  # dfImages = imgCrawl(path, sClassLabels)
  # npImages = dfImages.values
  extractedTime = time.time()
  print "Extracted images in " + str(extractedTime-start) +'sec'
  print "Sequencing Images ..."
  blocks = imageSequencing(testNpImages, 5)
  sequencedTime = time.time()
  print "Sequenced images in " + str(sequencedTime-extractedTime) +'sec'
  print "Computing gradient on each block ..."
  gradients = hogAllBlocks(blocks)
  hogedTime = time.time()
  print "Computed gradients in " + str(hogedTime - sequencedTime) + 'sec'
  print "Clustering gradients ..."
  gradientLabels, sizes = clusterGradients(gradients, NB_CLUSTERS, MAXITER)
  clusteredItme = time.time()
  print "Clustered gradients in " + str(hogedTime - sequencedTime) + 'sec'
  print "Computing histograms ..."
  histograms = makeHistograms(gradientLabels, NB_CLUSTERS, sizes)
  end = time.time()
  print "Computed histograms in " + str(int(end - hogedTime)) + 'sec'
  print "Total time : " + str(end-start) + 'sec'
  print "Histogram shape : " +str(histograms.shape)
