import cv2
import numpy as np
 
# importing library for plotting
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from skimage import data
from skimage.io import imread,imshow
from skimage.filters import threshold_otsu,threshold_mean
from PIL import Image
from skimage.color import rgb2gray

#img=img[100:400,100:400]
from skimage.transform import resize
img = imread(r'/d1.png')
plt.imshow(img)
#img.show()
# find frequency of pixels in range 0-255
img_new=img[:,:,0:3]
plt.imshow(img_new)
histr = cv2.calcHist([img_new],[0],None,[256],[0,256])
plt.plot(histr)
plt.show()
gray_img=rgb2gray(img_new)
plt.imshow(gray_img,cmap='gray')
thresh1=threshold_otsu(gray_img)
binary=gray_img>thresh1

ret, thresh1 = cv2.threshold(img, 90, 255, cv2.THRESH_BINARY)
plt.imshow(binary,cmap='gray')

masked = img_new.copy()
masked[binary==1]=255
#crop1=masked[100:400,0:400,:]
# rows and coloumn and all channels
#plt.imshow(crop1)
plt.imshow(masked)
