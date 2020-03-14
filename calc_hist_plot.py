import cv2 
from matplotlib import pyplot as plt 


# load an image in grayscale mode 
img = cv2.imread('ex.jpg',0) 
img2 = cv2.imread('ex.jpg',0)  
# calculate frequency of pixels in range 0-255 
histr = cv2.calcHist([img],[0],None,[256],[0,256])  
histr2 = cv2.calcHist([img2],[0],None,[256],[0,256])
# show the plotting graph of an image 
plt.plot(histr) 
plt.plot(histr2)
plt.show() 

