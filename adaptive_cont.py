import numpy as np
import cv2
from multiprocessing import Process
import argparse

def adap_contr(img1):
    img = cv2.imread(img1,0)
    #img2 =cv2.imread('pylon5/mc3bggp/aymen/Penguin_colonies_2000Pix/BEAU/Satellite/WV02_20111225130502_103001001071A100_11DEC25130502-M1BS-052888272030_01_P001_u08rf3031.tif',0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(img)
    cv2.imwrite('adaptive_contrast2.tif',cl1)
processes = []

for m in range(1,1):
   n = m + 1
   p = Process(target=adap_contr, args=(m, n))
   p.start()
   processes.append(p)

for p in processes:
    p.join()



if __name__== "__main__":
     parser = argparse.ArgumentParser()
     parser.add_argument('img', help='IMAGE')
     args = parser.parse_args()
     adap_contr(args.img)

#img = cv2.imread('/pylon5/mc3bggp/aymen/Penguin_colonies_2000Pix/BEAU/Aerial/CA256832V0230_Panel1.tif',0)
#img2 =cv2.imread('pylon5/mc3bggp/aymen/Penguin_colonies_2000Pix/BEAU/Satellite/WV02_20111225130502_103001001071A100_11DEC25130502-M1BS-052888272030_01_P001_u08rf3031.tif',0)  
# create a CLAHE object (Arguments are optional).
#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#cl1 = clahe.apply(img)
#cl2 = clahe.apply(img2)
#cv2.imwrite('adaptive_contrast1.tif',cl1)
#cv2.imwrite('adaptive_contrast2.tif',cl2)
