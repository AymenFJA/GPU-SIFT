import os 
import sys
import subprocess

def match():

	cmd = '/home/aymen/SummerRadical/SIFT-GPU/cudasift'
	img1 = '/pylon5/mc3bggp/aymen/Penguin_colonies_gpu/2000/RGBREF_x-2100000y+0750000_2000Pix.tif'
	img2 = '/pylon5/mc3bggp/aymen/Penguin_colonies_gpu/2000/RGBREF_x-2100000y+0750000_2000Pix.tif'
	x1 = '2000'
	y1 = '2000'
	x2 = '2000'
	y2 = '2000'
	p = subprocess.check_call([cmd, img1, '0', '0', x1, y1, img2, '0', '0', x2, y2])
	
	print (p)

match()
