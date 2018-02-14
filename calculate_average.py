import cv2
import os
import numpy as np
image_dir = os.getcwd()
img_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]
imgs = []
ultimate_blue = 0;
ultimate_red = 0;
ultimate_green = 0;
for f in img_files:
    imgs.append(cv2.imread(f))
for i,image in enumerate(imgs):
	length = len(image)*len(image[0])
	sum_blue = 0;
	sum_green = 0;
	sum_red = 0;
	for rows in image:
		for columns in rows:
#			print(columns)
		#	print(columns.shape)
			sum_blue += columns[0]
			sum_green += columns[1]
			sum_red += columns[2]
		#print(j)
	#print("HELLO")
	#print(image[0].shape)
	ultimate_blue += sum_blue/length
	ultimate_green += sum_green/length
	ultimate_red += sum_red/length
	#print(imgs[0][0].size)
	#print(imgs[0][1].size)
	#print(imgs[0][2].size)
	print(sum_blue/length)
	print("Done with image #" + str(i))
#print(ultimate_blue)
#print(len(imgs))
print(ultimate_blue/len(imgs))
print(ultimate_green/len(imgs))
print(ultimate_red/len(imgs))

