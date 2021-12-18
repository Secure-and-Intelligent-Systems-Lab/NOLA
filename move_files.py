import numpy as np
from natsort import natsorted as ns
import os
import shutil

for folder in ns(os.listdir("All/Train/")):
	if not os.path.isdir("Frames/Train_frames/"+folder):
		os.mkdir("Frames/Train_frames/"+folder)
	for folder2 in ns(os.listdir("All/Train/"+folder)):
		shutil.move("All/Train/"+folder+"/"+folder2+"/frames", "Frames/Train_frames/"+ folder+"/"+folder2)


for folder in ns(os.listdir("Test/")):
	os.mkdir("Frames/Test_frames/"+folder)
	shutil.move("Test/"+folder+"/frames", "Frames/Test_frames/"+ folder)