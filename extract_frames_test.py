import numpy as np
from natsort import natsorted 
import os

for folder in natsorted(os.listdir("Test/")):

	if not os.path.isdir("Test/" + folder + "/"  + "frames"):
		os.mkdir("Test/" + folder + "/"  + "frames")
	# d = 0
	os.system("ffmpeg -hwaccel cuda -i {0} -framerate 30 -r 30 -f image2 {1}%d.png".format(str("Test/" + folder  + "/video.mp4"),str("Test/" + folder  + "/" + "frames/")))
	# d+=1