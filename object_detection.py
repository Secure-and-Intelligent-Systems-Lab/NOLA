import numpy as np
from natsort import natsorted 
import os

for folder in natsorted(os.listdir("./Test/"))[0:1]:
	# for subf in natsorted(os.listdir("./Test/" + folder + "/")):
	Names = list()
	for files in natsorted(os.listdir("./Test/" + folder  + "/frames/"))[0:9000]:
		Names.append("./Test/" + folder  + "/frames/" + files)
	np.savetxt("./Test/" + folder   + "/Names.txt",Names,fmt='%s')
	
	os.system("/home/keval/Documents/Models/darknet/darknet detector test /home/keval/Documents/Models/darknet/cfg/coco.data /home/keval/Documents/Models/darknet/cfg/yolov4-csp.cfg /home/keval/Documents/Models/darknet/yolov4-csp.weights -thresh 0.25 -ext_output -out {0} -dont_show < {1}".format("./Test/" + folder  +"/" + folder+ ".json","./Test/" + folder  + "/Names.txt"))
	# d+=1