from natsort import natsorted
import os 
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import argparse
from sklearn import metrics

parser = argparse.ArgumentParser(description='Train the model for N stage')
parser.add_argument('stage', type=int)

args = parser.parse_args()

def odit_stat(stat):
    new_stat = list()
    new_stat.append(0)
    for i in range(0,len(stat)):
        new_stat.append(np.max((0,new_stat[i]+stat[i]-7)))
    return new_stat


def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = np.arange(0,dets.shape[0])

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def get_stat(folder):
    print(folder)
    feat_stat = np.zeros((1,9000))
    track_stat = np.zeros((1,9000))
    
    P = np.load(test_path+folder+"/tracks.npy",allow_pickle=True)
    object_file = test_path+folder+"/"+folder+".json"
    with open(object_file) as f:
        objects_detected = json.load(f)
    time = folder.split("_")[1]
    
    rel_idx = np.where(np.isin(P[:,2],objects) ==True)[0]
    tracks = np.unique(P[rel_idx,1])
    
    
    for i in range(0,len(objects_detected),1):
        num_person = 0
        num_veh = 0
        for obj in objects_detected[i]['objects']:
            if obj["name"] == "person" and obj["confidence"] > 0.6:
                num_person+=1
            elif obj["name"] in objects and obj["confidence"] > 0.6:
                num_veh+=1
        feat = [num_veh,num_person,int(time)]
        norm_feat = (feat - min_a_temp)/(max_a_temp-min_a_temp)
        val,ind = (nbrs2.kneighbors(np.array(norm_feat).reshape(-1, 3),n_neighbors=5))
        feat_stat[0,objects_detected[i]['frame_id']-1] = np.max((np.sum(val),feat_stat[0,objects_detected[i]['frame_id']-1]))
    
    
    for j in tracks:
#         print(len(tracks))
        dets = list()
        idx = np.where(P[:,1]==j)
        
        for q in idx[0]:
            dets.append(P[q,3])
        clean_track = nms(np.array(dets),0.7)
        if len(clean_track) > 5:
#             for ct in clean_track:
            feat = [P[q,3][0],P[q,3][1],P[q,3][2],P[q,3][3],objects.index(P[idx[0][0],2])]
            norm_feat = (feat - min_a_spat)/(max_a_spat-min_a_spat)
            val,ind = (nbrs1.kneighbors(np.array(norm_feat).reshape(-1, 5),n_neighbors=5))
            feat_stat[0,P[q,0]-1] = np.max((feat_stat[0,P[q,0]-1],np.sum(val)))
            
            if len(P[idx[0],3]) > 20:
                actual = []
                actual_next = []
                actual_ind = []
                for k in range(0,len(P[idx[0],3])-20):
                    path = np.array([list(x) for x in P[idx[0],3][k:k+20]])
                    fut = np.array([list(x) for x in P[idx[0],3][k+20:k+21]])
                    path[:,0] = path[:,0]/1280
                    path[:,1] = path[:,1]/720
                    path[:,2] = path[:,2]/1280
                    path[:,3] = path[:,3]/720

                    fut[:,0] = fut[:,0]/1280
                    fut[:,1] = fut[:,1]/720
                    fut[:,2] = fut[:,2]/1280
                    fut[:,3] = fut[:,3]/720
                    
                    actual.append(path)
                    actual_next.append(fut)
                    actual_ind.append(P[idx[0],0][k+20]+1)
                    
            
            error = (model.predict(np.array(actual),verbose=0)- np.array(actual_next).reshape(-1,4))**2
            error = np.sqrt(np.sum(error,1))
#             print(error.shape,np.array(actual_next).shape)
            for stat_idx in range(len(error)):
                track_stat[0,actual_ind[stat_idx]-1] = np.max((track_stat[0,actual_ind[stat_idx]-1],error[stat_idx]))
            
    return feat_stat,track_stat


objects = ["car","bike","truck","cart"]


trainpath = "./NOLA/Train/"


spat_feat = list()
track_feat_train = list()
temp_feat = []

for cl_stage in os.listdir(trainpath)[0:args.stage+1]:
	path = trainpath + cl_stage + "/"
	for folder in os.listdir(path):

	    P = np.load(path+folder+"/tracks.npy",allow_pickle=True)
	    object_file = path+folder+"/"+folder+".json"
	    with open(object_file) as f:
	        objects_detected = json.load(f)
	    time = folder.split("_")[1]
	    
	    tracks = np.unique(P[:,1])
	    rel_idx = np.where(np.isin(P[:,2],objects) ==True)[0]
	    tracks = np.unique(P[rel_idx,1])



	    for i in range(0,len(objects_detected),30):
	        num_person = 0
	        num_veh = 0
	        for obj in objects_detected[i]['objects']:
	            if obj["name"] == "person" and obj["confidence"] > 0.6:
	                num_person+=1
	            elif obj["name"] in objects and obj["confidence"] > 0.6:
	                num_veh+=1
	        temp_feat.append([num_veh,num_person,int(time)])        

	    for j in tracks:
	        dets = list()
	        idx = np.where(P[:,1]==j)
	        for q in idx[0]:
	            dets.append(P[q,3])
	        clean_track = nms(np.array(dets),0.7)
	        if len(clean_track) > 10:
	            for ct in clean_track:
	                spat_feat.append([dets[ct][0],dets[ct][1],dets[ct][2],dets[ct][3],objects.index(P[idx[0][0],2])])
	            track_feat_train.append(P[idx[0],3])

X_train = []
X_test = []
for track in track_feat_train:
    for ind in range(0,len(track)-21,5):
        X_train.append(np.array([list(x) for x in track[ind:ind+20]]))
        X_test.append(np.array([list(x) for x in track[ind+20:ind+21]]))

X_train = np.array(X_train)
X_train[:,:,0] = X_train[:,:,0]/1280
X_train[:,:,1] = X_train[:,:,1]/720
X_train[:,:,2] = X_train[:,:,2]/1280
X_train[:,:,3] = X_train[:,:,3]/720

X_test = np.array(X_test).reshape(-1,4)
X_test[:,0] = X_test[:,0]/1280
X_test[:,1] = X_test[:,1]/720
X_test[:,2] = X_test[:,2]/1280
X_test[:,3] = X_test[:,3]/720

model = Sequential()
model.add(LSTM(20, input_shape=(20,4),return_sequences=True))
model.add(LSTM(20,return_sequences=True))
model.add(LSTM(20,return_sequences=False))
model.add(Dense(4))
model.compile(loss='mse', optimizer='adam')
model.fit(X_train,X_test, epochs=10, batch_size=72)

spat_feat = np.array(spat_feat)
temp_feat = np.array(temp_feat)

min_a_spat= np.min(spat_feat[:,0:5],0)
max_a_spat = np.max(spat_feat[:,0:5],0)

min_a_temp= np.min(temp_feat[:,0:3],0)
max_a_temp = np.max(temp_feat[:,0:3],0)

for i in range(0,5):
    spat_feat[:,i] = (spat_feat[:,i] - np.min(spat_feat[:,i]))/(np.max(spat_feat[:,i])-np.min(spat_feat[:,i]))
    

for i in range(0,3):
    temp_feat[:,i] = (temp_feat[:,i] - np.min(temp_feat[:,i]))/(np.max(temp_feat[:,i])-np.min(temp_feat[:,i]))


from sklearn.neighbors import NearestNeighbors
nbrs1 = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(spat_feat)
nbrs2 = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(temp_feat)

add_f = list()
precision_f = list()
gt = np.loadtxt("gt.txt",dtype=str,delimiter=',')

path = "./Test"
test_path = "./Test/"
all_stat = list()
all_folders = list()

for folder in os.listdir(test_path):
    stat = get_stat(folder)
    fstat = stat[0][0,:] + stat[1][0,:]
    all_stat.append(fstat)
    all_folders.append(folder)



for thr in range(0,200,30):
    add = list()
    precision = list()
    tp = list()
    fp = list()
    thr = thr/100
    for idxx in range(len(all_stat)):
        folder = all_folders[idxx]
        stat = odit_stat(all_stat[idxx])
        #plt.plot(stat)
        #plt.show()
        
        if folder in gt[:,0]:
            gr = gt[np.where(gt[:,0] == folder)[0][0],:]
            gt_fold = np.zeros((1,9000))
            gt_fold[int(gr[1]):int(gr[2])] = 1
            if gr[3] != "-1":
                gt_fold[int(gr[3]):int(gr[4])] = 1
            if gr[5] != "-1":
                gt_fold[int(gr[5]):int(gr[6])] = 1
                
        else:
            gt_fold = np.zeros((1,9000))


        alarm = np.where(np.array(stat) > thr)[0]
        if len(alarm) > 0:
            if alarm[0] < int(gr[1]) :
                add.append(0)
                
#                 fp.append(len(np.where(alarm<int(gr[1]))[0]))
                fp.append(1)
                
#                 fp.append(1+len(np.where(np.where(np.diff(alarm) > 2)[0] < int(gr[1]))[0]))
                if True in (alarm > int(gr[1])) & (alarm < int(gr[2])):

#                 if np.max(alarm) > int(gr[1]) and np.min(alarm) < int(gr[1]):
                    tp.append(1)
                else:
                    tp.append(0)
                
      
            else:
                print(folder,alarm[0],int(gr[1]),thr)
                if True in (alarm > int(gr[1])) & (alarm < int(gr[2])):
                    if alarm[0]-int(gr[1]) < 1000:
                        tp.append(1)
                        add.append(alarm[0]-int(gr[1]))
                    else:
                        tp.append(0)
                        add.append(1000)
                else:
                    add.append(1000)
                fp.append(0)
                
                
        else:
            tp.append(0)
            fp.append(0)
            add.append(9000)
            
    if np.mean(tp) + np.mean(fp) != 0:
        add_f.append(np.mean(add))
        precision_f.append(np.sum(tp)/(np.sum(tp)+np.sum(fp)))

print("APD:",metrics.auc(np.array(add_f)/9000, precision_f))