#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 03:51:29 2017

@author: Arpan

Description: Prepare the test dataset. The kth_read_sequences is not considered.
All frames of the action video are converted to LMDB.

"""
import cv2
import numpy as np
import os
import sys
import lmdb
import caffe

# Input: 
# srcVideoFolder: where the test action videos are located
# setIDs: list of person ids in one test set, not required
# Output: Create optical flow visualization data, transformed to HSV space to BGR
# ToDo: write the feature onto a file and convert to lmdb.
def construct_datasets(srcVideoFolder, setIDs):
    videosList = os.listdir(srcVideoFolder)
    print("No of videos = "+str(len(videosList)))      #  for test
    lmdb_name = os.path.join(srcVideoFolder, "LMDB_seq", "OF_lmdb")    
    if not os.path.exists(os.path.dirname(lmdb_name)):
        os.makedirs(os.path.dirname(lmdb_name))
    
    videoCount = 1
    start_id = 0
    N = 400         # Approx number of OF frames per action video. For LMDB memory map size
    X_list = []
    #y_list = []
    # 4 times the size of 200 videos(training has 191 vids) , each vid approx N=400 frames
    map_size = len(videosList)*N*120*160*3*4    # Upper limit ~21 GB for training set
    env = lmdb.open(lmdb_name, map_size=map_size)
    i = start_id
    # loop to extract optical flow from one video per iteration
    for video in videosList:
        print "#####  Video "+str(videoCount)+" :: "+video+"  ##### "
        
        get_optical_flow_vid(os.path.join(srcVideoFolder, video), X_list)
        #print "X_list length = "+str(len(X_list))
        #print "y_list  = "+str(len(y_list))
        #print X_list[1].shape      # (120, 160, 3) for OF visualizations
        #print type(X_list[1])
        # Save X_list to LMDB, incrementally
        print "Saving the frames to LMDB ... starting : " + str(i)
        with env.begin(write=True) as txn:
            while len(X_list)!=0:
                image_mat = X_list[0]    # retrieve the first element
                del X_list[0]    # delete the first element
                vid_label = get_video_label(video)
                
                # change dimension from 120x160x3 to 3x120x160
                image_mat = np.rollaxis(image_mat, 2)
                
                datum = caffe.proto.caffe_pb2.Datum()
                datum.channels = image_mat.shape[0]
                datum.height = image_mat.shape[1]
                datum.width = image_mat.shape[2]
                datum.data = image_mat.tobytes()  # or .tostring() if numpy < 1.9
                datum.label = vid_label
                
                str_id = '{:08}'.format(i)      # saved in a sequence
                # The encode is only essential in Python 3
                txn.put(str_id.encode('ascii'), datum.SerializeToString())
                i = i+1
            
            
        print "Saved to LMDB --> #Frames : "+str(i)
        videoCount = videoCount + 1
        # Uncomment following line whlie testing 
        #break
    

    #env.close()
    
    
# from a srcVideoPath, get the optical flow data of the video, corresponding to 
# frames specified in action_seq_str 
def get_optical_flow_vid(srcVideo, X_list):
    cap = cv2.VideoCapture(srcVideo)
    
    if not cap.isOpened():
        print "Error in reading the video file ! Abort !"
        sys.exit(0)
    ####################################################
#    # extract the start and end frame nos from action_seq_str 
#    # Eg string = '1-75, 120-190, 235-310, 355-435'
#    start_frames = []
#    end_frames = []
#    for marker in action_seq_str.split(','):
#        temp = marker.split('-')        # ' 120-190'
#        start_frames.append(int(temp[0]))   # 120
#        end_frames.append(int(temp[1]))
#    # sanity check condition
#    if len(start_frames)!=len(end_frames):
#        print "Error in reading the frame markers from file ! Abort !"
#        sys.exit(0)
    ####################################################
    dimensions = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    #print "Dimensions (H x W) : "+str(dimensions)
    #print "FPS : "+str(fps)
    #print "RGB Frame shape : (120, 160, 3) :: Gray Frame shape : (120, 160)"
    #print "Start Times : "+str(start_frames)
    #print "End Times   : "+str(end_frames)
    
    #curr_label = get_video_label(srcVideo)
    stime = 0
    #for i, stime in enumerate(start_frames):
    cap.set(cv2.CAP_PROP_POS_FRAMES, stime)
    ret, prev_frame = cap.read()
    # convert frame to GRAYSCALE
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    stime = stime + 1
    # iterate over the frames to find the optical flows
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #print "RGB Frame shape :" + str(frame.shape)+" Gray FS : "+str(curr_frame.shape)
            
        #cv2.calcOpticalFlowFarneback(prev, next, pyr_scale, levels, winsize, 
           #                iterations, poly_n, poly_sigma, flags[, flow])
        # prev(y,x)~next(y+flow(y,x)[1], x+flow(y,x)[0])
        flow = cv2.calcOpticalFlowFarneback(prev_frame,curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        #print "For frames: ("+str(stime-1)+","+str(stime)+") :: shape : "+str(flow.shape)
        #print "Flow stats (mean[mag], min, max) : "+str((np.mean(flow[:,0]), np.min(flow[:,0]), np.max(flow[:,0])))
        #print "Flow stats (mean[orient], min, max) : "+str((np.mean(flow[:,1]), np.min(flow[:,1]), np.max(flow[:,1])))
        #cv2.imshow('Current Frame (Gray)', curr_frame)
        #cv2.imshow('Prev Frame(Gray)', prev_frame)
            
        # Visualization
        vis_vectors = draw_flow(prev_frame, flow, 8)
        vis_bgr = draw_flow_bgr(flow, frame)
        #cv2.imshow('Flow Vis', vis_vectors)
        #cv2.imshow('Flow BGR', vis_bgr)
        X_list.append(vis_bgr)       # append to the list
        #y_list.append(curr_label)
        #cv2.imshow('RGB Frame', frame)
        stime = stime + 1
        prev_frame = curr_frame
        # uncomment following line while testing
        #waitTillEscPressed()    
        
    cv2.destroyAllWindows()
    cap.release()
    return

def get_video_label(srcVid):
    if "boxing" in srcVid:
        return 0
    elif "handclapping" in srcVid:
        return 1
    elif "handwaving" in srcVid:
        return 2
    elif "jogging" in srcVid:
        return 3
    elif "running" in srcVid:
        return 4
    elif "walking" in srcVid:
        return 5

#############################################################
# Not Used 
# X are is the features, matrix of size Nx(dimensions)
# y is the vector of size N
# N is the size of data to be stored in lmdb, N=1000 (maybe)
def convert_to_lmdb(lmdb_name, N, X, y, start_id):
    #image_mat = []
    # create the folder, if not exists
    if not os.path.exists(os.path.dirname(lmdb_name)):
        os.makedirs(os.path.dirname(lmdb_name))
        
    map_size = N*120*160*3*2      # 2 times the size of N images
    env = lmdb.open(lmdb_name, map_size=map_size)
    i = start_id
    with env.begin(write=True) as txn:
        # txn is a Transaction object
        #for i in range(N):
        while i<(start_id+N):
            if len(X)==0:
                print "Data Buffer is empty !"
                break
            image_mat = X[0]    # retrieve the first element
            del X[0]    # delete the first element
            # change dimension from 120x160x3 to 3x120x160
            image_mat = np.rollaxis(image_mat, 2)
            
            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels = image_mat.shape[0]
            datum.height = image_mat.shape[1]
            datum.width = image_mat.shape[2]
            datum.data = image_mat.tobytes()  # or .tostring() if numpy < 1.9
            datum.label = int(y[i])
            
            str_id = '{:08}'.format(i)
            # The encode is only essential in Python 3
            txn.put(str_id.encode('ascii'), datum.SerializeToString())
            i = i+1
            
    # return the ID where it stops
    return i
#############################################################


# draw the OF field on image, with grids, decrease step for finer grid
def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

def draw_flow_bgr(flow, sample_frame):
    hsv = np.zeros_like(sample_frame)
    #print "hsv_shape : "+str(hsv.shape)
    hsv[...,1] = 255
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr
    
def waitTillEscPressed():
    while(True):
        if cv2.waitKey(10)==27:
            print("Esc Pressed")
            return
    
if __name__=="__main__":
    # the dataset folder contains 6 folders boxing, running etc containing videos for each
    # It also contains 00sequences.txt where meta info is given
    dataset = "/home/hadoop/VisionWorkspace/KTH_OpticalFlow/dataset"
    seq_path = os.path.join(dataset, "kth_sequences.txt")
    #import kth_read_sequences
    #seq_dict = kth_read_sequences.read_sequences_file(seq_path)
    
    trainPersons = ["11", "12", "13", "14", "15", "16", "17", "18"]
    valPersons = ["19", "20", "21", "23", "24", "25", "01", "04"]
    testPersons = ["22", "02", "03", "05", "06", "07", "08", "09", "10"]
    
    #print "Step 1: Construct Dataset : " 
    #construct_datasets(os.path.join(dataset, "kth_actions_train"), trainPersons, seq_dict)
    #construct_datasets(os.path.join(dataset, "kth_actions_validation"), valPersons, seq_dict)
    construct_datasets(os.path.join(dataset, "kth_actions_test"), testPersons)
    #print "Dataset in LMDB constructed successfully !!"
    
    ###########################################################
    # Running the caffe model    
    #proc = subprocess.Popen(["/home/hadoop/caffe/build/tools/caffe","train","--solver=optical_flow_lenet_solver.prototxt"],stderr=subprocess.PIPE)
    #res = proc.communicate()[1]

    #caffe.set_mode_gpu()
    #solver = caffe.get_solver("config.prototxt")
    #solver.solve()
    
    #print res
    ###########################################################
    # Applying the model
    
    #net = caffe.Net("demoDeploy.prototxt", "./opt_flow_quick_iter_20000.caffemodel", caffe.TEST)
    #print(get_data_for_id_from_lmdb("/home/lnmiit/caffe/examples/optical_flow/val_opt_flow_lmdb/", "00000209"))
    #l, f = get_data_for_id_from_lmdb("/home/lnmiit/caffe/examples/optical_flow/val_opt_flow_lmdb/", "00000209")

    
    
    
    
    
    