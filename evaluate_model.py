#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 03:40:47 2017

@author: hadoop

Description: Evaluation of model on test set
"""

import cv2
import caffe
import os
import matplotlib.pyplot as plt
import numpy as np
import lmdb

# dimension of mat changed from 3x120x160 to 120x160x3 by rolling twice the last dim
def convert_to_bgr(mat):
    mat = np.rollaxis(np.rollaxis(mat,2), 2)    # roll axis
    return mat

# http://research.beenfrog.com/code/2015/03/28/read-leveldb-lmdb-for-caffe-with-python.html
def get_data_for_id_from_lmdb(lmdb_name, id):
    lmdb_env = lmdb.open(lmdb_name, readonly=True)
    lmdb_txn = lmdb_env.begin()
    
    lmdb_cursor = lmdb_txn.cursor()
    raw_datum = lmdb_txn.get(id)
    datum = caffe.proto.caffe_pb2.Datum()
    datum.ParseFromString(raw_datum)
    label = datum.label
    flat_x = np.fromstring(datum.data, dtype=np.uint8)
    x = flat_x.reshape(datum.channels, datum.height, datum.width)
    #img = convert_to_bgr(x)
    #cv2.imshow("BGR_OF", img)
    #print "Label = "+str(label)
     
    return(label, x)

# Input: 
# net: caffe.Classifier() object  --> Initialized Test network
# srcVideoFolder: where the action videos are located of corresponding set
# setIDs: list of person ids in one of train, val, or test set
def evaluate_on_test_data(net, srcVideoFolder, bgThresh):
    videosList = os.listdir(srcVideoFolder)
    print("No of videos = "+str(len(videosList)))      # 191 for training
    #lmdb_name = os.path.join(srcVideoFolder, "LMDB_temp", "OF_lmdb")    
    #if not os.path.exists(os.path.dirname(lmdb_name)):
    #    os.makedirs(os.path.dirname(lmdb_name))
    
    videoCount = 0
    # create a probalities matrix (no_of_vid x no_of_classes) 
    prediction_mat = np.zeros((len(videosList), 6))      # 6 classes for actions, keep vstacking for videos
    actual_labels = np.zeros((len(videosList),))
    # loop to extract optical flow from one video per iteration, and get final prediction
    for video in videosList:
        print "#####  Video "+str(videoCount+1)+" :: "+video+"  ##### "+str(bgThresh)
        predict_on_optical_flow(net, os.path.join(srcVideoFolder, video),  prediction_mat, videoCount, bgThresh)
        actual_labels[videoCount] = get_video_label(video)
        print "Final Freq of Predictions :: "
        print prediction_mat[videoCount,:]
        videoCount = videoCount + 1
        # Uncomment following line whlie testing 
        #if videoCount==5:
        #    break
    
    # Compute final score
    
    return (prediction_mat, actual_labels)
    
# srcVideo: path of a test video
# net: caffe.Classifier object
# pred_mat: no_of_test_videos X no_of_classes matrix to store the freq of predictions
# vid_no: i_th video to access the row of pred_mat
def predict_on_optical_flow(net, srcVideo, pred_mat, vid_no, bgThresh):
    cap = cv2.VideoCapture(srcVideo)
    
    if not cap.isOpened():
        import sys
        print "Error in reading the video file ! Abort !"
        sys.exit(0)

    dimensions = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    #vid_label = get_video_label(srcVideo)
    fgbg = cv2.createBackgroundSubtractorMOG2()     #bg subtractor
    ret, prev_frame = cap.read()
    fgmask = fgbg.apply(prev_frame)
    # convert frame to GRAYSCALE
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    # iterate over the frames to find the optical flows
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        
        curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # To find the background mask and skip the frame if foreground is absent
        fgmask = fgbg.apply(frame)
        if np.sum(fgmask)<bgThresh:
            #print "BG frame skipped !!"
            prev_frame = curr_frame
            continue
        
        
        #cv2.calcOpticalFlowFarneback(prev, next, pyr_scale, levels, winsize, 
        #                iterations, poly_n, poly_sigma, flags[, flow])
        # prev(y,x)~next(y+flow(y,x)[1], x+flow(y,x)[0])
        flow = cv2.calcOpticalFlowFarneback(prev_frame,curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        #print "For frames: ("+str(stime-1)+","+str(stime)+") :: shape : "+str(flow.shape)
        #print "Flow stats (mean[mag], min, max) : "+str((np.mean(flow[:,0]), np.min(flow[:,0]), np.max(flow[:,0])))
        #print "Flow stats (mean[orient], min, max) : "+str((np.mean(flow[:,1]), np.min(flow[:,1]), np.max(flow[:,1])))
        #cv2.imshow('Current Frame', curr_frame)
        #cv2.imshow('Prev Frame', prev_frame)
        
        # Visualization
        vis_vectors = draw_flow(prev_frame, flow, 8)
        vis_bgr = draw_flow_bgr(flow, frame)
        #cv2.imshow('Flow Vis', vis_vectors)
        #cv2.imshow('Vectors ', vis_vectors)
        #cv2.imshow('Flow BGR', vis_bgr)
        
        # process image for providing input to classifier.
        # convert image to float32, scale, BGR to RGB (as in skimage)
        # https://github.com/BVLC/caffe/issues/2598
        vis_bgr = vis_bgr.astype(np.float32, copy=False)
        vis_bgr = vis_bgr / 255.
        vis_bgr = vis_bgr[:,:,(2,1,0)]  
        
        # Voting mechanism:
        # predict on the optical flow and add 1 to the corresponding class prediction of
        # that video, no need to store the probs of each predictions. Highest count wins
        frame_pred_probs = net.predict([vis_bgr])
        class_pred = frame_pred_probs[0].argmax()       #index for highest prob
        pred_mat[vid_no,class_pred] = pred_mat[vid_no,class_pred] + 1
        #print "--> Predicted class :: "+str(class_pred)
        
        #cv2.imshow('RGB Frame', frame)
        prev_frame = curr_frame
        # uncomment following block of code to write visualizations to files
        #keyPressed = waitTillEscPressed()    
        #if keyPressed==1:
        #    break
            
    cv2.destroyAllWindows()
    cap.release()
    return

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

# Convert a flow matrix to the HSV and then RGB image space
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
            return 0
        if cv2.waitKey(10)==121:
            print("'y' pressed . Quit !!")
            return 1

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

if __name__=='__main__':
    caffe.set_mode_gpu()
    # load the model
    #MODEL = "deploy_OF_lenet.prototxt"
    MODEL = "deploy_OF_alexnet_mirror.prototxt"
    #MODEL = "deploy_OF_alexnet.prototxt"
    #PRETRAINED = "opt_flow_lenet_snap_iter_20000.caffemodel"
    PRETRAINED = "OF_alexnet_mirror_snap_iter_40000.caffemodel"
    #PRETRAINED = "OF_alexnet_snap_iter_40000.caffemodel"
    
    bgThresholds = [105000, 115000]
    net = caffe.Classifier(MODEL, PRETRAINED, channel_swap=(2,1,0) , raw_scale=255, image_dims=(120, 160, 3))
    
    # Predict on the validation set videos for each value of bgThreshold
    for th in bgThresholds:
        prediction_labels, actual_labels = evaluate_on_test_data(net, "/home/hadoop/VisionWorkspace/KTH_OpticalFlow/dataset/kth_actions_test", th)

        print "Predicted Labels : "
        print prediction_labels
        print "Actual Labels : "
        print actual_labels
    
        # Save these matrices in files
        outfile = file("test_results/predicted_labels_alexnet_mirror_bgThresh_"+str(th)+"_testSet.bin", "wb")
        np.save(outfile, prediction_labels)
        outfile.close()
    
#        outfile = file("test_results/actual_labels_alexnet_mirror_bgThresh.bin", "wb")
#        np.save(outfile, actual_labels)
#        outfile.close()
    
    
        # get the list of prediction for all videos, length is no_of_videos_in_set
        predictions = np.argmax(prediction_labels, axis=1)
        # prediction matrix for prediction X ground_truth matrix
        predictions_mat = np.zeros((6,6))
    
        for i in range(0,predictions_mat.shape[0]):
            for j in range(0,predictions_mat.shape[1]):
                predictions_mat[i,j] = sum((predictions==i) & (actual_labels==j))
    
        print "Final Predictions :"
        print predictions_mat
        print "Final accuracy (%age) : "+str(100.*np.trace(predictions_mat)/np.sum(predictions_mat))

###########################################################################
    # load the test result files to compute accuracies
    # Uncomment this block to load evaluation results from files and display stats
    
#    infile = file("test_results/predicted_labels_alexnet_mirror_bgThresh_105000_testSet.bin", "rb")
#    prediction_labels = np.load(infile)
#    infile.close()
#    
#    infile = file("test_results/actual_labels_alexnet_mirror_bgThresh.bin", "rb")
#    actual_labels = np.load(infile)
#    infile.close()
#    
#    predictions = np.argmax(prediction_labels, axis=1)
#    predictions_mat = np.zeros((6,6))
#    for i in range(0,predictions_mat.shape[0]):
#            for j in range(0,predictions_mat.shape[1]):
#                predictions_mat[i,j] = sum((predictions==i) & (actual_labels==j))
#    
#    print "Final Predictions :"
#    print predictions_mat
#    print "Final accuracy (%age) : "+str(100.*np.trace(predictions_mat)/np.sum(predictions_mat))
