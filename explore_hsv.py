#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 21:46:09 2017

Description: Exploring HSV space 

@author: Arpan
"""

import cv2
import numpy as np

def waitTillEscPressed():
    while(True):
        if cv2.waitKey(10)==27:
            print("Esc Pressed")
            return

if __name__=="__main__":
    hsv = np.zeros((120, 160, 3), dtype="uint8")
    #print hsv
    
    hsv[...,1] = 255
#    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
#    
#    hsv[...,0] = ang*180/np.pi/2
#    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
#    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    print "Displaying frames : "
    i = 0
    while(i<256):
        hsv[..., 0] = i
        j = 0
        while(j<256):
            hsv[...,2] = j
            j=j+5
            print "hsv[...,0] = "+str(i)+"  :: hsv[...,2] = "+str(j)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            cv2.imshow("HSV Frame", hsv)
            cv2.imshow("BGR Frame", bgr)
            waitTillEscPressed()
        i = i+5
        
    
    #cap = cv2.VideoCapture("dataset/kth_actions_train/person11_boxing_d1_uncomp.avi")
        
    #if not cap.isOpened():
    #    print "Error in reading the video file ! Abort !"
    #    sys.exit(0)
        
    cv2.destroyAllWindows()
