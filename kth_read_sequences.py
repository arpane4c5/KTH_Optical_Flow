#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 17:18:10 2017

@author: hadoop
"""

import os
import csv

# read the frames of action into a dictionary as
# key: filename
# value: string of sequences defined as "1-10, 40-90, ..."
# dictionary has 600 values, one is missing
def read_sequences_file(filePath):
    d = {}
    with open(filePath,'rb') as tsvin:
        tsvin = csv.reader(tsvin, delimiter='\t')
        #csvout = csv.writer(csvout)
        for row in tsvin:
            # if row is not and empty list then add the 
            if len(row)>0:
                  key = row[0].strip()
                  value = row[-1].strip()
                  d[key] = value
    print len(d)
    return d

if __name__ == "__main__":
    
    print "Read a TSV file and extract the contents :"
    filename = "kth_sequences.txt"
    d = read_sequences_file(os.path.join("/home/hadoop/VisionWorkspace/KTH_OpticalFlow/dataset",filename))
    
    