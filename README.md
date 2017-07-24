# KTH_OpticalFlow
This repository contains the code supporting the paper "Action Recognition using Optical Flow Visualizations" by Arpan Gupta and M. Sakthi Balan.

## Description of files

* optical_flow.py : Used for creation of training LMDB (and validation LMDB for checking accuracy of model trained but that is not actual accuracy on validation set). Only the frames where actions occur is considered. Thereofore, it uses kth_read_sequences.py, where kth_sequences annotations are used. 

* create_test_set.py : Used for creation of validation and test set LMDB (for all the frames of the videos).
                                                                         
* optical_flow_alexnet_mirror.prototxt : Model for training (using Caffe)

* optical_flow_alexnet_mirror_solver.prototxt : Solver file

* OF_alexnet_mirror_snap_iter_40000.caffemodel : Trained model file.

* deploy_OF_alexnet_mirror.prototxt : Defined model, used at the time of testing (in evaluate_model.py)

* evaluate_model.py : Used to test the trained model on all the frames of a test set and calculate the accuracy. Background subtraction thresholding used here.

* Misc. files : explore_hsv.py , lmdb_operations.py