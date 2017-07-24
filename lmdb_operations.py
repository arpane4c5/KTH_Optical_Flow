# Script to convert the optical flows into lmdb data
import numpy as np
import os
import lmdb
import caffe
import cv2

# call this for data_dim = (3, 120, 160) to save BGR data to lmdb
def create_lmdb(d, no_of_classes, data_path,data_dim=(2,120,160), prefix='train'):
	# import lmdb before calling this function
	# data is the training data matrix with (N x C x W x H)
	# N is the batch size, C is the no of channels (2 in optical flow)
	# W = 120 , H = 160
	# label is a 1 to 6 for class labels corresponding to boxing, running etc
	print('function called !!')
	N = 100
	idCount = 0
	# To save the data into lmdb. 
	# prepare the database
	#x = np.zeros((N,) + data_dim, dtype=float)
	x = np.zeros((N,) + data_dim, dtype="uint8")
	# y = np.zeros((N,) + (no_of_classes, ), dtype=np.uint8)
	y = np.zeros(N, dtype=int)
	map_size = x.nbytes * N *2
	env = lmdb.open(os.path.join(data_path,prefix+'_bgr_flow_lmdb'), map_size=map_size)

	try:
		count = 0		# No of optical flow 3D matrices read from files
		countFiles = 0
		for filename in d.keys():
			#print(str(count)+"reading : "+filename)
			# open a file. In a loop put iteration value in place of boxingvideo25
			filePath = os.path.join(os.path.join(data_path,'kth_'+prefix),
				'kth_'+prefix+filename+'.bin')
			print(str(count)+' Reading' + filePath)
			flowFile = file(filePath, 'rb')
			countFiles = countFiles + 1
			# read repeatedly from the file till the end of file is met,
			# label for all the frames inside this file is the same
			# EOF will throw an exception
			try:
				while(1):
					flow = np.load(flowFile)
					# Change the shape of matrix to (2, 120, 160) from (120, 160, 2)
					flow = np.rollaxis(flow, 2)
					# convert the OF matrix to (3, 120, 160) BGR matrix
					bgr = OF_matrix_to_bgr(flow)

					if count == 0:
						x.fill(0)
						y.fill(0)
					# Copy the values in x and y arrays
					#x[count,...] = flow
					x[count,...] = bgr
					#y[count, int(d[filename])-1] = 1
					y[count] = int(d[filename]) - 1
					print("label added : "+str(y[count]))
					count = count + 1
					
					# print(str(count) + ' ' + str(x.shape))

					# if x and y buffers are full, then write to lmdb
					if count == N:
						# txn is a Transaction object
						with env.begin(write=True) as txn:
							for i in range(N):
								datum = caffe.proto.caffe_pb2.Datum()
								datum.channels = x.shape[1]
								datum.height = x.shape[2]
								datum.width = x.shape[3]
								datum.data = x[i].tobytes()  # or .tostring() if numpy < 1.9
								# Convert label into a 6D label vector
								#datum.label = y[i,...]
								datum.label = y[i]
								#datum.label[d[filename]-1] = 1

								# str_id = '{:08}'.format((countFiles-1)*N+i)
								str_id = '{:08}'.format(idCount)
								# The encode is only essential in Python 3
								txn.put(str_id.encode('ascii'), datum.SerializeToString())
								idCount = idCount + 1
						count = 0
			except IOError:

				if countFiles == len(d):
					# txn is a Transaction object
					with env.begin(write=True) as txn:
						for i in range(count):
							datum = caffe.proto.caffe_pb2.Datum()
							datum.channels = x.shape[1]
							datum.height = x.shape[2]
							datum.width = x.shape[3]
							datum.data = x[i].tobytes()  # or .tostring() if numpy < 1.9
							# Convert label into a 6D label vector
							#datum.label = y[i,...]
							datum.label = y[i]
							
							#datum.label[d[filename]-1] = 1
							# str_id = '{:08}'.format((countFiles-1)*N+i)
							str_id = '{:08}'.format(idCount)
							# The encode is only essential in Python 3
							txn.put(str_id.encode('ascii'), datum.SerializeToString())
							idCount = idCount + 1

			finally:
				flowFile.close()
	except IOError:
		print("Error : Out Try block")
		pass


	return True


# receive a (2, 120, 160) OF matrix and return the (3, 120, 160) BGR matrix
def OF_matrix_to_bgr(f):
    f = np.rollaxis(np.rollaxis(f, 2), 2)
    #print(f.shape)
    hsv = np.zeros((120,160,3), dtype='uint8')
    #print(hsv.shape)
    hsv[...,1] = 255
    mag, ang = cv2.cartToPolar(f[...,0], f[...,1])
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    hsv[...,0] = (ang*180/np.pi/2)
    # getting a (120, 160, 3) dimension bgr matrix
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    # rollaxis to get (3, 120, 160) dimension bgr matrix
    bgr = np.rollaxis(bgr, 2)
    return bgr


def read_from_lmdb(path, lmdb_dir):
	env = lmdb.open(os.path.join(path, lmdb_dir), readonly=True)
	# with env.begin() as txn:
	# 	raw_datum = txn.get(b'00000000')

	# datum = caffe.proto.caffe_pb2.Datum()
	# datum.ParseFromString(raw_datum)

	# flat_x = np.fromstring(datum.data, dtype=np.uint8)
	# x = flat_x.reshape(datum.channels, datum.height, datum.width)
	# y = datum.label

	# Iterating <key, value> pairs is also easy:

	with env.begin() as txn:
		cursor = txn.cursor()
		for key, value in cursor:
			print(key, value)
			break


# import numpy as np 
# import lmdb
# import caffe

# N = 1000


# Let's pretend this is interesting data
# X = np.zeros((N, 3, 32, 32), dtype=np.uint8)
# y = np.zeros(N, dtype=np.int64)

# # We need to prepare the database for the size. We'll set it 10 times
# # greater than what we theoretically need. There is little drawback to
# # setting this too big. If you still run into problem after raising
# # this, you might want to try saving fewer entries in a single
# # transaction.
# map_size = X.nbytes * 10

# env = lmdb.open('optFlow_lmdb', map_size=map_size)

# with env.begin(write=True) as txn:
#     # txn is a Transaction object
#     for i in range(N):
#         datum = caffe.proto.caffe_pb2.Datum()
#         datum.channels = X.shape[1]
#         datum.height = X.shape[2]
#         datum.width = X.shape[3]
#         datum.data = X[i].tobytes()  # or .tostring() if numpy < 1.9
#         datum.label = int(y[i])
#         str_id = '{:08}'.format(i)

#         # The encode is only essential in Python 3
#         txn.put(str_id.encode('ascii'), datum.SerializeToString())


# # Open and inspect an existing lmdb 
# import numpy as np
# import lmdb
# import caffe

# env = lmdb.open('mylmdb', readonly=True)
# with env.begin() as txn:
#     raw_datum = txn.get(b'00000000')

# datum = caffe.proto.caffe_pb2.Datum()
# datum.ParseFromString(raw_datum)

# flat_x = np.fromstring(datum.data, dtype=np.uint8)
# x = flat_x.reshape(datum.channels, datum.height, datum.width)
# y = datum.label

# Iterating <key, value> pairs is also easy:

# with env.begin() as txn:
#     cursor = txn.cursor()
#     for key, value in cursor:
#         print(key, value)
