import caffe
import lmdb
import numpy as np
import scipy.io as sio
import cv2
import numpngw
import time


def D1_error(D_gt,D_est,tau):
	E = abs(D_gt-D_est)
	n_err   = ((D_gt>0)*(E>tau[0])*(E/(D_gt+1e-10)>tau[1])).sum().astype(float)
	n_total = (D_gt>0).sum()
	d_err = n_err/n_total
	return d_err


def end_point_error(D_gt, D_est):
	E = abs(D_gt-D_est)
	n_total = (D_gt>0).sum()
	E[D_gt == 0] = 0
	return E.sum() / n_total


caffe.set_device(0)
caffe.set_mode_gpu()
model_def = 'models/dispnet/finetune_deploy.prototxt'
model_weights = 'disp7_10_iter_610000.caffemodel'
#model_weights = 'kitti_loss_iter_90000.caffemodel'
net = caffe.Net(model_def, model_weights, caffe.TEST)

N = 80#4370
er1 = 0
er2 = 0
time1 = time.time()
for i in range(N):
	net.forward()
	gth = net.blobs['scaled_disp_gt'].data[0,0,:,:]
	out = net.blobs['predict_disp0'].data[0,0,:,:]
	out[out<0] = 0
	er1 += end_point_error(gth,out)
	#out = cv2.resize(out1,gth.shape[::-1])
	er2 += D1_error(gth,out,(3,0.05))
	print 'batch:%04d' %i

time2 = time.time()
er1 /= N
er2 /= N

print r'EPE:',er1
print r'D1:', er2
print time2 - time1, 'seconds'

out = np.uint16(out*256)
gth = np.uint16(gth*256) 
numpngw.write_png('gthk.png',gth)
numpngw.write_png('outk.png',out)

get outk.png gthk.png
net.forward()
gth = net.blobs['scaled_disp_gt'].data[0,0,:,:]
out = net.blobs['predict_disp0'].data[0,0,:,:]
out[out<0] = 0
out = np.uint16(out*256)
gth = np.uint16(gth*256) 
numpngw.write_png('gthk.png',gth)
numpngw.write_png('outk.png',out)
'''
net.blobs['img0'].reshape(1,3,384, 1280) 
net.blobs['img1'].reshape(1,3,384, 1280)
net.blobs['scaled_disp_gt'].reshape(1,1,384, 1280)

mu0 = np.load('data/stereo2015_zip/training/img0_train_mean.npy')
mu1 = np.load('data/stereo2015_zip/training/img1_train_mean.npy')
scale = 1/255.0

lmdb_path_img0 = 'data/stereo2015_zip/training/stereo_val_img0_lmdb'
lmdb_path_img1 = 'data/stereo2015_zip/training/stereo_val_img1_lmdb'
lmdb_path_gth = 'data/stereo2015_zip/training/stereo_val_gth_lmdb'
'''
net.blobs['img0'].reshape(1,3,540,960) 
net.blobs['img1'].reshape(1,3,540,960)
net.blobs['scaled_disp_gt'].reshape(1,1,540,960)

mu0 = np.load('data/flyingthings/img0_train_mean.npy')
mu1 = np.load('data/flyingthings/img1_train_mean.npy')
scale = 1/255.0

lmdb_path_img0 = 'data/flyingthings/stereo_val_img0_lmdb'
lmdb_path_img1 = 'data/flyingthings/stereo_val_img1_lmdb'
lmdb_path_gth = 'data/flyingthings/stereo_val_gth_lmdb'


#open lmdb
lmdb_env_img0 = lmdb.open(lmdb_path_img0)
lmdb_env_img1 = lmdb.open(lmdb_path_img1)
lmdb_env_gth = lmdb.open(lmdb_path_gth)
#begin transaction
lmdb_txn_img0 = lmdb_env_img0.begin()
lmdb_txn_img1 = lmdb_env_img1.begin()
lmdb_txn_gth = lmdb_env_gth.begin()
#get cursor
lmdb_cursor_img0 = lmdb_txn_img0.cursor()
lmdb_cursor_img1 = lmdb_txn_img1.cursor()
lmdb_cursor_gth = lmdb_txn_gth.cursor()
#get data object
datum_img0 = caffe.proto.caffe_pb2.Datum()
datum_img1 = caffe.proto.caffe_pb2.Datum()
datum_gth = caffe.proto.caffe_pb2.Datum()

for k0, v0 in lmdb_cursor_img0:
	for k1, v1 in lmdb_cursor_img1:
		for kg, vg in lmdb_cursor_gth:
			#parse back to datum
			datum_img0.ParseFromString(v0)
			datum_img1.ParseFromString(v1)
			datum_gth.ParseFromString(vg)
			img0 = (caffe.io.datum_to_array(datum_img0)-mu0)*scale
			img1 = (caffe.io.datum_to_array(datum_img1)-mu1)*scale
			disp = caffe.io.datum_to_array(datum_gth)/256.0
			net.blobs['img0'].data[0] = img0
			net.blobs['img1'].data[0] = img1
			net.blobs['scaled_disp_gt'].data[0] = disp
			net.forward()
			gth = net.blobs['blob49'].data[0,0,:,:]
			out1 = net.blobs['predict_disp1'].data[0,0,:,:]
			out1[out1<0] = 0
			abs(gth-out1).mean()
			#out = cv2.resize(out1,gth.shape[::-1])
			print disp_error(gth,out1,(5,0.05))
			out = np.uint16(out1*256)
			gth = np.uint16(gth*256) 
			numpngw.write_png('gthk.png',gth)
			numpngw.write_png('outk.png',out)	

			
lmdb_env_img0.close()
lmdb_env_img1.close()
lmdb_env_gth.close()

solver = caffe.AdamSolver('models/dispnet/solver.prototxt')
solver.net.copy_from('kitti_loss_iter_90000.caffemodel')
solver.step(1)

gth = solver.net.blobs['scaled_disp_gt'].data[0,0,:,:]
out1 = solver.net.blobs['predict_disp1'].data[0,0,:,:]
gth1 = solver.net.blobs['blob49'].data[0,0,:,:]


er2 = (abs(gth1-out1)).sum()/gth1.size
out = cv2.resize(out1,gth.shape[::-1])
er1 = ((gth>0).astype(float)*(abs(gth-out)>3.0).astype(float)).sum()/gth.size
out = np.uint16(out1*256)
gth = np.uint16(gth*256) 
numpngw.write_png('gthk.png',gth)
numpngw.write_png('outk.png',out)

get outk.png gthk.png
sio.savemat('out.mat', {'out': out,'gth':gth})

solver = caffe.AdamSolver('models/dispnet/finetune_solver.prototxt')
solver.net.copy_from('kitti_loss_iter_90000.caffemodel')

gth6 = solver.net.blobs['blob24'].data[0,:,:,:]
out6 = solver.net.blobs['predict_flow6'].data[0,:,:,:]
gth5 = solver.net.blobs['blob29'].data[0,:,:,:]
out5 = solver.net.blobs['predict_flow5'].data[0,:,:,:]
gth4 = solver.net.blobs['blob34'].data[0,:,:,:]
out4 = solver.net.blobs['predict_flow4'].data[0,:,:,:]
gth3 = solver.net.blobs['blob39'].data[0,:,:,:]
out3 = solver.net.blobs['predict_flow3'].data[0,:,:,:]
gth2 = solver.net.blobs['blob44'].data[0,:,:,:]
out2 = solver.net.blobs['predict_flow2'].data[0,:,:,:]
gth1 = solver.net.blobs['blob49'].data[0,0,:,:]
out1 = solver.net.blobs['predict_flow1'].data[0,0,:,:]

out = np.uint16(out1*256)
gth = np.uint16(gth1*256) 
er = (abs(gth1-out1)>3.0).astype(float).sum()/gth1.size
numpngw.write_png('gthk.png',gth[0,:,:])
numpngw.write_png('outk.png',out[0,:,:])
