from python_pfm import readPFM
from scipy.misc import imread
import numpy as np
import os, sys

if len(sys.argv)-1 != 2:
    print("Use this tool to calculate error\n"
          "Usage for single image pair:\n"
          "    ./evaluation.py D_gt.pfm D_est.pfm\n"
          "\n")
disp_files = sys.argv[1:]

#disp_files = ['0000000-gt.pfm', 'dispnet-corr1d-pred-0000000.pfm']
for disp_file in disp_files:
    if not os.path.isfile(disp_file):
        print('Disparity map %s not found' % disp_file)
        sys.exit(1)

#DEST = 'data/disp/test/'
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

def sizes_equal(size1, size2):
    return size1[0] == size2[0] and size1[1] == size2[1]

if disp_files[0].split('.')[-1] == 'pfm':
    disp_gth, scale1 = readPFM(disp_files[0])
elif disp_files[0].split('.')[-1] == 'png':
    disp_gth = (imread(disp_files[0])).astype(float) / 256

disp_est, scale2 = readPFM(disp_files[1])
disp_est[disp_est<0]=0

sz_gth = disp_gth.shape
sz_est = disp_est.shape

if not sizes_equal(sz_gth, sz_est):
    print('Disparity maps do not have the same size.')
    sys.exit(1)

er1 = end_point_error(disp_gth, disp_est)
er2 = D1_error(disp_gth, disp_est, (3,0.05))
print('EPE: %f' % er1)
print('D1 error: %f' % er2)
