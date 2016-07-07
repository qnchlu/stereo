import caffe
from python_pfm import readPFM
import matplotlib.pyplot as plt
import numpy as np
import os, sys

if len(sys.argv)-1 != 2:
    print("Use this tool to calculate error\n"
          "Usage for single image pair:\n"
          "    ./run_dispnet.py img_L.png img_R.png\n"
          "\n")
img_files = sys.argv[1:]
#img_files = ['models/DispNetCorr1D/data/0000000-imgL.ppm',
#                'models/DispNetCorr1D/data/0000000-imgR.ppm']
caffe.set_device(0)
caffe.set_mode_gpu()
model_def = 'models/DispNetCorr1D/deploy_fly.prototxt'
model_weights = 'models/DispNetCorr1D/model/DispNetCorr1D_CVPR2016.caffemodel'

net = caffe.Net(model_def, model_weights, caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['img0'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
#transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

img_L0 = caffe.io.load_image(img_files[0])
img_R0 = caffe.io.load_image(img_files[1])
img_L = transformer.preprocess('data', img_L0)
img_R = transformer.preprocess('data', img_R0)

net.blobs['img0'].data[...] = img_L
net.blobs['img1'].data[...] = img_R
net.forward()

output = net.blobs['predict_disp_final'].data[0,0,:,:]

disp_est = -1*output
#minus = np.uint8(np.zeros(output.shape))
#minus[disp_est<0] = 255

# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap
#plt.imshow(np.uint16(abs(256*output)))
#plt.imshow(minus)
plt.imshow(abs(output))
plt.show()
