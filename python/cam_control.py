import ue4cv, time
import numpy as np
from math import pi
import caffe
from python_pfm import readPFM
from math import ceil
import numpngw
import matplotlib.pyplot as plt
import os, sys


# Try to connect our python client to the game

N_img = 10
Cdist = 40
#for i in range(N_img):
PATH = 'unreal/'
imgL_name = 'imgL.png'
imgR_name = 'imgR.png'
depth_name = 'depth.png'
disp_est_name = 'disp.png'
net = None

def message_handler(message):
    print 'Got server message %s' % repr(message)
    if message == 'clicked':
        loc_l = np.asarray(ue4cv.client.request('vget /camera/0/location').split(' ')).astype(float)
        rot = np.asarray(ue4cv.client.request('vget /camera/0/rotation').split(' ')).astype(float) / 180.0 * pi
        dirct = [np.cos(rot[0])*np.cos(rot[1]), np.cos(rot[0])*np.sin(rot[1]), np.sin(rot[0])]
        left = np.cross(dirct, [0., 0., 1.])
        left = left / ((left**2).sum())**(1/2)
        loc_r = loc_l - left * Cdist
        ue4cv.client.request('vset /camera/0/location %f %f %f' % (loc_l[0], loc_l[1], loc_l[2]))
        filename = ue4cv.client.request('vget /camera/0/lit ' + PATH + imgL_name)
        print 'Image is saved to %s' % (PATH+imgL_name)

        filename = ue4cv.client.request('vget /camera/0/depth ' + PATH + depth_name)
        print 'Depth is saved to %s' % (PATH+depth_name)

        ue4cv.client.request('vset /camera/0/location %f %f %f' % (loc_r[0], loc_r[1], loc_r[2]))
        filename = ue4cv.client.request('vget /camera/0/lit ' + PATH + imgR_name)
        print 'Image is saved to %s' % (PATH+imgR_name)

        img_files = (PATH+imgL_name, PATH+imgR_name, PATH+disp_est_name)
        process(img_files)

#loc_l = np.random.uniform(-100, 100, 3)
#img_files = ['models/DispNetCorr1D/data/0000000-imgL.ppm',
#                'models/DispNetCorr1D/data/0000000-imgR.ppm']
def init_caffe():
    global net, transformer
    caffe.set_mode_gpu()
    caffe.set_device(1)
    model_def = 'deploy_unreal.prototxt'
    model_weights = 'model/DispNetCorr1D_CVPR2016.caffemodel'

    net = caffe.Net(model_def, model_weights, caffe.TEST)

    transformer = caffe.io.Transformer({'data': net.blobs['img0'].data.shape})

    transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
    #transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

def process(image_files):
#    if not net:
#        init_caffe()
    caffe.set_mode_gpu()
    caffe.set_device(1)
    model_def = 'deploy_unreal.prototxt'
    model_weights = 'model/DispNetCorr1D_CVPR2016.caffemodel'

    net = caffe.Net(model_def, model_weights, caffe.TEST)

    transformer = caffe.io.Transformer({'data': net.blobs['img0'].data.shape})

    transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
    #transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

    img_L0 = caffe.io.load_image(image_files[0])
    img_R0 = caffe.io.load_image(image_files[1])

    img_L = transformer.preprocess('data', img_L0)
    img_R = transformer.preprocess('data', img_R0)

    net.blobs['img0'].data[...] = img_L
    net.blobs['img1'].data[...] = img_R
    print('caffe initialized...')
    net.forward()

    output = net.blobs['predict_disp_final'].data[0,0,:,:]
    disp_est = -1*output
    numpngw.write_png(image_files[2], np.uint16(disp_est*256))

if __name__ == '__main__':
    init_caffe()
    ue4cv.client.message_handler = message_handler
    ue4cv.client.connect()
    fig, ax = plt.subplots()
    image = np.zeros((300, 300))
    ax.imshow(image)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    ue4cv.client.disconnect()
