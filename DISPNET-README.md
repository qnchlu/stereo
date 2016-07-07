# Caffe with DispNet v0.5 

This the relase of the CVPR 2016 versions of DispNet and DispNetCorr1D. 

NOTE: We will publish code for training networks after CVPR. 

It comes as a fork of the current caffe master branch and with trained networks,
as well as examples to use them.

To get started with DispNet, first compile caffe, by configuring a

    "Makefile.config" (example given in Makefile.config.example)

then make with 

    $ make -j 5 all tools

Go to this folder:

    ./dispnet-release/models/DispNet/

From this folder you can execute the scripts we prepared:
To try out DispNet on a sample image pair, run

    ./demo.py data/0000000-imgL.ppm data/0000000-imgR.ppm

## License and Citation

Please cite this paper in your publications if you use FlowNet for your research:

    @inproceedings{MIFDB16,
      author       = "N. Mayer and E. Ilg and P. H{\"a}usser and P. Fischer and D. Cremers and A. Dosovitskiy and T. Brox",
      title        = "A Large Dataset to Train Convolutional Networks for Disparity, Optical Flow, and Scene Flow Estimation",
      booktitle    = "IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",
      month        = "June",
      year         = "2016",
      url          = "http://lmb.informatik.uni-freiburg.de/Publications/2016/MIFDB16"
    }


