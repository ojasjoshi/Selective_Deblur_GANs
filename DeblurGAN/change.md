### solve bluring algorith square image size restricction ###
## bluring algo ##
1. comment out bluring algorithm sqaure image restriction 
###### ends ##########


################### solve deblur image cropping problem ################
## models/network.py ## 
1. line 148 to line 152: enforce the G network output the same size as input. G network outputs residual of input image. Adding them gives deblured image.

## test.py command line ##
1. add "--resize_or_crop resize" ; default is crop 
################### ends ################

## train instructions ##
1. Loss: In models/conditional_gan_model.py, line 89 to lin 101  (add YOLO loss here)
2. Loss function low level: In models/losses.py; (tune loss here)
	- perceptua loss: line 24 to line 45; 
	- wGANGP: line 126 to line 155;
3. Train alternating frequency for netD and netG: in models/conditional_gan_model.py: line 103 to line 113: self.criticUpdates (default =5)


####### solve training problem: combine data #####
1. changed dataset/combine_A_and_B.py; the data is in directory: ~/data/blur_sharp/DeblurGAN_blur/blurred_sharp/blur_sharp_pair/train  
2. Note: data has to be put into a folder called "train"


## question ##
1. base opt: "which direction" ?

## left ## 
1. to figure out conditional_gan_model strcture 
