import os
import cv2
import torch
import datetime
import numpy as np
from torch.multiprocessing import Pool
from torch.autograd import Variable

from darknet import Darknet19
from datasets.pascal_voc import VOCDataset
import utils.yolo as yolo_utils
import utils.network as net_utils
from utils.timer import Timer
import cfgs.config as cfg                               #see that this imports darknet19exp1
from random import randint

trained_model = '/home/foredawnlin/vision/new_yolo2/yolo2-pytorch/models/yolo-voc.weights.h5'        #path to yolo weight file 
thresh = 0.2                                          #hyper parameter

net = Darknet19()
net_utils.load_net(trained_model, net)
net.cuda()
net.eval()

net_loss = Darknet19()
net_utils.load_net(trained_model, net_loss)
net_loss.cuda()
net_loss.train()
print('load net succ...')

def preprocess(img_var): #fname
    # return fname
    # image = cv2.imread(fname)
    #print img_var.shape
    image_tensor = img_var
    im_data = np.expand_dims(
        yolo_utils.preprocess_test((image_tensor, None, cfg.multi_scale_inp_size), 0)[0], 0)
    return image_tensor, im_data



def get_yolo_loss(im_fake,im_real):
        
    #INPUT:  GET BATCH AS (batch_size x M x N x 3) ###
    #OUTPUT:  returns a list of size batch_size indexed accordingly ###

    ## multi processing 
    # pool = Pool(processes=1)

    # (image_blur_batch, im_data_blur_batch) = pool.imap(preprocess, im_fake, chunksize=1)
    # (image_sharp_batch, im_data_sharp_batch) = pool.imap(preprocess, im_real, chunksize=1)

    ## CPU processing 
    #print np.asarray(list(map(preprocess, im_fake))).shape
    # (image_blur_batch, im_data_blur_batch) = map(preprocess, im_fake)
    # (image_sharp_batch, im_data_sharp_batch) = map(preprocess, im_real)
    im_fake_np = np.asarray(map(preprocess, im_fake))
    im_real_np = np.asarray(map(preprocess, im_real))

    loss_list = []
    bbox_loss_list = []
    iou_loss_list = []
    cls_loss_list = []
    
    ###### FORWARD PROP SHARP IMAGE ######
    # print("*****REAL*****")
    # for _, (image_blur, im_data_blur, image_sharp, image_data_sharp) in enumerate(
            # zip(im_blur_batch,im_data_blur_batch,image_sharp_batch,im_data_sharp_batch)):

    for _, (image_blur, im_data_blur, image_sharp, im_data_sharp) in enumerate(
            [np.hstack((x,y)) for x,y in zip(im_fake_np,im_real_np)]):        
        
        im_data_sharp = net_utils.np_to_variable(im_data_sharp, is_cuda=True,                   #convert to Pytorch Variable
                                           volatile=True).permute(0, 3, 1, 2)

        bbox_pred, iou_pred, prob_pred,_,_,_ = net(im_data_sharp)                               #get predictions for sharp image
        # to numpy
        bbox_pred = bbox_pred.data.cpu().numpy()                
        iou_pred = iou_pred.data.cpu().numpy()
        prob_pred = prob_pred.data.cpu().numpy()

        # print (np.shape(bbox_pred),np.shape(iou_pred),np.shape(prob_pred))

        bboxes, scores, cls_inds = yolo_utils.postprocess(bbox_pred,                            #CHECK
                                                          iou_pred,
                                                          prob_pred,
                                                          image_sharp.shape,
                                                          cfg,
                                                          thresh
                                                          )    

        ## add gt_boxes, gt_classes from these predictions ##

        gt_boxes = []
        gt_classes = []
        bboxes_per_image = []
        classes_per_image = []

        # populating all_boxes with bbox and score info                                         # dont need 
        for j in range(20):             #put num_of_classes here
            inds = np.where(cls_inds == j)[0]
            if len(inds) == 0:
                pass
                continue
            c_bboxes = bboxes[inds]

            bboxes_per_image.append(c_bboxes)
            for k in range(inds.shape[0]):
                classes_per_image.append(j)

        if(len(classes_per_image)==0):
        	print("No bbox found for given threshold")
        	return 0,0,0,0

        else:

        #process into gt_boxes
    	    gt_classes.append(np.hstack(classes_per_image))
    	    gt_boxes.append(np.vstack(bboxes_per_image))


    	    #print(gt_classes,gt_boxes)
    	    im_data_blur = net_utils.np_to_variable(im_data_blur,                             #forward pass of blur image
    	                                   is_cuda=True,
    	                                   volatile=False).permute(0, 3, 1, 2)


    	    
            ###### LOSS PROP ######
            ##IMPORTANT     #add size_index argument according to the sharp image pass?

    	    size_index = 0 ## setting images size index 0:320, 1:352, 2:384, 3:416, 4:448, 5:480, 6:512, 7:544, 8:576')
    	    bbox_pred_blur, iou_pred_blur, prob_pred_blur,bbox_loss, iou_loss, cls_loss = net_loss(im_data_blur, gt_boxes, gt_classes, [[]])                                  #check size_index (0 is used in train for some weird reason), also check preprocess

    	    #Printing bboxes on blur
    	    bbox_pred_blur = bbox_pred_blur.data.cpu().numpy()                
    	    iou_pred_blur = iou_pred_blur.data.cpu().numpy()
    	    prob_pred_blur = prob_pred_blur.data.cpu().numpy()

    	    # print (np.shape(bbox_pred_blur),np.shape(iou_pred_blur),np.shape(prob_pred_blur))

    	    thresh_b = 0  #threshold for blur image detection(ojas)
    	    bboxes_blur, scores_blur, cls_inds_blur = yolo_utils.postprocess(bbox_pred,                            #CHECK
    	                                                      iou_pred,
    	                                                      prob_pred,
    	                                                      image_blur.shape,
    	                                                      cfg,
    	                                                      thresh_b
    	                                                      )    
    	    # print (bboxes_blur,cls_inds_blur)

    	    loss = bbox_loss + iou_loss + cls_loss

    	    #print(loss,bbox_loss,iou_loss,cls_loss)

    	    loss_list.append(loss)
    	    bbox_loss_list.append(bbox_loss)
    	    iou_loss_list.append(iou_loss)
    	    cls_loss_list.append(cls_loss)

	    return np.mean(loss_list), np.mean(bbox_loss_list), np.mean(iou_loss_list), np.mean(cls_loss_list)


