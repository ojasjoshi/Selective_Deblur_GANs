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
    image_tensor = img_var
    im_data = np.expand_dims(
        yolo_utils.preprocess_test((image_tensor, None, cfg.multi_scale_inp_size), 0)[0], 0)
    return image_tensor, im_data

def get_yolo_loss(im_fake,im_real):
    # trained_model = '/home/foredawnlin/vision/new_yolo2/yolo2-pytorch/models/yolo-voc.weights.h5'        #path to yolo weight file 
    # thresh = 0.01                                           #hyper parameter

    # net = Darknet19()
    # net_utils.load_net(trained_model, net)
    # net.cuda()
    # net.eval()

    # net_loss = Darknet19()
    # net_utils.load_net(trained_model, net_loss)
    # net_loss.cuda()
    # net_loss.train()
    # print('load net succ...')

    # pool = Pool(processes=1)

    image_blur,im_data_blur = preprocess(im_fake)

    loss_list = []
    bbox_loss_list = []
    iou_loss_list = []
    cls_loss_list = []

    image_sharp, im_data_sharp = preprocess(im_real)                     

    #forward for sharp
    # print("*****REAL*****")
    im_data_sharp = net_utils.np_to_variable(im_data_sharp, is_cuda=True,                   #convert to Pytorch Variable
                                       volatile=True).permute(0, 3, 1, 2)
    # im_data_sharp = Variable(im_data_sharp).cuda().permute(0, 3, 1, 2)


    bbox_pred, iou_pred, prob_pred,_,_,_ = net(im_data_sharp)                                     #get predictions for sharp image
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

    # add gt_boxes, gt_classes from these predictions
    im2show = yolo_utils.draw_detection(image_sharp, bboxes, scores, cls_inds, cfg)

    if im2show.shape[0] > 1100:
        im2show = cv2.resize(im2show,
                             (int(1000. *
                                  float(im2show.shape[1]) / im2show.shape[0]),
                              1000))
    # cv2.imshow('sharp', im2show)
    # cv2.waitKey(0)

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
    	return [[0]],[[0]],[[0]],[[0]]

    else:
    #process into gt_boxes
	    gt_classes.append(np.hstack(classes_per_image))
	    gt_boxes.append(np.vstack(bboxes_per_image))


	    #print(gt_classes,gt_boxes)
	    im_data_blur = net_utils.np_to_variable(im_data_blur,                             #forward pass of blur image
	                                   is_cuda=True,
	                                   volatile=False).permute(0, 3, 1, 2)
	    # im_data_blur = Variable(im_data_blur).permute(0, 3, 1, 2)

	    #add size_index argument?


	    # print("*****BLUR*****")
	    # size_index = randint(0, len(cfg.multi_scale_inp_size) - 1)
	    size_index = 0
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
	                                                      image_sharp.shape,
	                                                      cfg,
	                                                      thresh_b
	                                                      )    
	    # print (bboxes_blur,cls_inds_blur)
	    # add gt_boxes, gt_classes from these predictions
	    im2show_blur = yolo_utils.draw_detection(image_blur, bboxes_blur, scores_blur, cls_inds_blur, cfg, thresh_b)

	    if im2show_blur.shape[0] > 1100:
	        im2show_blur = cv2.resize(im2show,
	                             (int(1000. *
	                                  float(im2show_blur.shape[1]) / im2show.shape[0]),
	                              1000))
	    # cv2.imshow('blur', im2show_blur)
	    # cv2.waitKey(0)

	    loss = bbox_loss + iou_loss + cls_loss

	    #print(loss,bbox_loss,iou_loss,cls_loss)

	    loss_list.append(loss)
	    bbox_loss_list.append(bbox_loss)
	    iou_loss_list.append(iou_loss)
	    cls_loss_list.append(cls_loss)

	    return loss_list, bbox_loss_list, iou_loss_list, cls_loss_list


