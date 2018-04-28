import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .losses import init_loss
import cv2

import sys
sys.path.insert(0,"/home/foredawnlin/vision/new_yolo2/yolo2-pytorch")
#from diff_temp import get_yolo_loss
from diff_temp_batch import get_yolo_loss



try:
	xrange          # Python2
except NameError:
	xrange = range  # Python 3

class ConditionalGAN(BaseModel):
	def name(self):
		return 'ConditionalGANModel'

	def initialize(self, opt):
		BaseModel.initialize(self, opt)
		self.yolo_loss=0 ### Tong: initilize yolo_loss
		self.isTrain = opt.isTrain
		# define tensors
		self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
								   opt.fineSize, opt.fineSize)
		self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
								   opt.fineSize, opt.fineSize)

		# load/define networks
		#Temp Fix for nn.parallel as nn.parallel crashes oc calculating gradient penalty
		use_parallel = not opt.gan_type == 'wgan-gp'
		self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
									  opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids, use_parallel, opt.learn_residual)
		if self.isTrain:
			use_sigmoid = opt.gan_type == 'gan'
			self.netD = networks.define_D(opt.output_nc, opt.ndf,
										  opt.which_model_netD,
										  opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids, use_parallel)
		if not self.isTrain or opt.continue_train:
			self.load_network(self.netG, 'G', opt.which_epoch)
			if self.isTrain:
				self.load_network(self.netD, 'D', opt.which_epoch)

		if self.isTrain:
			self.fake_AB_pool = ImagePool(opt.pool_size)
			self.old_lr = opt.lr

			# initialize optimizers
			self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
												lr=opt.lr, betas=(opt.beta1, 0.999))
			self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
												lr=opt.lr, betas=(opt.beta1, 0.999))
												
			self.criticUpdates = 5 if opt.gan_type == 'wgan-gp' else 1
			
			# define loss functions
			self.discLoss, self.contentLoss = init_loss(opt, self.Tensor)

		print('---------- Networks initialized -------------')
		networks.print_network(self.netG)
		if self.isTrain:
			networks.print_network(self.netD)
		print('-----------------------------------------------')

	def set_input(self, input):
		AtoB = self.opt.which_direction == 'AtoB'
		input_A = input['A' if AtoB else 'B']
		input_B = input['B' if AtoB else 'A']
		self.input_A.resize_(input_A.size()).copy_(input_A)
		self.input_B.resize_(input_B.size()).copy_(input_B)
		self.image_paths = input['A_paths' if AtoB else 'B_paths']

	def forward(self):
		self.real_A = Variable(self.input_A)
		self.fake_B = self.netG.forward(self.real_A)
		self.real_B = Variable(self.input_B)

	# no backprop gradients
	def test(self):
		self.real_A = Variable(self.input_A, volatile=True)
		self.fake_B = self.netG.forward(self.real_A)
		self.real_B = Variable(self.input_B, volatile=True)

	# get image paths
	def get_image_paths(self):
		return self.image_paths

	def backward_D(self):
		self.loss_D = self.discLoss.get_loss(self.netD, self.real_A, self.fake_B, self.real_B) ### gp included 
		#print ('Im here')
		self.loss_D.backward()


	def convert2RGB(self,tensor):
		### Tong: This function convert pytroch tensor[1,3,w,h] to [w,h,3]
		### input tensor: [1,3,w,h] pytoch tensor 
		### return [w,h,3] numpy array normalized in btw [0,1]
		
		# R=tensor.data.cpu().numpy()[:,0,:,:]		
		# G=tensor.data.cpu().numpy()[:,1,:,:]
		# B=tensor.data.cpu().numpy()[:,2,:,:]
		# height=R.shape[1]
		# width=R.shape[2]
		# R=np.reshape(R,(height,width))
		# G=np.reshape(G,(height,width))
		# B=np.reshape(B,(height,width))
		# RGB=np.dstack((B,G,R))
		# Max=np.amax(RGB)
		# #print Max
		# Min=np.amin(RGB)
		# return np.true_divide(RGB-Min,Max-Min)

		############ batched output ##########
		data=tensor.data.cpu().numpy()
		Rs=data[:,0,:,:]		
		Gs=data[:,1,:,:]
		Bs=data[:,2,:,:]
		
		batchSize=Rs.shape[0]
		height=Rs.shape[1]
		width=Rs.shape[2]
		RGBs=[]
		for i in np.arange(batchSize):
			R=np.reshape(Rs[i,:,:],(height,width))
			G=np.reshape(Gs[i,:,:],(height,width))
			B=np.reshape(Bs[i,:,:],(height,width))  ### batchsize by m by n 
			RGB=np.dstack((B,G,R)) ### batch by m by n by 3
			Max=np.amax(RGB)
			#print Max.shape 
			#print Max
			Min=np.amin(RGB)
			RGB=np.true_divide(RGB-Min,Max-Min)
			RGBs.append(RGB)
		return np.array(RGBs)




	def backward_G(self):
		self.loss_G_GAN = self.discLoss.get_g_loss(self.netD, self.real_A, self.fake_B)
		#print ("[conditional_gan_model]:",self.real_B)
		#print ("[conditional_gan_model]:",self.fake_B)

		### add yolo loss ####
		self.yolo_loss_scale=100  ### scale for Yolo Tong Lin, 100 is comparable to initial G loss  
		self.G_loss_scale=1 ### scale for G_loss Tong Lin


		#cv2.imshow("img",real_B)
		#cv2.waitKey(0)
		real_B=self.convert2RGB(self.real_B)
		#print real_B.shape
		fake_B=self.convert2RGB(self.fake_B)
		### for batched data ######
		#real_B=np.reshape(real_B,(1,256,256,3))		
		#fake_B=np.reshape(fake_B,(1,256,256,3))
		#print fake_B.shape()
		# cv2.imshow("img",real_B[0,:,:,:])
		# cv2.waitKey(0)

		# cv2.imshow("img",real_B[1,:,:,:])
		# cv2.waitKey(0)
		
		# cv2.imshow("img",real_B[2,:,:,:])
		# cv2.waitKey(0)


		# cv2.imshow("img",real_B[3,:,:,:])
		# cv2.waitKey(0)

		## add yolo loss ##
		loss_list, bbox_loss_list, iou_loss_list, cls_loss_list=get_yolo_loss(fake_B,real_B)
		#yolo_loss=torch.cuda.FloatTensor(loss_list)
		self.yolo_loss=loss_list
		#print "[conditional_gan_model.py]: loss_yolo",yolo_loss*yolo_loss_scale
		# Second, G(A) = B
		self.loss_G_Content = self.contentLoss.get_loss(self.fake_B, self.real_B) * self.opt.lambda_A


		self.loss_G = self.loss_G_GAN + self.loss_G_Content

		print loss_list,self.loss_G.data.cpu() 
		#print "[conditional_gan_model.py]: loss_g", self.loss_G*G_loss_scale
		

		self.loss_G= torch.add(self.G_loss_scale*self.loss_G,self.yolo_loss_scale*self.yolo_loss)/2 ## add yolo loss to loss_G
		#print "[conditional_gan_model.py]: loss_g+yolo_loss",self.loss_G
		self.loss_G.backward()


	def get_yolo_loss(self):
		### Tong: this function returns yolo_loss 
		return self.yolo_loss


	def optimize_parameters(self): ## Tong Lin: forward and backward 
		self.forward()

		for iter_d in xrange(self.criticUpdates):
			self.optimizer_D.zero_grad()
			self.backward_D()
			self.optimizer_D.step()

		self.optimizer_G.zero_grad()
		self.backward_G()
		self.optimizer_G.step()

	def get_current_errors(self):
		return OrderedDict([('G_GAN', self.loss_G_GAN.data[0]),
							('G_L1', self.loss_G_Content.data[0]),
							('D_real+fake', self.loss_D.data[0])
							])

	def get_current_visuals(self):
		real_A = util.tensor2im(self.real_A.data)
		fake_B = util.tensor2im(self.fake_B.data)
		real_B = util.tensor2im(self.real_B.data)
		return OrderedDict([('Blurred_Train', real_A), ('Restored_Train', fake_B), ('Sharp_Train', real_B)])

	def save(self, label):
		self.save_network(self.netG, 'G', label, self.gpu_ids)
		self.save_network(self.netD, 'D', label, self.gpu_ids)

	def update_learning_rate(self):
		lrd = self.opt.lr / self.opt.niter_decay
		lr = self.old_lr - lrd
		for param_group in self.optimizer_D.param_groups:
			param_group['lr'] = lr
		for param_group in self.optimizer_G.param_groups:
			param_group['lr'] = lr
		print('update learning rate: %f -> %f' % (self.old_lr, lr))
		self.old_lr = lr
