#### Make Data #### 
** To make data(blur_shape_image pair) I edited the script, pls use the edited version**
1. Put sharp imgs into folder A 
2. Put corresponding blur imgs into folder B 
3. run ../dataset/combine_A_and_B.py, make sure that you set destination folder; make sure that the last level of your folder strcture is called "train" 

#### run training #### 
python3 ../train.py --dataroot /home/foredawnlin/data/blur_sharp/DeblurGAN_blur/blurred_sharp/blur_sharp_pair --learn_residual --resize_or_crop resize_and_crop --checkpoints_dir train --name trial --batchSize 1 --fineSize 256 
1. checkpoints_dir refers to the directory where you run train.py. Thus setting it as train means that you will save models in currentDir/train  



#### selective scripts explanation for combining YOLO #### 
** train.py ** 
1. def train: this function trains our neural network. You can change display frequency here. (including erro, epoches, and displays in local server)


** data/aligned_dataset.py (for manipulate image input) **
1.  def __getitem__: in this function, you can change the input image size, choose whether crop or not, find the location of random crop in image 
2. In data/custom_dataset_data_loader.py: 
	def CreateDataset: three options correspond to three scripts; each of the scripts has similar structure as data/aligned_dataset.py  

** In models/conditional_gan_model.py(for combining losses) **
1. def backward_D (~line 89): total loss for D network 
2. def backward_G(~line 94): total loss for G network (add Yolo loss here, fake_B is generated fake data, real_B is real data, real_A not used )
3. def optimize_parameters(~line 103): self.criticUpdates: controls the frequency to train D network per G network 
gedit 
1. The two scripts contains cmd line options for training 
2. Change batchSize in base_options.py 
3. Change display frequencies (errors, epoch, etc) in train_options.py
4. Do not change loadSizeX and loadSizeY, reseiz_or_crop, they are simply not used in the codes. Mange input image size in data/aligned_dataset.py    




## Change in models/conditional_gan_model.py ##
1. added a function that converts pytorch tensor to RGB numpy array to pass into get_yolo_loss 
2. add function "get_yolo_loss" in backward_G in btw line ~100  



## Changes in YOLOV2 Pytorch version ## (OJAS)
Major change in darknet.py/forward
1. Added 3 more return arguments and class attributes (one for each loss)
2. (For future reference) Check **here** for nonetype error (while adding self.losses) when doing backward pass
3. commented out lines 117/118

(A) GANs is passing an BGR image to the get_yolo_loss function
1. refer to diff_temp on email to add for loop when inputs are in batches (Try with pool.imap)
2. Changes in yolo.py/preprocess_test -- Removed imread
3. Changes in network.py/np_to_variable (No change)

(B) GANs is passing a Variable wrapper image to get_yolo_loss function 
1.	diff_temp.py line 21 changes to image_var.data
2.	(Need not be necessary if you keep np_to_variable line) diff_temp.py line 24 changes to torch.toTensor(im_data) 
3. Replace np_to_variable with wrapper torch.autograd.Variable().cuda()
4. **Major changes** will be required in resizing as input is now a tensor not an image


Approach 1: 

Input full size image (No random crops)
YOLO pre-trained

Approach 2:

Random crops
YOLO pre-trained
Detection Loss: Ground truth = YOLO(cropped image), train_value=YOLO(Generated cropped image). Loss=f(Ground truth, train_value)

Approach 3:

Random crops
Prioritizing (having a different probability distribution for selection of offset coordinates for every image), f_1 -> (x_offset,y_offset) based on P(x,y). it gets reinforced every iteration of image.
YOLO pre-trained
Detection Loss: Ground truth = YOLO(cropped image), train_value=YOLO(Generated cropped image). Loss=f(Ground truth, train_value)

Approach 4:

Random crops
YOLO pre-trained
Detection_Loss= f_loss(YOLO(Stitch Random crop back to original image),YOLO(Ground truth)) 
Add priority from Approach 3

Approach 5:

Random crops
YOLO training 
(mid-term paper Approach) 
