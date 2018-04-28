## cmd options ## 
1. all cmd options are in option/base_option.py and option/train_options.py 

## To make data(blur_shape_image pair) ##
1. Put sharp imgs into folder B 
2. Put corresponding blur imgs into folder A 
3. run ../datasets/combine_A_and_B.py, make sure that you set destination folder; make sure that the last level of your folder strcture is called "train" and put blur_shapr parin in the train
(example: python ../datasets/combine_A_and_B.py --fold_A /home/foredawnlin/data/blur_sharp/DeblurGAN_blur/blurred_sharp_voc_2007/blurred/ --fold_B /home/foredawnlin/data/blur_sharp/DeblurGAN_blur/blurred_sharp_voc_2007/boxImg/ --fold_AB /home/foredawnlin/data/blur_sharp/DeblurGAN_blur/blurred_sharp_voc_2007/blur_boxImg_pair/train/ )


## To run train.py ## 
1. start server: sudo python3 -m visdom.server  (-port portnum)  defaul 8097 
2. run train.sh 
3. "--resize or crop" to specify whether to train full size img; if not training full size img, need to specify scale factor and fineSize(crop Size)
**note** : you can specify location of saved models in train.sh; all cmd options refers to "cmd options"
4. in models/conditional_gan_model.py, change file path at line 14 to your own path to yolo2-pytorch 

## To use no_crop training and generation ## 
1. In model/network.py, line 148 to line 154, uncomment the lines 

## To continue training ##
1. In cmd opts, set "--continue_train" to true, it will find the model where u saved your model from train.py

## resize and random_crop ##
1. data/base_dataset.py  is unused 
2. in data/aligned_dataset.py line 29 to line 64: Things can be changed: image size to train; locate the crop in img and control crop size   
