## train DeblurGAN _default data ##
#python3 ../train.py --dataroot /home/foredawnlin/data/blur_sharp/DeblurGAN_blur/blurred_sharp/blur_sharp_pair --learn_residual --resize_or_crop None --checkpoints_dir train --#name trial --batchSize 1 #--fineSize 256 --loadSizeX 720 --loadSizeY 360 


## train pascal_voc_origianl ##
#python ../train.py --dataroot /home/foredawnlin/data/blur_sharp/DeblurGAN_blur/blurred_sharp_voc_2007/blur_sharp_pair --learn_residual --resize_or_crop None --checkpoints_dir #train --name trial --batchSize 1 #--continue_train #--fineSize 256 --loadSizeX 720 --loadSizeY 360 

## train pascal_voc_augmented data ##
python ../train.py --dataroot /home/foredawnlin/data/blur_sharp/DeblurGAN_blur/blurred_sharp_voc_2007/blur_boxImg_pair --learn_residual --resize_or_crop None --checkpoints_dir #train --name trial --batchSize 1 #--continue_train #--fineSize 256 --loadSizeX 720 --loadSizeY 360 


## train pascal_voc_augmented data with random crop##
# python ../train.py --dataroot /home/foredawnlin/data/blur_sharp/DeblurGAN_blur/blurred_sharp_voc_2007/blur_boxImg_pair --learn_residual --resize_or_crop crop --checkpoints_dir train --name trial --batchSize 16 --fineSize 256 #--continue_train #--fineSize 256 --loadSizeX 720 --loadSizeY 360 

