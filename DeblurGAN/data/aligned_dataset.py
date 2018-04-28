import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image


class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        print ("dir_AB:",self.dir_AB)
        print (1)
        self.AB_paths = sorted(make_dataset(self.dir_AB))

        #assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)
        #print ('Im here')

    def __getitem__(self, index):
        #print ('111111111111111111111111111111111111111111111111111111111111111111')
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')

        #print ('[aligned_dataset]: w',w,'h',h)

        ######## Tong added: Keep original img size to train #############        
        if self.opt.resize_or_crop=='None':
            #print ('Im here')

            #AB = AB.resize((self.opt.loadSizeX * 2, self.opt.loadSizeY), Image.BICUBIC)
            AB = self.transform(AB)
            w_total = AB.size(2)
            w = int(w_total / 2)
            h = AB.size(1)
            A=AB[:,0:h,0:w]
            B=AB[:,0:h,w:w_total]
            ####### helper line ########
            #import numpy as np 
            #print (np.shape(A), np.shape(B))
            #### helper line ends ########
        ######## Ends ###################
        else: 
            AB = AB.resize((self.opt.loadSizeX * 2, self.opt.loadSizeY), Image.BICUBIC)
            AB = self.transform(AB)

            w_total = AB.size(2)
            w = int(w_total / 2)
            h = AB.size(1)   
            w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1)) ### Tong Lin: find left corner point of random crop 
            h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1)) ### Tong Lin: find top corner point of random crop

            A = AB[:, h_offset:h_offset + self.opt.fineSize,
                   w_offset:w_offset + self.opt.fineSize]
            B = AB[:, h_offset:h_offset + self.opt.fineSize,
                   w + w_offset:w + w_offset + self.opt.fineSize]

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)

        return {'A': A, 'B': B,
                'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        #print ('im here 2')
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'
