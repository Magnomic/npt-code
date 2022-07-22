import os.path
import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util
from PIL import Image


def normalize(img, min_val = 0, max_val = 1):
    return (img - np.min(img)) / (np.max(img) - np.min(img)) * (max_val - min_val) + min_val

class DatasetDnCNN(data.Dataset):
    """
    # -----------------------------------------
    # Get L/H for denosing on AWGN with fixed sigma.
    # Only dataroot_H is needed.
    # -----------------------------------------
    # e.g., DnCNN
    # -----------------------------------------
    """

    def __init__(self, opt):
        super(DatasetDnCNN, self).__init__()
        print('Dataset: Denosing on AWGN with fixed sigma. Only dataroot_H is needed.')
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.patch_size = opt['H_size'] if opt['H_size'] else 64
        self.sigma = opt['sigma'] if opt['sigma'] else 25
        self.sigma_test = opt['sigma_test'] if opt['sigma_test'] else self.sigma
        self.content_patch_size = opt['content_patch_size']
        self.style_patch_size = opt['style_patch_size']
        self.dncnn_train_patch_size = max(self.content_patch_size, self.style_patch_size)
        # ------------------------------------
        # get path of H
        # return None if input is None
        # ------------------------------------
        self.paths_L = util.get_image_paths(opt['dataroot_L'])
        self.paths_H = opt['dataroot_H']
#         print(self.paths_L)
#         self.paths_L_test = util.get_image_paths('/home/haiwen/noise_reduction/data/noise')

    def __getitem__(self, index):

        # ------------------------------------
        # get L image
        # ------------------------------------
        L_path = self.paths_L[index]
        img_L = np.load(L_path)
        img_L = Image.fromarray(img_L)
        img_L = normalize(img_L.resize((self.dncnn_train_patch_size, self.dncnn_train_patch_size), Image.BILINEAR),0,255)
        img_L = np.expand_dims(img_L, axis=2)
        
        high_sol_name = L_path.split('/')[7].split('_')[0]
        H_path = self.paths_H.format(high_sol_name)
        img_H = np.load(H_path)
        img_H = normalize(img_H[:self.content_patch_size, :self.content_patch_size], 0, 255)
        if img_H.shape[0] < self.dncnn_train_patch_size:
            img_H = Image.fromarray(img_H)
            img_H = img_H.resize((self.dncnn_train_patch_size, self.dncnn_train_patch_size), Image.BILINEAR)
        img_H = np.array(img_H)
        img_H = np.expand_dims(img_H, axis=2) # HxWx1
    
        if self.opt['phase'] == 'train':
            """
            # --------------------------------
            # get L/H patch pairs
            # --------------------------------
            """
            H, W, _ = img_H.shape

            # --------------------------------
            # randomly crop the patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            patch_L = img_L[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
#             print('img_H:{}'.format(img_H.shape))
#             print('img_L:{}'.format(img_L.shape))
#             print('H:{}'.format(patch_H.shape))
#             print('L:{}'.format(patch_L.shape))
            # --------------------------------
            # augmentation - flip, rotate
            # --------------------------------
            mode = random.randint(0, 7)
            patch_H = util.augment_img(patch_H, mode=mode)
            patch_L = util.augment_img(patch_L, mode=mode)
            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_H = util.uint2tensor3(patch_H)
            img_L = util.uint2tensor3(patch_L)
#             img_L = img_H.clone()
            # --------------------------------
            # add noise
            # --------------------------------
#             noise = torch.randn(img_L.size()).mul_(self.sigma/255.0)
#             img_L.add_(noise)

        else:
            """
            # --------------------------------
            # get L/H image pairs
            # --------------------------------
            """
            img_H = util.uint2single(img_H)
            img_L = util.uint2single(img_L)
#             img_L = np.copy(img_H)

            # --------------------------------
            # add noise
            # --------------------------------
#             np.random.seed(seed=0)
#             img_L += np.random.normal(0, self.sigma_test/255.0, img_L.shape)

            # --------------------------------
            # HWC to CHW, numpy to tensor
            # --------------------------------
            img_L = util.single2tensor3(img_L)
            img_H = util.single2tensor3(img_H)

        return {'L': img_L, 'H': img_H, 'H_path': H_path, 'L_path': L_path}

    def __len__(self):
        return len(self.paths_L)
