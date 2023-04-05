import sys
import h5py
import numpy as np
from PIL import Image
import gc
import os
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, convert_rgb_to_y, calc_psnr, ssim

def save_h5(imgs_dir, phase, scale, patch_size, stride, batch_size = 100, cache_size=256):
    print('counting total patch number, please wait')
    total_patch_number = 0
    for img in sorted(os.listdir(imgs_dir)):
        hr = Image.open(os.path.join(imgs_dir, img)).convert('RGB')
        total_patch_number += ((hr.width // scale - patch_size) // stride + 1) * ((hr.height // scale - patch_size) // stride + 1)
    print('total patch number : ' + str(total_patch_number))

    h5f = h5py.File(r'./Datasets/{}_X{}.h5'.format(phase, scale), 'w')
    try:
        # train: 110550 for X4; 221487 for X3; 558822 for X2
        # valid: 14175 for X4; 28161 for X3; 71115 for X2
        # The first dimension of chunks should be an integral multiple of the batch_size during training
        hlset = h5f.create_dataset('lh', (total_patch_number, 2, patch_size, patch_size), maxshape=(None, 2, patch_size, patch_size), dtype='f', chunks = (cache_size, 2, patch_size, patch_size))
        # write to file by batch to avoid OOM
        idx = 0 # starting index of a batch
        patch_number = 0 # patch number in a batch
        total_number = 0 # total number to let me know when to stop training
        batch_size = batch_size # better be a factor of image number
        image_idx_of_batch = 0 # i
        batch = 0
        patches = [] # (patch_idx dimension, lr/hr Y_channel dimension, height dimension, width dimension)
        if len(os.listdir(imgs_dir)) % batch_size != 0:
            print('warning: ' + str(len(os.listdir(imgs_dir)) % batch_size) + \
                  ' images will not be used, check batch_size. len(): ' + str(len(os.listdir(imgs_dir))))
            return
        # maybe it is confusing, but sorted only called once
        print('batch ' + str(batch + 1) + ' processing')
        for img in sorted(os.listdir(imgs_dir)):
            hr = Image.open(os.path.join(imgs_dir, img)).convert('RGB')
            hr_width = hr.width
            hr_height = hr.height
            lr = hr.resize((hr_width // scale, hr_height // scale), resample=Image.BICUBIC)
            hr = np.array(hr).astype(np.float32)
            lr = np.array(lr).astype(np.float32)
            hr = convert_rgb_to_y(hr)
            lr = convert_rgb_to_y(lr)
            for x in range(0, lr.shape[0] - patch_size + 1, stride):
                for y in range(0, lr.shape[1] - patch_size + 1, stride):
                    # Add HR patch and LR patch to patches.
                    # Continued lr and hr are much more efficient for IO than seperate dataset I used before
                    patches.append([np.pad(lr[x // scale:x // scale + patch_size // scale, y // scale:y // scale + patch_size // scale],
                                           ((0, patch_size * (scale - 1) // scale),
                                           (0, patch_size * (scale - 1) // scale))),
                                    hr[x:x + patch_size, y:y + patch_size]])

            if image_idx_of_batch < batch_size - 1:
                image_idx_of_batch += 1
            # write to h5file by batch
            else:
                patch_number = len(patches)
                patches = np.array(patches)
                patches = patches / 255
                print(patches.shape)
                # shuffle all patches in the batch, thus batch_size should be high for a good shuffle
                shuffle_ix = np.random.permutation(np.arange(patch_number))
                patches = patches[shuffle_ix]
                hlset[idx : idx + patch_number] = patches

                del patches
                gc.collect()

                patches = []
                idx += patch_number
                total_number += patch_number
                patch_number = 0
                image_idx_of_batch = 0
                batch += 1
                print('batch:' + str(batch) + ' of ' + str(len(os.listdir(imgs_dir)) // batch_size))
        print(total_number)
    except BaseException as e:
        print(e)
    finally:
        h5f.close()